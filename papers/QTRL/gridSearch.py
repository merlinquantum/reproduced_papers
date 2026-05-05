import concurrent.futures
import itertools
import time
from util import train_cartpole_qtrl, HybridQMLModel 

# ==========================================
# 1. LA FONCTION EXÉCUTÉE PAR CHAQUE PROCESSUS
# ==========================================
def run_experiment(config):
    exp_id = config['id']
    params = config['params']
    
    print(f"Processus {exp_id} DÉMARRÉ | LR: {params['lr']}, Modes: {params['nb_modes']}, Photons: {params['nb_photons']}")
    
    # 1. Instanciation du modèle avec TOUS les paramètres dynamiques
    # Note : Je pars du principe que q_input_size = nb_modes.
    model = HybridQMLModel(
        q_output_size=params['q_output_size'], 
        nb_photons=params['nb_photons'], 
        nb_modes=params['nb_modes'], 
        hidden_sizes=params['hidden_sizes'], 
        final_output_size=params['final_output_size'] # Doit toujours valoir 8
    )
    
    # 2. Lancement de l'entraînement avec le lr et le num_episodes dynamiques
    # Assure-toi que ta fonction train_cartpole_qtrl accepte bien ces arguments !
    score = train_cartpole_qtrl(model, num_episodes=params['num_episodes'], learning_rate=params['lr'])
    
    return {
        "id": exp_id,
        "params": params,
        "score": score
    }


# ==========================================
# 2. LE CHEF D'ORCHESTRE (MAIN)
# ==========================================
if __name__ == "__main__":
    # -- A. Préparation de la grille de paramètres (Grid Search Complet) --
    
    # Remplis ces listes avec les valeurs que tu souhaites tester.
    # Attention à ne pas mettre trop de valeurs pour éviter de faire exploser ton processeur.
    q_output_sizes_list = [4, 8]
    nb_photons_list = [1, 2]
    nb_modes_list = [2, 4]
    hidden_sizes_list = [[8, 8], [16, 16], [32, 16]]
    num_episodes_list = [500, 1000]
    lrs_list = [0.01, 0.005]
    
    # Fixé à 8 par les contraintes mathématiques de CartPole
    final_output_sizes_list = [8] 
    
    # On crée TOUTES les combinaisons possibles avec itertools
    combinations = list(itertools.product(
        q_output_sizes_list, 
        nb_photons_list, 
        nb_modes_list, 
        hidden_sizes_list, 
        final_output_sizes_list, 
        num_episodes_list, 
        lrs_list
    ))
    
    # On formate les combinaisons dans notre liste d'expériences
    experiments = []
    for i, combo in enumerate(combinations):
        experiments.append({
            "id": i + 1,
            "params": {
                "q_output_size": combo[0],
                "nb_photons": combo[1],
                "nb_modes": combo[2],
                "hidden_sizes": combo[3],
                "final_output_size": combo[4],
                "num_episodes": combo[5],
                "lr": combo[6]
            }
        })

    print(f"🚀 Lancement du Grid Search : {len(experiments)} configurations générées.")
    start_time = time.time()

    results = []
    
    # -- B. Lancement du Multiprocessing --
    # max_workers=10 lance 10 modèles en même temps. 
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for result_data in executor.map(run_experiment, experiments):
            results.append(result_data)

    # -- C. Analyse et Output Final --
    results.sort(key=lambda x: x["score"], reverse=True)
    meilleur_modele = results[0]

    print(f"\n⏱️ Entraînement global terminé en {time.time() - start_time:.2f} secondes.")
    print("==================================================")
    print("🏆 Meilleur modèle trouvé avec les paramètres suivants :")
    for key, value in meilleur_modele['params'].items():
        print(f"   - {key} : {value}")
    print(f"🎯 Score obtenu : {meilleur_modele['score']}")
    print("==================================================")