
# from util import *


# # --- 4. LANCEMENT DU CODE ---
# # N'oubliez pas que votre MappingModel doit sortir au moins 8 valeurs
# final_output_size = 8

# model = HybridQMLModel(
#     q_input_size=2, q_output_size=4, nb_photons=1, nb_modes=2, 
#     hidden_sizes=[16, 16], final_output_size=8
# )
# train_cartpole_qtrl(model, num_episodes=1500)

import argparse
from util import *

if __name__ == "__main__":
    # 1. Initialisation du parseur
    parser = argparse.ArgumentParser(description="Entraînement Quantum-Train RL sur CartPole")

    # 2. Définition des arguments avec leurs valeurs par défaut
    parser.add_argument("--q_input_size", type=int, default=2, 
                        help="Taille de l'entrée quantique (défaut: 2)")
    
    parser.add_argument("--q_output_size", type=int, default=4, 
                        help="Taille de la sortie quantique intermédiaire (défaut: 4)")
    
    parser.add_argument("--nb_photons", type=int, default=1, 
                        help="Nombre de photons (défaut: 1)")
    
    parser.add_argument("--nb_modes", type=int, default=2, 
                        help="Nombre de modes du circuit (défaut: 2)")
    
    # nargs='+' permet d'accepter une liste de plusieurs valeurs séparées par un espace
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[16, 16], 
                        help="Tailles des couches cachées (ex: --hidden_sizes 32 16 8)")
    
    parser.add_argument("--final_output_size", type=int, default=8, 
                        help="Taille de la sortie finale (défaut: 8, requis pour CartPole)")
    
    parser.add_argument("--num_episodes", type=int, default=1500, 
                        help="Nombre d'épisodes pour l'entraînement RL (défaut: 1500)")

    # 3. Récupération des arguments passés dans le terminal
    args = parser.parse_args()

    print("=== Configuration de l'entraînement ===")
    print(f"Paramètres : {vars(args)}")
    print("=======================================\n")

    # 4. Injection des arguments dans le modèle
    model = HybridQMLModel( 
        q_output_size=args.q_output_size, 
        nb_photons=args.nb_photons, 
        nb_modes=args.nb_modes, 
        hidden_sizes=args.hidden_sizes, 
        final_output_size=args.final_output_size
    )

    # 5. Lancement de l'entraînement
    train_cartpole_qtrl(model, num_episodes=args.num_episodes)