#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path
from lib.util import *

def train_and_evaluate(args, run_dir):
    """
    Fonction principale d'exécution.
    'args' est le dictionnaire final (defaults.json + surcharges du terminal).
    """
    run_mode = args.get("run_mode", "train")
    
    if run_mode == "gridsearch":
        print("🔍 Lancement du Grid Search...")
        # from .gridsearch import run_gridsearch
        # run_gridsearch(args, run_dir)
        return

    # Suite classique pour l'entraînement
    print(f"==================================================")
    print(f"🚀 Initialisation de l'expérience sur : {args['env_name']}")
    print(f"⚙️  Backend utilisé : {args.get('backend', 'Non défini')}")
    print(f"📂 Dossier de sauvegarde : {run_dir}")
    print(f"==================================================")
    
    # 1. Calcul des paramètres requis selon l'environnement
    if args["env_name"] == "CartPole":
        total_weights_needed = 4 * 2
    elif args["env_name"] == "MiniGrid":
        total_weights_needed = 147 * 3
    else:
        print(f"❌ Environnement inconnu : {args['env_name']}")
        sys.exit(1)
        
    # 2. Création du modèle via le polymorphisme
    model = create_hybrid_model(args, total_weights_needed)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"numbers of trainable parameters : {total_params}")

    # 3. Lancement de l'algorithme RL
    if args["env_name"] == "CartPole":
        train_cartpole_qtrl(model, num_episodes=args["num_episodes"], learning_rate=args["lr"])
    elif args["env_name"] == "MiniGrid":
        train_minigrid_qtrl(model, num_episodes=args["num_episodes"], learning_rate=args["lr"], seed=args.get("seed", 42))



# ==============================================================================
# MODE STANDALONE : EXÉCUTÉ UNIQUEMENT SI TU LANCES "python lib/runner.py"
# ==============================================================================
if __name__ == "__main__":
    set_global_seed(seed=42)
    
    # 1. CHARGEMENT DU FICHIER DE BASE (defaults.json)
    racine_projet = Path(__file__).resolve().parent.parent
    chemin_defaults = racine_projet / "configs" / "defaults.json"
    
    try:
        with open(chemin_defaults, "r", encoding="utf-8") as f:
            args_config = json.load(f)
            print(f"✅ Fichier de base chargé : {chemin_defaults}")
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier {chemin_defaults} est introuvable.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Erreur : Le fichier defaults.json est mal formaté.", file=sys.stderr)
        sys.exit(1)

    # 2. MINI-PARSEUR COMPLET (pour intercepter toutes tes commandes terminal)
    parser = argparse.ArgumentParser(description="Standalone Runner pour QTRL")
    
    # --- Variables de contrôle ---
    parser.add_argument("--env_name", type=str, choices=["CartPole", "MiniGrid"])
    parser.add_argument("--backend", type=str, choices=["merlin_mlp", "merlin_mps", "torchquantum"])
    parser.add_argument("--run_mode", type=str, choices=["train", "gridsearch"])
    
    # --- Variables d'entraînement ---
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_episodes", type=int)
    
    # --- Variables de l'architecture quantique (QLayer) ---
    parser.add_argument("--q_output_size", type=int)
    parser.add_argument("--nb_photons", type=int)
    parser.add_argument("--nb_modes", type=int)
    
    # --- Variables d'expressivité (Mapping) ---
    parser.add_argument("--hidden_sizes", type=int, nargs="+", help="Ex: --hidden_sizes 8 8")
    parser.add_argument("--bond_dim", type=int, help="Expressivité pour le modèle MPS")

    # 3. FUSION : TERMINAL > DEFAULTS.JSON
    cli_args = parser.parse_args()
    
    surcharges = []
    for key, value in vars(cli_args).items():
        # Si tu as tapé une valeur dans le terminal (value n'est pas None), on écrase le dictionnaire
        if value is not None:
            args_config[key] = value
            surcharges.append(f"{key}={value}")

    if surcharges:
        print(f"⚠️  Surcharges appliquées via CLI : {', '.join(surcharges)}")

    # 4. EXÉCUTION
    dossier_sauvegarde = "./runs/standalone_run"
    train_and_evaluate(args_config, dossier_sauvegarde)