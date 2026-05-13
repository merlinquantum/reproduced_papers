import sys
from lib.util import HybridQMLModel, train_cartpole_qtrl, train_minigrid_qtrl

def train_and_evaluate(args, run_dir):
    
    print(f"==================================================")
    print(f"Initialisation : {args['env_name']}")
    print(f"Save directory : {run_dir}")
    print(f"==================================================")


    model = HybridQMLModel(
        q_output_size=args["q_output_size"],
        nb_photons=args["nb_photons"],
        nb_modes=args["nb_modes"],
        hidden_sizes=args["hidden_sizes"],
        final_output_size=args["final_output_size"]
    )

    if args["env_name"] == "CartPole":
        print("Lauching of CartPole...\n")
        train_cartpole_qtrl(model, num_episodes=args["num_episodes"], learning_rate=args["lr"])
        
    elif args["env_name"] == "MiniGrid":
        print("launching of miniGrid...\n")
        train_minigrid_qtrl(model, num_episodes=args["num_episodes"], learning_rate=args["lr"])
        
    else:
        print(f"Critical Error : unknown environment '{args['env_name']}'", file=sys.stderr)
        sys.exit(1)