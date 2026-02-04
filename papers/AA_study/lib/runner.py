import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.AA_study.utils.utils import str_to_bool, parse_args
from papers.AA_study.lib.run_bas import run_bas
from papers.AA_study.lib.amplitude_limitations import (
    reproduce_fig_1,
    reproduce_fig_2,
    reproduce_fig_3,
    reproduce_fig_4,
)


def train_and_evaluate(cfg, run_dir: Path) -> None:
    exp_to_run = cfg.get("exp_to_run", "DEFAULT")
    generate_graph = not str_to_bool(cfg.get("dont_generate_graph", False))

    if exp_to_run == "DEFAULT":
        print("Running the DEFAULT experiment")
        print("Not yet implemented")
    elif exp_to_run == "BAS":
        print("Running the BAS experiment")
        run_bas(
            batch_size=cfg.get("batch_size", 50),
            num_epochs=cfg.get("num_epochs", 20),
            classical_epochs=cfg.get("classical_epochs", 20),
            lr=cfg.get("lr", 0.01),
            run_dir=run_dir,
            generate_graph=generate_graph,
        )
    elif exp_to_run == "FIG1":
        print("Running the FIG1 experiment")
        reproduce_fig_1(
            num_max_samples=cfg.get("num_samples_per_class", 2000), run_dir=run_dir
        )
    elif exp_to_run == "FIG2":
        print("Running the FIG2 experiment")
        reproduce_fig_2(
            num_max_samples=cfg.get("num_samples_per_class", 2000), run_dir=run_dir
        )
    elif exp_to_run == "FIG3":
        print("Running the FIG3 experiment")
        reproduce_fig_3(
            num_max_samples=cfg.get("num_samples_per_class", 2000), run_dir=run_dir
        )
    elif exp_to_run == "FIG4":
        print("Running the FIG4 experiment")
        reproduce_fig_4(
            batch_size=cfg.get("batch_size", 50),
            num_epochs=cfg.get("num_epochs", 20),
            lr=cfg.get("lr", 0.01),
            num_samples_per_class=cfg.get("num_samples_per_class", 2000),
            run_dir=run_dir,
        )

    else:
        raise NameError("No experiment with that name")


def main():
    """
    Entry point for running the selected experiment.

    Returns
    -------
    None
    """
    args = parse_args()

    if args.exp_to_run == "DEFAULT":
        print("Running the DEFAULT experiment")
        print("Not yet implemented")
    elif args.exp_to_run == "BOND":
        print("Running the BAS experiment")
        run_bas(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            classical_epochs=args.classical_epochs,
            lr=args.lr,
            generate_graph=not args.dont_generate_graph,
        )
    elif args.exp_to_run == "FIG1":
        print("Running the FIG1 experiment")
        reproduce_fig_1(num_max_samples=args.num_samples_per_class)
    elif args.exp_to_run == "FIG2":
        print("Running the FIG2 experiment")
        reproduce_fig_2(num_max_samples=args.num_samples_per_class)
    elif args.exp_to_run == "FIG3":
        print("Running the FIG3 experiment")
        reproduce_fig_3(num_max_samples=args.num_samples_per_class)
    elif args.exp_to_run == "FIG4":
        print("Running the FIG4 experiment")
        reproduce_fig_4(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            num_samples_per_class=args.num_samples_per_class,
        )
    else:
        raise NameError("No experiment with that name")


if __name__ == "__main__":
    main()
