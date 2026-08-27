import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from papers.AA_study.lib.amplitude_limitations import (  # noqa: E402
    reproduce_fig_1,
    reproduce_fig_2,
    reproduce_fig_3,
    reproduce_fig_4,
    reproduce_fig_5,
    reproduce_fig_7,
)
from papers.AA_study.lib.run_bas import run_bas  # noqa: E402
from papers.AA_study.utils.utils import (  # noqa: E402
    parse_args,
    parse_sample_size_per_class_to_test,
    str_to_bool,
)


def train_and_evaluate(cfg, run_dir: Path) -> None:
    """
    Dispatch experiment execution based on a config dictionary.

    Parameters
    ----------
    cfg : dict
        Experiment configuration values.
    run_dir : pathlib.Path
        Output directory for plots and artifacts.

    Returns
    -------
    None
    """
    exp_to_run = cfg.get("exp_to_run", "DEFAULT")
    generate_graph = not str_to_bool(cfg.get("dont_generate_graph", False))

    sample_size_per_class_to_test = parse_sample_size_per_class_to_test(
        cfg.get("bond_dimensions_to_test")
    )

    if exp_to_run == "DEFAULT":
        print("Running the DEFAULT (BAS) experiment")
        run_bas(
            batch_size=cfg.get("batch_size", 50),
            num_epochs=cfg.get("num_epochs", 20),
            classical_epochs=cfg.get("classical_epochs", 20),
            lr=cfg.get("lr", 0.01),
            run_dir=run_dir,
            generate_graph=generate_graph,
        )
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
    elif exp_to_run == "FIG5":
        print("Running the FIG5 experiment")
        reproduce_fig_5(
            num_max_samples=cfg.get("num_samples_per_class", 250), run_dir=run_dir
        )
    elif exp_to_run == "FIG7":
        print("Running the FIG7 experiment")
        reproduce_fig_7(
            dataset_to_run=cfg.get("dataset_to_run", "MNIST"),
            sample_size_per_class=sample_size_per_class_to_test or [1, 10, 100, 1000],
            batch_size=cfg.get("batch_size", 50),
            num_epochs=cfg.get("num_epochs", 20),
            lr=cfg.get("lr", 0.01),
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
        print("Running the BAS experiment")
        run_bas(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            classical_epochs=args.classical_epochs,
            lr=args.lr,
            generate_graph=not args.dont_generate_graph,
        )
    elif args.exp_to_run == "BAS":
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
    elif args.exp_to_run == "FIG5":
        print("Running the FIG5 experiment")
        reproduce_fig_5(num_max_samples=args.num_samples_per_class)
    elif args.exp_to_run == "FIG7":
        print("Running the FIG7 experiment")
        reproduce_fig_7(
            dataset_to_run=args.batch_size,
            sample_size_per_class=args.sample_size_per_class_to_test,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
        )
    else:
        raise NameError("No experiment with that name")


if __name__ == "__main__":
    main()
