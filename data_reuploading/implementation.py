#!/usr/bin/env python3
"""
Data Re-uploading paper reproduction experiments.
- Main: reproduces Figure 5 from the paper (default)
- Benchmarks: architecture grid run (9 designs) and tau/alpha parameter grid on circles/moons datasets
- Loads configuration from JSON via --config
- Sets up logging, output directory, and config snapshot
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
from lib.architecture_grid_run import run_architecture_grid
from lib.config import deep_update, default_config, load_config
from lib.paper_datasets import (
    CirclesDataset,
    MoonsDataset,
    OverheadMNISTDataset,
    TetrominoDataset,
)
from lib.reuploading_experiment import MerlinReuploadingClassifier
from lib.tau_alpha_grid_run import run_tau_alpha_grid
from utils.utils import plot_figure_5

# -----------------------------
# Core placeholders
# -----------------------------


def setup_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    # Extend with torch / jax determinism if relevant


def reproduce_figure_5(cfg, run_dir: Path) -> None:
    """Reproduce Figure 5 from the paper - accuracy vs number of layers."""
    logger = logging.getLogger(__name__)
    logger.info("Reproducing Figure 5 from paper")

    # Get dataset hyperparameters
    dataset_name = cfg["experiment"]["dataset"]
    train_size = cfg["dataset"]["train_size"]
    test_size = cfg["dataset"]["test_size"]

    # Parameters from the paper
    alpha = cfg["experiment"]["alpha"]
    tau = cfg["experiment"]["tau"]
    try:
        min_layers = cfg["experiment"]["min_layers"]
    except KeyError:
        min_layers = 1
    range_num_layers = range(min_layers, cfg["experiment"]["max_layers"] + 1)
    reps = cfg["experiment"]["repetitions"]

    # Training parameters
    training_params = {
        "max_epochs": cfg["training"]["epochs"],
        "learning_rate": cfg["training"]["lr"],
        "batch_size": cfg["dataset"]["batch_size"],
        "patience": cfg["training"]["patience"],
        "track_history": True,
    }

    all_train_accuracies = []
    all_test_accuracies = []

    logger.info(
        f"Running {len(range_num_layers)} layer configurations x {reps} repetitions"
    )

    for num_layers in range_num_layers:
        train_accuracies = []
        test_accuracies = []

        for r in range(reps):
            logger.info(f"Training model: layers={num_layers}, rep={r + 1}/{reps}")

            # Recreate dataset for each repetition to get different random splits
            if dataset_name == "circles":
                rep_dataset = CirclesDataset(n_train=train_size, n_test=test_size)
            elif dataset_name == "moons":
                rep_dataset = MoonsDataset(n_train=train_size, n_test=test_size)
            elif dataset_name == "tetromino":
                rep_dataset = TetrominoDataset(n_train=train_size, n_test=test_size)
            elif dataset_name == "overhead":
                rep_dataset = OverheadMNISTDataset(
                    n_train=train_size,
                    n_test=test_size,
                    balanced=True,
                    root="data/overhead",
                )
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            X_tr, y_tr = rep_dataset.train
            X_te, y_te = rep_dataset.test

            model = MerlinReuploadingClassifier(
                dimension=X_tr.shape[1], num_layers=num_layers, design="AA", alpha=alpha
            )

            model.fit(X_tr, y_tr, tau=tau, **training_params)

            train_acc = model.score(X_tr, y_tr)
            test_acc = model.score(X_te, y_te)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            logger.info(f"  Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

        all_train_accuracies.append(train_accuracies)
        all_test_accuracies.append(test_accuracies)

    # Save results
    results = {
        "dataset": dataset_name,
        "range_num_layers": list(range_num_layers),
        "train_accuracies": all_train_accuracies,
        "test_accuracies": all_test_accuracies,
        "config": cfg,
    }

    results_file = run_dir / "figure_5_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Generate plot
    plot_figure_5(all_train_accuracies, all_test_accuracies, range_num_layers)
    import matplotlib.pyplot as plt

    plot_path = run_dir / f"figure_5_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved results to: {results_file}")
    logger.info(f"Saved plot to: {plot_path}")


def run_design_benchmark(cfg, run_dir: Path) -> None:
    """Run the architecture design benchmark (9 different designs)."""
    logger = logging.getLogger(__name__)
    logger.info("Running architecture design benchmark")

    run_architecture_grid(cfg, run_dir)

    logger.info("Architecture design benchmark completed")


def run_tau_alpha_benchmark(cfg, run_dir: Path) -> None:
    """Run the tau/alpha parameter benchmark."""
    logger = logging.getLogger(__name__)
    logger.info("Running tau/alpha parameter benchmark")

    run_tau_alpha_grid(cfg, run_dir)

    logger.info("Tau/alpha parameter benchmark completed")


def train_and_evaluate(cfg, run_dir: Path) -> None:
    """Main entry point that routes to the appropriate experiment."""
    logger = logging.getLogger(__name__)
    logger.info("Starting experiment")
    logger.debug("Resolved config: %s", json.dumps(cfg, indent=2))

    experiment_type = cfg.get("experiment", {}).get("type", "figure_5")

    if experiment_type == "figure_5":
        reproduce_figure_5(cfg, run_dir)
    elif experiment_type == "design_benchmark":
        run_design_benchmark(cfg, run_dir)
    elif experiment_type == "tau_alpha_benchmark":
        run_tau_alpha_benchmark(cfg, run_dir)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    logger.info("Experiment completed")


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Data Re-uploading paper reproduction runner"
    )
    p.add_argument("--config", type=str, help="Path to JSON config", default=None)
    p.add_argument("--seed", type=int, help="Random seed", default=None)
    p.add_argument("--outdir", type=str, help="Base output directory", default=None)
    p.add_argument(
        "--device", type=str, help="Device string (cpu, cuda:0, mps)", default=None
    )

    # Experiment type selection
    p.add_argument(
        "--design-benchmark",
        action="store_true",
        help="Run architecture design benchmark (9 different designs)",
    )
    p.add_argument(
        "--tau-alpha-benchmark",
        action="store_true",
        help="Run tau/alpha parameter grid benchmark",
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=["circles", "moons", "tetromino", "overhead"],
        help="Dataset to use (circles, moons, tetromino, or overhead)",
        default=None,
    )

    # Common training overrides
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--layers", type=int, help="Max number of layers for Figure 5", default=None
    )
    p.add_argument(
        "--repetitions", type=int, help="Number of repetitions", default=None
    )

    return p


def configure_logging(level: str = "info", log_file: Path | None = None) -> None:
    """Configure root logger with stream handler and optional file handler.

    Example usage:
        configure_logging("debug")
        logger = logging.getLogger(__name__)
        logger.info("Message")
    """
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    log_level = level_map.get(str(level).lower(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)
    # Reset handlers to avoid duplicates on reconfiguration
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def resolve_config(args: argparse.Namespace):
    cfg = default_config()

    # Add data reuploading specific defaults
    cfg["experiment"] = {
        "type": "figure_5",
        "dataset": "circles",
        "alpha": np.pi / 10,
        "tau": 1.0,
        "max_layers": 15,
        "repetitions": 5,
        "designs": "AA",
    }

    # Override training defaults for paper reproduction
    cfg["training"] = {
        "epochs": 10_000,
        "optimizer": "adam",
        "lr": 1e-3,
        "patience": 1_000,
        "weight_decay": 0.0,
    }

    cfg["dataset"]["batch_size"] = 400  # Full batch as in paper
    cfg["dataset"]["train_size"] = 400
    cfg["dataset"]["test_size"] = 100
    if args.dataset is not None:
        if args.dataset == "tetromino":
            cfg["dataset"]["train_size"] = 100
            cfg["dataset"]["test_size"] = 48
        elif args.dataset == "overhead":
            cfg["dataset"]["train_size"] = 1776
            cfg["dataset"]["test_size"] = 222
            cfg["experiment"]["max_layers"] = 4
            cfg["experiment"]["min_layers"] = 4
    cfg["outdir"] = "results"

    # Load from file if provided
    if args.config:
        file_cfg = load_config(Path(args.config))
        cfg = deep_update(cfg, file_cfg)

    # Determine experiment type from flags
    if args.design_benchmark:
        cfg["experiment"]["type"] = "design_benchmark"
    elif args.tau_alpha_benchmark:
        cfg["experiment"]["type"] = "tau_alpha_benchmark"

    # Apply CLI overrides
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.device is not None:
        cfg["device"] = args.device
    if args.dataset is not None:
        cfg["experiment"]["dataset"] = args.dataset
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["dataset"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.layers is not None:
        cfg["experiment"]["max_layers"] = args.layers
    if args.repetitions is not None:
        cfg["experiment"]["repetitions"] = args.repetitions

    return cfg


def main(argv: list[str] | None = None) -> int:
    # Ensure we operate from the template directory
    configure_logging("info")  # basic console logging before config is resolved
    script_dir = Path(__file__).resolve().parent
    if Path.cwd().resolve() != script_dir:
        logging.info("Switching working directory to %s", script_dir)
        os.chdir(script_dir)

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = resolve_config(args)
    setup_seed(cfg["seed"])

    # Prepare output directory with timestamped run folder
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out = Path(cfg["outdir"])
    run_dir = base_out / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging based on resolved config and add file handler in the run directory
    configure_logging(cfg.get("logging", {}).get("level", "info"), run_dir / "run.log")

    # Save resolved config snapshot
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2))

    # Execute training/eval pipeline
    train_and_evaluate(cfg, run_dir)

    logging.info("Finished. Artifacts in: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
