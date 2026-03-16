"""Runtime entrypoints for the Data Re-uploading project."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .architecture_grid_run import run_architecture_grid
from .paper_datasets import (
    CirclesDataset,
    MoonsDataset,
    OverheadMNISTDataset,
    TetrominoDataset,
)
from .reuploading_experiment import MerlinReuploadingClassifier
from .tau_alpha_grid_run import run_tau_alpha_grid


def run_train_eval(cfg: dict[str, Any], run_dir: Path) -> None:
    """Train/evaluate across layer counts and save metrics for Figure 5."""
    logger = logging.getLogger(__name__)
    logger.info("Running train/eval sweep for Figure 5 metrics")

    dataset_name = cfg["experiment"]["dataset"]
    train_size = cfg["dataset"]["train_size"]
    test_size = cfg["dataset"]["test_size"]
    alpha = cfg["experiment"]["alpha"]
    tau = cfg["experiment"]["tau"]
    min_layers = cfg["experiment"].get("min_layers", 1)
    range_num_layers = range(min_layers, cfg["experiment"]["max_layers"] + 1)
    reps = cfg["experiment"]["repetitions"]

    training_params = {
        "max_epochs": cfg["training"]["epochs"],
        "learning_rate": cfg["training"]["lr"],
        "batch_size": cfg["dataset"]["batch_size"],
        "patience": cfg["training"]["patience"],
        "track_history": True,
    }

    all_train_accuracies: list[list[float]] = []
    all_test_accuracies: list[list[float]] = []
    logger.info(
        "Running %s layer configurations x %s repetitions",
        len(range_num_layers),
        reps,
    )

    for num_layers in range_num_layers:
        train_accuracies: list[float] = []
        test_accuracies: list[float] = []

        for r in range(reps):
            logger.info("Training model: layers=%s, rep=%s/%s", num_layers, r + 1, reps)

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
            else:  # pragma: no cover - config validation elsewhere
                raise ValueError(f"Unknown dataset: {dataset_name}")

            X_tr, y_tr = rep_dataset.train
            X_te, y_te = rep_dataset.test

            model = MerlinReuploadingClassifier(
                dimension=X_tr.shape[1],
                num_layers=num_layers,
                design="AA",
                alpha=alpha,
            )

            model.fit(X_tr, y_tr, tau=tau, **training_params)
            train_accuracies.append(model.score(X_tr, y_tr))
            test_accuracies.append(model.score(X_te, y_te))

        all_train_accuracies.append(train_accuracies)
        all_test_accuracies.append(test_accuracies)

    results = {
        "dataset": dataset_name,
        "range_num_layers": list(range_num_layers),
        "train_accuracies": all_train_accuracies,
        "test_accuracies": all_test_accuracies,
        "config": cfg,
    }
    results_file = run_dir / "train_eval_results.json"
    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info("Saved results to: %s", results_file)


def run_design_benchmark(cfg: dict[str, Any], run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Running architecture design benchmark")
    run_architecture_grid(cfg, run_dir)
    logger.info("Architecture design benchmark completed")


def run_tau_alpha_benchmark(cfg: dict[str, Any], run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Running tau/alpha parameter benchmark")
    run_tau_alpha_grid(cfg, run_dir)
    logger.info("Tau/alpha parameter benchmark completed")


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """Main entry point that routes to the appropriate experiment."""
    logger = logging.getLogger(__name__)

    experiment_type = cfg.get("experiment", {}).get("type", "train_eval")

    if experiment_type in {"train_eval", "figure_5"}:
        run_train_eval(cfg, run_dir)
    elif experiment_type == "design_benchmark":
        run_design_benchmark(cfg, run_dir)
    elif experiment_type == "tau_alpha_benchmark":
        run_tau_alpha_benchmark(cfg, run_dir)
    else:  # pragma: no cover - config validation elsewhere
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    logger.info("Experiment completed")


__all__ = [
    "train_and_evaluate",
    "run_train_eval",
    "run_design_benchmark",
    "run_tau_alpha_benchmark",
]
