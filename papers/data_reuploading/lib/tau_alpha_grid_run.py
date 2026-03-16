"""
tau_alpha_grid_run.py – benchmark of τ/α hyper-parameters (metrics-only)
=======================================================================
Train **MerlinReuploadingClassifier** on a Cartesian grid of (τ, α) values
and save metrics + figure data for later visualization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_artifacts import (
    compute_decision_grid,
    compute_grid_spec,
    compute_probability_features,
)
from .paper_datasets import CirclesDataset, MoonsDataset
from .reuploading_experiment import MerlinReuploadingClassifier


def run_tau_alpha_grid(cfg: dict[str, Any], run_dir: Path) -> None:
    """Run the tau/alpha grid experiment using configuration."""
    logger = logging.getLogger(__name__)

    dataset_name = cfg["experiment"]["dataset"]
    train_size = cfg["dataset"]["train_size"]
    test_size = cfg["dataset"]["test_size"]
    if dataset_name == "circles":
        dataset = CirclesDataset(n_train=train_size, n_test=test_size)
    elif dataset_name == "moons":
        dataset = MoonsDataset(n_train=train_size, n_test=test_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X_tr, y_tr = dataset.train
    X_te, y_te = dataset.test

    depths = cfg["experiment"].get("depths", range(2, 5))
    depths = [int(depth) for depth in depths]
    list_tau = cfg["experiment"].get("tau_values", np.logspace(-3, 2, 6))
    list_alpha = cfg["experiment"].get(
        "alpha_values", np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]) * np.pi
    )
    list_tau = [float(val) for val in list_tau]
    list_alpha = [float(val) for val in list_alpha]
    resolution = cfg["experiment"].get("grid_resolution", 220)
    grid_spec = compute_grid_spec(X_tr, X_te, resolution)

    hyperparams = {
        "max_epochs": cfg["training"]["epochs"],
        "learning_rate": cfg["training"]["lr"],
        "batch_size": cfg["dataset"]["batch_size"],
        "patience": cfg["training"].get("patience", 1000),
        "track_history": False,
    }

    metrics: list[dict[str, Any]] = []
    figure_data: dict[str, np.ndarray] = {}

    for depth in depths:
        logger.info("Processing depth %s...", depth)
        total_models = len(list_tau) * len(list_alpha)
        model_count = 0

        for tau in list_tau:
            for alpha in list_alpha:
                model_count += 1
                logger.info(
                    "  Training model %s/%s: tau=%.3f, alpha=%.3fπ",
                    model_count,
                    total_models,
                    tau,
                    alpha / np.pi,
                )
                model = MerlinReuploadingClassifier(
                    dimension=X_tr.shape[1],
                    num_layers=depth,
                    design="AA",
                    alpha=alpha,
                )
                model.fit(X_tr, y_tr, tau=tau, **hyperparams)

                train_acc = float(model.score(X_tr, y_tr))
                test_acc = float(model.score(X_te, y_te))
                metrics.append(
                    {
                        "depth": depth,
                        "tau": tau,
                        "alpha": alpha,
                        "train_accuracy": train_acc,
                        "test_accuracy": test_acc,
                    }
                )

                key_suffix = f"L{depth}_tau{tau:.6f}_alpha{alpha:.6f}"
                figure_data[f"diff_{key_suffix}"] = compute_decision_grid(
                    model, grid_spec
                )
                feat_tr, feat_te = compute_probability_features(model, X_tr, X_te)
                figure_data[f"feat_tr_{key_suffix}"] = feat_tr
                if feat_te is not None:
                    figure_data[f"feat_te_{key_suffix}"] = feat_te

    figure_path = run_dir / "tau_alpha_benchmark_figure_data.npz"
    np.savez_compressed(figure_path, **figure_data)

    results = {
        "experiment": "tau_alpha_benchmark",
        "dataset": dataset_name,
        "depths": depths,
        "tau_values": list_tau,
        "alpha_values": list_alpha,
        "grid": {
            "resolution": grid_spec.resolution,
            "xlim": list(grid_spec.xlim),
            "ylim": list(grid_spec.ylim),
        },
        "metrics": metrics,
        "figure_data_path": str(figure_path),
        "config": cfg,
    }
    results_file = run_dir / "tau_alpha_benchmark_results.json"
    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info("Saved results to: %s", results_file)
    logger.info("Saved figure data to: %s", figure_path)


__all__ = ["run_tau_alpha_grid"]
