"""
architecture_grid_run.py – benchmark of 9 re-uploading designs (metrics-only)
=============================================================================
Train **MerlinReuploadingClassifier** for every (data-block, var-block) pair
in {A,B,C}² and save metrics + figure data for later visualization.
"""

from __future__ import annotations

import itertools
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


def run_architecture_grid(cfg: dict[str, Any], run_dir: Path) -> None:
    """Run the architecture grid experiment using configuration."""
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

    depths = cfg["experiment"].get("depths", range(1, 10))
    depths = [int(depth) for depth in depths]
    alpha = cfg["experiment"].get("alpha", np.pi / 10)
    resolution = cfg["experiment"].get("grid_resolution", 220)
    grid_spec = compute_grid_spec(X_tr, X_te, resolution)

    hyperparams = {
        "max_epochs": cfg["training"]["epochs"],
        "learning_rate": cfg["training"]["lr"],
        "batch_size": cfg["dataset"]["batch_size"],
        "patience": cfg["training"].get("patience", 1000),
        "tau": cfg["experiment"].get("tau", 1.0),
        "track_history": False,
    }

    designs = cfg["experiment"].get(
        "designs", [a + b for a, b in itertools.product("ABC", repeat=2)]
    )
    metrics: list[dict[str, Any]] = []
    figure_data: dict[str, np.ndarray] = {}

    for depth in depths:
        logger.info("Processing depth %s...", depth)
        for idx, design in enumerate(designs):
            logger.info(
                "  Training design %s (%s/%s)...", design, idx + 1, len(designs)
            )
            model = MerlinReuploadingClassifier(
                dimension=X_tr.shape[1], num_layers=depth, design=design, alpha=alpha
            )
            model.fit(X_tr, y_tr, **hyperparams)

            train_acc = float(model.score(X_tr, y_tr))
            test_acc = float(model.score(X_te, y_te))
            metrics.append(
                {
                    "depth": depth,
                    "design": design,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                }
            )

            key_suffix = f"L{depth}_{design}"
            figure_data[f"diff_{key_suffix}"] = compute_decision_grid(model, grid_spec)
            feat_tr, feat_te = compute_probability_features(model, X_tr, X_te)
            figure_data[f"feat_tr_{key_suffix}"] = feat_tr
            if feat_te is not None:
                figure_data[f"feat_te_{key_suffix}"] = feat_te

    figure_path = run_dir / "design_benchmark_figure_data.npz"
    np.savez_compressed(figure_path, **figure_data)

    results = {
        "experiment": "design_benchmark",
        "dataset": dataset_name,
        "depths": depths,
        "designs": designs,
        "grid": {
            "resolution": grid_spec.resolution,
            "xlim": list(grid_spec.xlim),
            "ylim": list(grid_spec.ylim),
        },
        "metrics": metrics,
        "figure_data_path": str(figure_path),
        "config": cfg,
    }
    results_file = run_dir / "design_benchmark_results.json"
    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info("Saved results to: %s", results_file)
    logger.info("Saved figure data to: %s", figure_path)


__all__ = ["run_architecture_grid"]
