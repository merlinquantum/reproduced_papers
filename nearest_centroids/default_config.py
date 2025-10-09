# information.py

import json
from pathlib import Path
from copy import deepcopy


def default_config() -> dict:
    """Return the default configuration dictionary."""
    return {
        "seed": 42,
        "device": "cpu",
        "outdir": "./runs",
        "logging": {
            "level": "info",
        },
        "dataset": {
            "name": "mnist",
            "batch_size": 100,
            "n_components": 8,  # PCA components
            "transform_flatten": True,
        },
        "training": {
            "epochs": 10,
            "lr": 0.05,
            "optimizer": "adagrad",
            "n_repeats": 10,
        },
        "experiments": [],
    }


def load_config(path: Path) -> dict:
    """Load configuration from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge two dictionaries."""
    merged = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = deep_update(merged[k], v)
        else:
            merged[k] = deepcopy(v)
    return merged


# --------------------------------------
# Dataset / Experiment Information
# --------------------------------------

DATASETS = {
    "mnist": {
        "description": "MNIST handwritten digits dataset (10 classes, 28x28).",
        "n_classes": 10,
        "default_experiments": [
            {"classes": [0, 1], "n_samples": 40},
            {"classes": [2, 7], "n_samples": 40},
        ],
    },
    "iris": {
        "description": "Iris flower dataset (3 classes, 4 features).",
        "n_classes": 3,
        "default_experiments": [
            {"classes": [0, 1], "n_samples": 25},
            {"classes": [1, 2], "n_samples": 25},
        ],
    },
}
