import json
from pathlib import Path


def load_config(path: Path):
    """Load a JSON config file into a dict.

    Only JSON is supported in this template.
    """
    ext = path.suffix.lower()
    with path.open("r") as f:
        if ext == ".json":
            return json.load(f)
        raise ValueError(f"Unsupported config extension for template (use JSON): {ext}")


def deep_update(base, updates):
    """Recursively update dict `base` with `updates`. Returns the updated dict."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def default_config():
    """Return default configuration as a plain dictionary."""
    return {
        "seed": 42,
        "outdir": "outdir",
        "device": "cpu",
        "dataset": {
            "name": "<DATASET_NAME>",
            "root": "data",
            "split": "train",
            "batch_size": 32,
        },
        "model": {
            "name": "<MODEL_NAME>",
            "params": {"hidden_dim": 128, "layers": 2},
        },
        "training": {
            "epochs": 10,
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
        "logging": {"level": "info", "save_config_snapshot": True},
    }
