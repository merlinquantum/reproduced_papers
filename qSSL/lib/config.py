import json
from pathlib import Path


def load_config(path: Path):
    ext = path.suffix.lower()
    with path.open("r") as f:
        if ext == ".json":
            return json.load(f)
        raise ValueError(f"Unsupported config extension (use JSON): {ext}")


def deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def default_config():
    return {
        "seed": 42,
        "outdir": "outdir",
        "device": "cpu",
        "dataset": {"root": "./data", "classes": 2, "batch_size": 128},
        "model": {
            "backend": "classical",  # classical | qiskit | merlin
            "width": 8,
            "loss_dim": 128,
            "batch_norm": False,
            "temperature": 0.07,
            "layers": 2,
            "encoding": "vector",
            "q_ansatz": "sim_circ_14_half",
            "q_sweeps": 1,
            "activation": "null",
            "shots": 100,
            "modes": 10,
            "no_bunching": False,
        },
        "training": {"epochs": 2, "ckpt_step": 1, "le_epochs": 100},
        "logging": {"level": "info"},
    }
