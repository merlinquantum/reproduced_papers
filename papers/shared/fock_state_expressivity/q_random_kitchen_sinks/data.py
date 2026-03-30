"""Shared moons dataset utilities for q_random_kitchen_sinks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from runtime_lib.data_paths import paper_data_dir
except ModuleNotFoundError:
    _REPO_ROOT = next(
        (p for p in Path(__file__).resolve().parents if (p / "runtime_lib").exists()),
        None,
    )
    if _REPO_ROOT and str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from runtime_lib.data_paths import paper_data_dir


def load_moons(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate, split, scale, and cache the moons dataset."""
    n_samples = int(cfg.get("n_samples", 200))
    noise = float(cfg.get("noise", 0.2))
    random_state = int(cfg.get("random_state", 42))
    test_prop = float(cfg.get("test_prop", 0.4))
    scaling = cfg.get("scaling", "Standard")

    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_prop, random_state=random_state
    )

    if scaling == "Standard":
        scaler = StandardScaler()
    elif scaling == "MinMax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scaling}")

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    raw_cache = cfg.get("cache_dir")
    if raw_cache:
        cache_dir = Path(raw_cache)
        if not cache_dir.is_absolute():
            cache_dir = (
                paper_data_dir("fock_state_expressivity")
                / "q_random_kitchen_sinks"
                / cache_dir
            )
    else:
        cache_dir = (
            paper_data_dir("fock_state_expressivity")
            / "q_random_kitchen_sinks"
            / "cache"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "moons_x_train.npy", x_train)
    np.save(cache_dir / "moons_x_test.npy", x_test)
    np.save(cache_dir / "moons_y_train.npy", y_train)
    np.save(cache_dir / "moons_y_test.npy", y_test)

    return x_train, x_test, y_train, y_test


def target_function(x):
    """Cosine-based target used for random feature fitting."""
    return np.sqrt(2) * np.cos(x)


__all__ = ["load_moons", "target_function"]
