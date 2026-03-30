"""Dataset utilities for the Quantum Gaussian Kernel project (shared)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


@dataclass
class GaussianGrid:
    x: np.ndarray
    x_on_pi: np.ndarray
    delta: np.ndarray
    targets: list[np.ndarray]
    sigma_labels: list[str]
    sigma_values: list[float]


def build_gaussian_grid(cfg: dict) -> GaussianGrid:
    """Prepare the 1D grid used to fit Gaussian kernels."""
    step = float(cfg.get("step", 0.05))
    span = float(cfg.get("span", np.pi))
    center = float(cfg.get("center", 0.0))
    sigmas = cfg.get("sigmas", [1.0, 0.5, 0.33, 0.25])
    num_points = int(2 * span / step) + 1

    x = np.linspace(-span, span, num=num_points)
    x_on_pi = x / np.pi
    delta = (x - center) ** 2
    targets = [np.exp(-delta / (2 * sigma * sigma)) for sigma in sigmas]
    labels = [f"sigma={sigma:.2f}" for sigma in sigmas]

    return GaussianGrid(
        x=x,
        x_on_pi=x_on_pi,
        delta=delta,
        targets=targets,
        sigma_labels=labels,
        sigma_values=list(sigmas),
    )


def _generate_dataset(kind: str, num_samples: int, cfg: dict):
    if kind == "circular":
        noise = float(cfg.get("noise", 0.05))
        return make_circles(n_samples=num_samples, noise=noise, random_state=42)
    if kind == "moon":
        noise = float(cfg.get("noise", 0.2))
        return make_moons(n_samples=num_samples, noise=noise, random_state=42)
    if kind == "blob":
        centers = int(cfg.get("centers", 2))
        cluster_std = float(cfg.get("cluster_std", 4.0))
        return make_blobs(
            n_samples=num_samples,
            centers=centers,
            cluster_std=cluster_std,
            random_state=42,
        )
    raise ValueError(f"Unknown dataset type: {kind}")


def prepare_classification_data(cfg: dict) -> dict[str, dict[str, torch.Tensor]]:
    """
    Generate circle, moon, and blob datasets with standardization & caching.

    Args:
        cfg: {
            "datasets": {"circular": {...}, "moon": {...}, "blob": {...}},
            "num_samples": 200,
            "test_size": 0.4,
            "scaling_factor": 0.65
        }
    """
    raw_cache = cfg.get("cache_dir")
    if raw_cache:
        cache_dir = Path(raw_cache)
        if not cache_dir.is_absolute():
            cache_dir = (
                paper_data_dir("fock_state_expressivity")
                / "q_gaussian_kernel"
                / cache_dir
            )
    else:
        cache_dir = (
            paper_data_dir("fock_state_expressivity") / "q_gaussian_kernel" / "cache"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    force_regen = cfg.get("force_regenerate", False)
    test_size = float(cfg.get("test_size", 0.4))
    num_samples = int(cfg.get("num_samples", 200))
    scaling = float(cfg.get("scaling_factor", 0.65))

    datasets_cfg = cfg.get("datasets", {})
    payload: dict[str, dict[str, torch.Tensor]] = {}

    for name in ["circular", "moon", "blob"]:
        cache_path = cache_dir / f"{name}_dataset.pt"
        if cache_path.exists() and not force_regen:
            payload[name] = torch.load(cache_path)
            continue

        x, y = _generate_dataset(name, num_samples, datasets_cfg.get(name, {}))
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=42
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train) * scaling
        x_test = scaler.transform(x_test) * scaling

        tensors = {
            "x_train": torch.tensor(x_train, dtype=torch.float32),
            "x_test": torch.tensor(x_test, dtype=torch.float32),
            "y_train": torch.tensor(y_train, dtype=torch.float32),
            "y_test": torch.tensor(y_test, dtype=torch.float32),
        }
        torch.save(tensors, cache_path)
        payload[name] = tensors

    return payload


__all__ = [
    "GaussianGrid",
    "build_gaussian_grid",
    "prepare_classification_data",
]
