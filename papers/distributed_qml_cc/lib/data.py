"""Synthetic 8-dimensional dataset for binary classification.

The dataset follows the construction of Appendix B in
Hwang et al., "Distributed quantum machine learning via classical
communication", arXiv:2408.16327. It is designed to have low linear
correlation between any individual attribute and the binary label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SyntheticDatasetConfig:
    n_dim: int = 8
    n_samples: int = 2048
    n_clusters: int = 32
    samples_per_cluster: int = 64
    sphere_radius: float = np.pi / 4
    train_fraction: float = 1536.0 / 2048.0
    seed: int = 42


def _sample_in_ball(rng: np.random.Generator, n_samples: int, n_dim: int, radius: float) -> np.ndarray:
    """Uniformly sample n_samples points inside an n_dim ball of given radius.

    Uses the standard exponential / normalize trick: a uniform direction is the
    normalised Gaussian, and the radial coordinate is r * u**(1/n) for u uniform.
    """
    directions = rng.standard_normal(size=(n_samples, n_dim))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * rng.uniform(0.0, 1.0, size=(n_samples, 1)) ** (1.0 / n_dim)
    return directions * radii


def build_synthetic_dataset(
    cfg: SyntheticDatasetConfig | dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate the synthetic dataset used in Fig. 4 / Table I of the paper.

    Returns
    -------
    x_train, y_train, x_val, y_val : np.ndarray
        Training and validation arrays. Features have shape ``(N, n_dim)`` and
        labels have shape ``(N,)`` with values in ``{-1, +1}``.
    info : dict
        Metadata about the generated dataset, including the maximum absolute
        Pearson correlation between any attribute and the labels.
    """
    if cfg is None:
        cfg = SyntheticDatasetConfig()
    elif isinstance(cfg, dict):
        cfg = SyntheticDatasetConfig(**cfg)

    rng = np.random.default_rng(cfg.seed)

    if cfg.n_clusters * cfg.samples_per_cluster != cfg.n_samples:
        raise ValueError(
            "n_clusters * samples_per_cluster must equal n_samples; "
            f"got {cfg.n_clusters} * {cfg.samples_per_cluster} != {cfg.n_samples}"
        )
    if cfg.n_clusters % 2 != 0:
        raise ValueError("n_clusters must be even for a balanced binary partition")

    base_points = _sample_in_ball(rng, cfg.n_samples, cfg.n_dim, cfg.sphere_radius)

    # Each of the 2^n_dim translation corners lies on a {-r, +r}^n_dim hypercube.
    # We pick cfg.n_clusters distinct corners uniformly at random and translate
    # samples_per_cluster points by each chosen corner.
    n_corners = 2**cfg.n_dim
    if cfg.n_clusters > n_corners:
        raise ValueError(
            f"n_clusters={cfg.n_clusters} exceeds the available {n_corners} corners"
        )
    chosen = rng.choice(n_corners, size=cfg.n_clusters, replace=False)
    bit_patterns = ((chosen[:, None] >> np.arange(cfg.n_dim)[None, :]) & 1).astype(np.float64)
    corner_vectors = (2.0 * bit_patterns - 1.0) * cfg.sphere_radius  # shape (n_clusters, n_dim)

    # Assign balanced labels to clusters: half +1 and half -1.
    label_pool = np.array([+1] * (cfg.n_clusters // 2) + [-1] * (cfg.n_clusters // 2))
    rng.shuffle(label_pool)

    x_all = np.empty((cfg.n_samples, cfg.n_dim), dtype=np.float64)
    y_all = np.empty(cfg.n_samples, dtype=np.int64)
    for c in range(cfg.n_clusters):
        start = c * cfg.samples_per_cluster
        end = start + cfg.samples_per_cluster
        x_all[start:end] = base_points[start:end] + corner_vectors[c]
        y_all[start:end] = label_pool[c]

    # Per-attribute Pearson correlation with the label.
    rho = np.array([
        np.corrcoef(x_all[:, i], y_all)[0, 1] for i in range(cfg.n_dim)
    ])
    info = {
        "n_samples": cfg.n_samples,
        "n_dim": cfg.n_dim,
        "n_clusters": cfg.n_clusters,
        "samples_per_cluster": cfg.samples_per_cluster,
        "sphere_radius": cfg.sphere_radius,
        "seed": cfg.seed,
        "pearson_per_attr": rho.tolist(),
        "pearson_max_abs": float(np.max(np.abs(rho))),
    }

    # Shuffle and split.
    perm = rng.permutation(cfg.n_samples)
    x_all = x_all[perm]
    y_all = y_all[perm]
    n_train = int(round(cfg.train_fraction * cfg.n_samples))
    x_train, x_val = x_all[:n_train], x_all[n_train:]
    y_train, y_val = y_all[:n_train], y_all[n_train:]
    return x_train, y_train, x_val, y_val, info


def build_dataloaders(
    cfg: SyntheticDatasetConfig | dict | None = None,
    batch_size: int = 512,
    dtype: torch.dtype = torch.float32,
) -> tuple[DataLoader, DataLoader, dict]:
    """Convenience wrapper building train/val PyTorch DataLoaders."""
    x_train, y_train, x_val, y_val, info = build_synthetic_dataset(cfg)
    train_ds = TensorDataset(
        torch.as_tensor(x_train, dtype=dtype),
        torch.as_tensor(y_train, dtype=dtype),
    )
    val_ds = TensorDataset(
        torch.as_tensor(x_val, dtype=dtype),
        torch.as_tensor(y_val, dtype=dtype),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, info


__all__ = [
    "SyntheticDatasetConfig",
    "build_synthetic_dataset",
    "build_dataloaders",
]
