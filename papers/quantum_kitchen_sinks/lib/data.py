"""Datasets for the Quantum Kitchen Sinks reproduction.

Implements:
- `picture_frames`: the synthetic two-class 2D dataset from Fig. 3 of arXiv:1806.08321.
- `mnist35`: the (3,5) subset of MNIST used in Fig. 5.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np


def _picture_frame_points(
    n_per_class: int,
    inner_radius: float,
    outer_radius: float,
    noise: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample 2-D points uniformly on two concentric square frames.

    Class 0 lives on the inner square frame (side = 2 * inner_radius).
    Class 1 lives on the outer square frame (side = 2 * outer_radius).
    """

    def _sample_frame(n: int, half_side: float) -> np.ndarray:
        # Each sample chooses a side of the square, then a position along it.
        side = rng.integers(0, 4, size=n)
        u = rng.uniform(-half_side, half_side, size=n)
        x = np.empty(n)
        y = np.empty(n)
        x[side == 0] = u[side == 0]
        y[side == 0] = half_side
        x[side == 1] = u[side == 1]
        y[side == 1] = -half_side
        x[side == 2] = half_side
        y[side == 2] = u[side == 2]
        x[side == 3] = -half_side
        y[side == 3] = u[side == 3]
        return np.stack([x, y], axis=1)

    inner = _sample_frame(n_per_class, inner_radius)
    outer = _sample_frame(n_per_class, outer_radius)
    inner = inner + rng.normal(scale=noise, size=inner.shape)
    outer = outer + rng.normal(scale=noise, size=outer.shape)
    X = np.concatenate([inner, outer], axis=0)
    y = np.concatenate(
        [np.zeros(n_per_class, dtype=np.int64), np.ones(n_per_class, dtype=np.int64)]
    )
    return X, y


def load_picture_frames(
    n_train: int = 1600,
    n_test: int = 400,
    inner_radius: float = 0.4,
    outer_radius: float = 0.7,
    noise: float = 0.02,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test) for the picture-frames task."""
    rng = np.random.default_rng(seed)
    X_train, y_train = _picture_frame_points(
        n_train // 2, inner_radius, outer_radius, noise, rng
    )
    X_test, y_test = _picture_frame_points(
        n_test // 2, inner_radius, outer_radius, noise, rng
    )
    # Shuffle deterministically
    train_perm = rng.permutation(X_train.shape[0])
    test_perm = rng.permutation(X_test.shape[0])
    return (
        X_train[train_perm].astype(np.float32),
        y_train[train_perm],
        X_test[test_perm].astype(np.float32),
        y_test[test_perm],
    )


def _mnist_root(data_root: str | os.PathLike) -> Path:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_mnist35(
    data_root: str | os.PathLike = "data",
    n_train: int | None = None,
    n_test: int | None = None,
    standardize: bool = True,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test) for the (3,5)-MNIST subset.

    Labels are remapped: 3 -> 0, 5 -> 1.

    `n_train` / `n_test` cap each set after filtering for digits 3 and 5; if
    `None` the full filtered subsets are returned.  Per-image standardization
    (mean 0, std 1 per image) follows the standard MNIST preprocessing for
    QKS in the paper ("After standardizing the image [42]...").
    """
    from torchvision import datasets, transforms

    root = _mnist_root(data_root)
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(
        str(root), train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        str(root), train=False, download=True, transform=transform
    )

    def _filter(ds) -> Tuple[np.ndarray, np.ndarray]:
        # `ds.data` is a uint8 tensor of shape (N, 28, 28); .targets is a tensor of ints.
        targets = ds.targets.numpy()
        mask = (targets == 3) | (targets == 5)
        X = ds.data.numpy()[mask].astype(np.float32) / 255.0
        y = targets[mask]
        y = np.where(y == 3, 0, 1).astype(np.int64)
        # Flatten and standardize per image
        X = X.reshape(X.shape[0], -1)
        if standardize:
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True) + 1e-8
            X = (X - mean) / std
        return X, y

    X_train, y_train = _filter(train_set)
    X_test, y_test = _filter(test_set)

    rng = np.random.default_rng(seed)
    train_perm = rng.permutation(X_train.shape[0])
    test_perm = rng.permutation(X_test.shape[0])
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]
    X_test = X_test[test_perm]
    y_test = y_test[test_perm]
    if n_train is not None:
        X_train = X_train[:n_train]
        y_train = y_train[:n_train]
    if n_test is not None:
        X_test = X_test[:n_test]
        y_test = y_test[:n_test]
    return X_train, y_train, X_test, y_test


def load_dataset(cfg, data_root: str | os.PathLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch on `cfg['dataset']['name']`."""
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    seed = int(cfg.get("seed", 0))
    if name == "picture_frames":
        return load_picture_frames(
            n_train=int(ds_cfg.get("n_train", 1600)),
            n_test=int(ds_cfg.get("n_test", 400)),
            inner_radius=float(ds_cfg.get("inner_radius", 0.4)),
            outer_radius=float(ds_cfg.get("outer_radius", 0.7)),
            noise=float(ds_cfg.get("noise", 0.02)),
            seed=seed,
        )
    if name == "mnist35":
        return load_mnist35(
            data_root=Path(data_root) / "MNIST_raw_cache",
            n_train=ds_cfg.get("n_train"),
            n_test=ds_cfg.get("n_test"),
            standardize=bool(ds_cfg.get("standardize", True)),
            seed=seed,
        )
    raise ValueError(f"Unknown dataset: {name}")


__all__ = ["load_picture_frames", "load_mnist35", "load_dataset"]
