"""MNIST data loading for LatentQGAN reproduction.

Provides a per-class loader and a normalised 28x28 [0,1] tensor stream.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


def _resolve_root(cfg: dict | None = None) -> str:
    """Resolve the shared repository data directory used by torchvision."""
    if cfg is not None and cfg.get("data_root"):
        return cfg["data_root"]
    env = os.environ.get("DATA_DIR")
    if env:
        return env
    # Default to <reproduced_papers>/data, shared by all reproductions.
    reproduced_papers_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return os.path.join(reproduced_papers_root, "data")


def load_mnist(cfg: dict | None = None, train: bool = True) -> torch.Tensor:
    """Return MNIST images as a (N, 1, 28, 28) float tensor in [0, 1] and labels."""
    root = _resolve_root(cfg)
    os.makedirs(root, exist_ok=True)
    tx = transforms.Compose([transforms.ToTensor()])  # already in [0, 1]
    ds = datasets.MNIST(root, train=train, download=True, transform=tx)
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = torch.tensor([ds[i][1] for i in range(len(ds))])
    return imgs, labels


def subset_by_class(imgs: torch.Tensor, labels: torch.Tensor, cls: int, n: int | None = None) -> torch.Tensor:
    """Return at most ``n`` images for a single class."""
    mask = labels == cls
    out = imgs[mask]
    if n is not None:
        out = out[:n]
    return out


def autoencoder_loader(imgs: torch.Tensor, batch_size: int = 20, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(imgs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def gan_loader(latent_rows: torch.Tensor, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    """latent_rows: (N, 5, 8) normalised rows; output a TensorDataset over flattened rows.

    For per-iteration training of QGAN we just iterate batch_size=1 samples.
    """
    ds = TensorDataset(latent_rows)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
