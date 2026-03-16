"""Shared datasets/loaders for QRKD (MNIST/CIFAR-10)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from runtime_lib.data_paths import paper_data_dir
except Exception:  # pragma: no cover - runtime_lib may be absent in some contexts
    paper_data_dir = None


def _default_data_root() -> Path:
    if not paper_data_dir:
        raise RuntimeError(
            "Shared data resolver unavailable; DATA_DIR or runtime_lib required"
        )
    return paper_data_dir("QRKD")


DEFAULT_DATA_ROOT = _default_data_root()


@dataclass
class DataConfig:
    batch_size: int = 64
    num_workers: int = 0
    root: Path | str = DEFAULT_DATA_ROOT
    max_samples: int | None = None


def _resolve_root(root: Path | str) -> str:
    return str(Path(root).expanduser())


def _maybe_limit(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    limit = min(len(dataset), int(max_samples))
    from torch.utils.data import Subset  # local import to avoid global dependency

    return Subset(dataset, range(limit))


def mnist_loaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    data_root = _resolve_root(cfg.root)
    train = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    train = _maybe_limit(train, cfg.max_samples)
    test = _maybe_limit(test, cfg.max_samples)
    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, test_loader


def cifar10_loaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    data_root = _resolve_root(cfg.root)
    train = datasets.CIFAR10(data_root, train=True, download=True, transform=tfm)
    test = datasets.CIFAR10(data_root, train=False, download=True, transform=tfm)
    train = _maybe_limit(train, cfg.max_samples)
    test = _maybe_limit(test, cfg.max_samples)
    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, test_loader


__all__ = [
    "DataConfig",
    "DEFAULT_DATA_ROOT",
    "mnist_loaders",
    "cifar10_loaders",
]
