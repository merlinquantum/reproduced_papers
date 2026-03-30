"""Datasets and loaders for MNIST and CIFAR-10 (delegates to shared module)."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from papers.shared.QRKD.datasets import (  # type: ignore
        DEFAULT_DATA_ROOT,
        DataConfig,
        cifar10_loaders,
        mnist_loaders,
    )
except ModuleNotFoundError:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from papers.shared.QRKD.datasets import (  # type: ignore
        DEFAULT_DATA_ROOT,
        DataConfig,
        cifar10_loaders,
        mnist_loaders,
    )

__all__ = ["DataConfig", "DEFAULT_DATA_ROOT", "mnist_loaders", "cifar10_loaders"]
