#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

try:
    from papers.shared.QORC.datasets import (  # type: ignore
        get_dataloader,
        get_mnist_variant,
        seed_worker,
        split_fold_numpy,
        tensor_dataset,
    )
except ModuleNotFoundError:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from papers.shared.QORC.datasets import (  # type: ignore
        get_dataloader,
        get_mnist_variant,
        seed_worker,
        split_fold_numpy,
        tensor_dataset,
    )

__all__ = [
    "tensor_dataset",
    "seed_worker",
    "get_dataloader",
    "split_fold_numpy",
    "get_mnist_variant",
]
