"""Shared dataset utilities for QORC (MNIST variants via MerLin)."""

from __future__ import annotations

import sys
from pathlib import Path

from merlin.datasets.mnist_digits import (
    get_data_train_original as get_mnist_train,
    get_data_test_original as get_mnist_test,
)

from merlin.datasets.fashion_mnist import (
    get_data_train as get_fashion_mnist_train,
    get_data_test as get_fashion_mnist_test,
)

from merlin.datasets.k_mnist import (
    get_data_test as get_k_mnist_test,
    get_data_train as get_k_mnist_train
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from runtime_lib.data_paths import paper_data_dir
except Exception:  # pragma: no cover - allow offline reuse
    paper_data_dir = None

try:
    import merlin.datasets.utils as _datasets_utils
except Exception:  # pragma: no cover - optional merlin dependency
    _datasets_utils = None

def _data_root() -> Path:
    if not paper_data_dir:
        raise RuntimeError("Shared data resolver unavailable; DATA_DIR or runtime_lib required")
    return paper_data_dir("QORC")


_MERLIN_DATA_ROOT = _data_root()

if _datasets_utils:
    def _custom_data_dir() -> Path:
        return _MERLIN_DATA_ROOT

    _datasets_utils.get_venv_data_dir = _custom_data_dir  # type: ignore[attr-defined]


def get_mnist_variant(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        X_train, y_train, _ = get_mnist_train()
        X_test, y_test, _ = get_mnist_test()
    elif dataset_name == "k-mnist" or dataset_name == "kmnist":
        X_train, y_train, _ = get_k_mnist_train()
        X_test, y_test, _ = get_k_mnist_test()
    elif dataset_name == "fashion-mnist" or dataset_name == "fashion_mnist":
        X_train, y_train, _ = get_fashion_mnist_train()
        X_test, y_test, _ = get_fashion_mnist_test()
    else:
        raise ValueError(
            "Unknown dataset: {dataset_name}. Expected 'mnist', 'k-mnist', or 'fashion-mnist'."
        )

    return [X_train, y_train, X_test, y_test]


__all__ = [
    "get_mnist_variant",
]
