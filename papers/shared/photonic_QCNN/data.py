"""Shared dataset utilities for the photonic QCNN reproduction."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from runtime_lib.data_paths import paper_data_dir

    from . import paper, scratch
except ModuleNotFoundError:
    # Ensure repository root is on sys.path so paper modules can be imported.
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from runtime_lib.data_paths import paper_data_dir

    from . import paper, scratch

REPO_ROOT = Path(__file__).resolve().parents[2]

SHARED_DATA_DIR = paper_data_dir("photonic_QCNN")
LEGACY_DATA_DIR = REPO_ROOT / "photonic_QCNN" / "data"


def _ensure_shared_dir() -> Path:
    SHARED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return SHARED_DATA_DIR


def _ensure_shared_file(filename: str) -> Path:
    shared_path = _ensure_shared_dir() / filename
    legacy_path = LEGACY_DATA_DIR / filename

    if not shared_path.exists() and legacy_path.exists():
        shutil.copy2(legacy_path, shared_path)

    if not shared_path.exists():
        raise FileNotFoundError(
            f"Required dataset asset '{filename}' not found. Place it under {shared_path.parent} or provide it explicitly."
        )

    return shared_path


def _limit_samples(x_train, x_test, y_train, y_test, max_samples, random_state):
    if max_samples is None:
        return x_train, x_test, y_train, y_test

    rng = np.random.default_rng(random_state)

    def _sample_split(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) <= max_samples:
            return x, y
        indices = rng.permutation(len(x))[:max_samples]
        return x[indices], y[indices]

    x_train, y_train = _sample_split(x_train, y_train)
    x_test, y_test = _sample_split(x_test, y_test)
    return x_train, x_test, y_train, y_test


def get_dataset(
    dataset_name: str, source: str, random_state: int, max_samples: int | None = None
):
    """
    Load train/test splits for the requested dataset.

    Args:
        dataset_name: One of {"BAS", "Custom BAS", "MNIST"}.
        source: "paper" (original assets) or "scratch" (locally generated).
        random_state: RNG seed for scratch generation.
        max_samples: Optional cap per split for faster smoke tests.
    """

    if dataset_name not in {"BAS", "Custom BAS", "MNIST"}:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    if source not in {"paper", "scratch"}:
        raise ValueError(f"Invalid source: {source}")

    if dataset_name == "BAS":
        if source == "paper":
            x_train, x_test, y_train, y_test = paper.get_bas()
        else:
            x_train, x_test, y_train, y_test = scratch.get_bas(
                random_state=random_state
            )
        return _limit_samples(
            x_train, x_test, y_train, y_test, max_samples, random_state
        )

    if dataset_name == "Custom BAS":
        if source == "paper":
            _ensure_shared_file("FULL_DATASET_600_samples.bin")
            x_train, x_test, y_train, y_test = paper.get_custom_bas()
        else:
            x_train, x_test, y_train, y_test = scratch.get_custom_bas(
                random_state=random_state
            )
        return _limit_samples(
            x_train, x_test, y_train, y_test, max_samples, random_state
        )

    # MNIST
    if source == "paper":
        x_train, x_test, y_train, y_test = paper.get_mnist()
    else:
        x_train, x_test, y_train, y_test = scratch.get_mnist(random_state=random_state)
    return _limit_samples(x_train, x_test, y_train, y_test, max_samples, random_state)


def get_dataset_description(x_train, x_test, y_train, y_test, dataset_name: str) -> str:
    arrays = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    description = ""

    for name, array in arrays.items():
        description += f"\n\U0001f4ca{dataset_name} - {name}\n"
        description += "-" * (len(name) + 4) + "\n"
        description += f"Shape      : {array.shape}\n"
        description += f"Dtype      : {array.dtype}\n\n"

        flat = array.flatten()
        if np.issubdtype(array.dtype, np.number):
            description += f"Min        : {flat.min()}\n"
            description += f"Max        : {flat.max()}\n"
            description += f"Mean       : {flat.mean():.4f}\n"
            description += f"Std        : {flat.std():.4f}\n\n"
        else:
            description += "Non-numeric data\n\n"

        unique_vals = np.unique(array)
        if unique_vals.size <= 20:
            description += f"Unique vals: {unique_vals}\n"
        else:
            description += (
                f"Unique vals: {unique_vals[:10]} ... (total {len(unique_vals)})\n"
            )

    return description


def save_dataset_description(
    x_train, x_test, y_train, y_test, dataset_name: str, save_path
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w") as f:
        f.write(get_dataset_description(x_train, x_test, y_train, y_test, dataset_name))


def convert_dataset_to_tensor(x_train, x_test, y_train, y_test):
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    return x_train_t, x_test_t, y_train_t, y_test_t


def convert_tensor_to_loader(x_train, y_train, batch_size: int = 6):
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader


def convert_scalar_labels_to_onehot(y_train, y_test):
    num_classes = 2
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]
    return y_train_onehot, y_test_onehot


def main():
    details_dir = _ensure_shared_dir() / "datasets_details"

    scratch_bas = get_dataset("BAS", "scratch", 42)
    save_dataset_description(*scratch_bas, "BAS", details_dir / "BAS_scratch.txt")

    scratch_custom = get_dataset("Custom BAS", "scratch", 42)
    save_dataset_description(
        *scratch_custom, "Custom BAS", details_dir / "custom_BAS_scratch.txt"
    )

    scratch_mnist = get_dataset("MNIST", "scratch", 42)
    save_dataset_description(*scratch_mnist, "MNIST", details_dir / "MNIST_scratch.txt")

    paper_bas = get_dataset("BAS", "paper", 42)
    save_dataset_description(*paper_bas, "BAS", details_dir / "BAS_paper.txt")

    paper_custom = get_dataset("Custom BAS", "paper", 42)
    save_dataset_description(
        *paper_custom, "Custom BAS", details_dir / "custom_BAS_paper.txt"
    )

    paper_mnist = get_dataset("MNIST", "paper", 42)
    save_dataset_description(*paper_mnist, "MNIST", details_dir / "MNIST_paper.txt")


if __name__ == "__main__":
    main()

__all__ = [
    "get_dataset",
    "get_dataset_description",
    "save_dataset_description",
    "convert_dataset_to_tensor",
    "convert_tensor_to_loader",
    "convert_scalar_labels_to_onehot",
]
