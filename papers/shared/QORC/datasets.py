"""Shared dataset utilities for QORC (MNIST variants via MerLin)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from merlin.datasets.fashion_mnist import (
    get_data_test as get_fashion_mnist_test,
)
from merlin.datasets.fashion_mnist import (
    get_data_train as get_fashion_mnist_train,
)
from merlin.datasets.k_mnist import get_data_test as get_k_mnist_test
from merlin.datasets.k_mnist import get_data_train as get_k_mnist_train
from merlin.datasets.mnist_digits import (
    get_data_test_original as get_mnist_test,
)
from merlin.datasets.mnist_digits import (
    get_data_train_original as get_mnist_train,
)
from torch.utils.data import Dataset

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
        raise RuntimeError(
            "Shared data resolver unavailable; DATA_DIR or runtime_lib required"
        )
    return paper_data_dir("QORC")


_MERLIN_DATA_ROOT = _data_root()

MEDMNIST_DATASETS = {
    "oct": "OCTMNIST",
    "octmnist": "OCTMNIST",
    "organs": "OrganSMNIST",
    "organsm": "OrganSMNIST",
    "organsmnist": "OrganSMNIST",
    "organa": "OrganAMNIST",
    "organamnist": "OrganAMNIST",
    "derma": "DermaMNIST",
    "dermamnist": "DermaMNIST",
}

MNIST_GAUSSIAN_PERCENTAGES = np.array(
    [1.2, 3.5, 7.9, 13.8, 18.8, 20.0, 16.6, 10.7, 5.4, 2.1],
    dtype=np.float64,
)
MNIST_IMBALANCED_PERCENTAGES = np.array(
    [64.3, 9.7, 8.0, 6.4, 3.9, 3.2, 1.6, 1.3, 1.0, 0.6],
    dtype=np.float64,
)

if _datasets_utils:

    def _custom_data_dir() -> Path:
        return _MERLIN_DATA_ROOT

    _datasets_utils.get_venv_data_dir = _custom_data_dir  # type: ignore[attr-defined]


class tensor_dataset(Dataset):
    def __init__(self, np_x, np_y, device, dtype, transform=None, n_side_pixels=None):
        if isinstance(np_x, torch.Tensor):
            self.np_x = np_x.detach().clone().to(device=device, dtype=dtype)
        else:
            self.np_x = torch.tensor(np_x, device=device, dtype=dtype)

        if isinstance(np_y, torch.Tensor):
            self.np_y = np_y.detach().clone().to(device=device, dtype=torch.long)
        else:
            self.np_y = torch.tensor(np_y, device=device, dtype=torch.long)

        self.n_items = self.np_x.shape[0]

        assert self.n_items == self.np_y.shape[0], (
            f"tensor_dataset: x and y do not have the same number of rows. "
            f"self.np_x.shape: {self.np_x.shape}, self.np_y.shape: {self.np_y.shape}"
        )

        self.transform = transform
        self.n_side_pixels = n_side_pixels

    def __getitem__(self, index):
        image = self.np_x[index]
        label = self.np_y[index]
        if self.transform:
            if self.n_side_pixels:
                n_pixels = self.n_side_pixels * self.n_side_pixels
                image = self.transform(
                    image.view(self.n_side_pixels, self.n_side_pixels)
                ).view(n_pixels)
            else:
                image = self.transform(image)
        return image, label

    def __len__(self):
        return self.n_items


def seed_worker(worker_id, seed=42):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, seed=42):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=lambda id: seed_worker(id, seed),
        generator=torch.Generator().manual_seed(seed),
    )


def split_fold_numpy(label, data, n_fold, fold_index, split_seed=-1):
    if split_seed >= 0:
        np.random.seed(split_seed)
        shuffled_indices = np.random.permutation(len(label))
        label = label[shuffled_indices]
        data = data[shuffled_indices]
    fold_size = len(label) // n_fold
    val_start = fold_index * fold_size
    val_end = (fold_index + 1) * fold_size if fold_index < n_fold - 1 else len(label)
    val_indices = np.arange(val_start, val_end)
    train_indices = np.array([i for i in range(len(label)) if i not in val_indices])
    return (
        label[val_indices],
        data[val_indices],
        label[train_indices],
        data[train_indices],
    )


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


def _load_medmnist(dataset_name):
    dataset_class_name = MEDMNIST_DATASETS[dataset_name.lower()]
    try:
        import medmnist
    except ImportError as exc:
        raise ImportError("MedMNIST datasets require the 'medmnist' package.") from exc

    dataset_class = getattr(medmnist, dataset_class_name)
    dataset_root = _MERLIN_DATA_ROOT / "medmnist"
    train_dataset = dataset_class(split="train", root=str(dataset_root), download=True)
    test_dataset = dataset_class(split="test", root=str(dataset_root), download=True)
    return (
        np.asarray(train_dataset.imgs),
        np.asarray(train_dataset.labels).reshape(-1),
        np.asarray(test_dataset.imgs),
        np.asarray(test_dataset.labels).reshape(-1),
    )


def _allocate_class_counts(percentages, sample_count):
    exact_counts = np.asarray(percentages, dtype=np.float64) * sample_count / 100.0
    counts = np.floor(exact_counts).astype(int)
    remainder = sample_count - int(counts.sum())
    if remainder:
        fractional_parts = exact_counts - counts
        largest_remainders = np.argsort(-fractional_parts, kind="stable")[:remainder]
        counts[largest_remainders] += 1
    return counts


def _sample_training_data(
    data, labels, sampling, sample_count, samples_per_class, seed
):
    normalized_sampling = sampling.lower()
    if normalized_sampling == "full":
        if sample_count is None:
            return data, labels
        if sample_count <= 0 or sample_count > len(labels):
            raise ValueError(
                "sample_count must be between 1 and the training-set size."
            )
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(len(labels), size=sample_count, replace=False)
        return data[selected_indices], labels[selected_indices]

    if normalized_sampling == "balanced":
        if samples_per_class is None:
            if sample_count is None or sample_count % 10:
                raise ValueError(
                    "Balanced MNIST sampling requires samples_per_class or a "
                    "sample_count divisible by 10."
                )
            samples_per_class = sample_count // 10
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive.")
        class_counts = np.full(10, samples_per_class, dtype=int)
    elif normalized_sampling in {"gauss", "gaussian", "imbal", "imbalanced"}:
        if samples_per_class is not None:
            raise ValueError(
                "samples_per_class is only valid for balanced MNIST sampling."
            )
        sample_count = 10000 if sample_count is None else sample_count
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        percentages = (
            MNIST_GAUSSIAN_PERCENTAGES
            if normalized_sampling in {"gauss", "gaussian"}
            else MNIST_IMBALANCED_PERCENTAGES
        )
        if normalized_sampling in {"imbal", "imbalanced"}:
            percentages = percentages[np.random.default_rng(seed).permutation(10)]
        class_counts = _allocate_class_counts(percentages, sample_count)
    else:
        raise ValueError(
            "Unknown MNIST sampling mode. Expected full, balanced, gauss, or imbal."
        )

    rng = np.random.default_rng(seed)
    selected_indices = []
    for class_label, class_count in enumerate(class_counts):
        class_indices = np.flatnonzero(labels == class_label)
        if class_count > len(class_indices):
            raise ValueError(
                f"Not enough training examples for class {class_label}: "
                f"requested {class_count}, available {len(class_indices)}."
            )
        selected_indices.extend(
            rng.choice(class_indices, size=class_count, replace=False).tolist()
        )
    selected_indices = np.asarray(selected_indices, dtype=int)
    rng.shuffle(selected_indices)
    return data[selected_indices], labels[selected_indices]


def get_qorc_dataset(
    dataset_name,
    sampling="full",
    sample_count=None,
    samples_per_class=None,
    seed=42,
):
    """Load a QORC dataset and optionally resample its training set.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier: mnist, oct, organs, organa, or derma.
    sampling : str
        MNIST training sampling mode: full, balanced, gauss, or imbal. MedMNIST
        datasets currently support full sampling only. Default value is full.
    sample_count : int|None
        Total number of training samples for a sampled MNIST dataset. Gaussian
        and imbal modes default to 10000. Default value is None.
    samples_per_class : int|None
        Number of examples per class for balanced MNIST sampling. Default value
        is None.
    seed : int
        Seed controlling sample selection and the imbal class permutation.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Training data, training labels, test data, and test labels.

    Raises
    ------
    ValueError
        If the dataset or sampling configuration is unsupported.
    """
    normalized_name = dataset_name.lower()
    if normalized_name in MEDMNIST_DATASETS:
        if (
            sampling.lower() != "full"
            or sample_count is not None
            or samples_per_class is not None
        ):
            raise ValueError("MedMNIST datasets currently support full sampling only.")
        return _load_medmnist(normalized_name)

    if normalized_name != "mnist":
        raise ValueError(
            "Unknown dataset. Expected mnist, oct, organs, organa, or derma."
        )
    train_data, train_labels, test_data, test_labels = get_mnist_variant("mnist")
    sampled_train_data, sampled_train_labels = _sample_training_data(
        train_data,
        np.asarray(train_labels).reshape(-1),
        sampling,
        sample_count,
        samples_per_class,
        seed,
    )
    return sampled_train_data, sampled_train_labels, test_data, test_labels


__all__ = [
    "tensor_dataset",
    "seed_worker",
    "get_dataloader",
    "split_fold_numpy",
    "get_mnist_variant",
    "get_qorc_dataset",
]
