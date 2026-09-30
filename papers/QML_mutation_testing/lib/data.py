"""Dataset preparation for the QML mutation-testing reproduction.

The paper evaluates tabular classification datasets (Iris, Wine, Breast Cancer)
with a QNN whose feature map consumes one feature per qubit.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torchvision.datasets as datasets
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

DatasetName = Literal["iris", "wine", "breast_cancer"]


@dataclass(frozen=True)
class DatasetSplits:
    """Tensors and metadata for one train/validation/test split.

    Attributes
    ----------
    x_train, y_train : torch.Tensor
        Training features (scaled) and integer labels.
    x_val, y_val : torch.Tensor
        Validation features and labels, carved out of the training pool and
        used only for hyperparameter selection.
    x_test, y_test : torch.Tensor
        Held-out test features and labels.
    x_suite, y_suite : torch.Tensor
        The fixed subset of ``n_suite_samples`` test samples that the paper
        feeds to the original model to derive the mutation test suite.
    suite_indices : np.ndarray
        Positions of the suite samples inside the test split.
    """

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_suite: torch.Tensor
    y_suite: torch.Tensor
    suite_indices: np.ndarray


def load_tabular_splits(
    *,
    name: DatasetName,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
    n_suite_samples: int = 20,
    feature_max: float = 1.0,
) -> DatasetSplits:
    """Load a tabular dataset and build the train/val/test/mutation-suite splits.

    Parameters
    ----------
    name : str
        Name of the dataset ("iris", "wine", or "breast_cancer").
    seed : int
        Seed controlling the stratified splits and the suite subsample.
    test_size : float
        Fraction of the full dataset held out for testing.
    val_size : float
        Fraction of the *training pool* used for hyperparameter selection.
    n_suite_samples : int
        Number of test samples passed to the original model to build the test suite.
    feature_max : float
        Upper bound of the min-max feature scaling.

    Returns
    -------
    DatasetSplits
        Scaled tensors plus the mutation-suite subset.
    """

    if name == "iris":
        features, labels = load_iris(return_X_y=True)
    elif name == "wine":
        features, labels = load_wine(return_X_y=True)
    elif name == "breast_cancer":
        features, labels = load_breast_cancer(return_X_y=True)
    else:
        raise ValueError(f"Unsupported tabular dataset: {name}")

    x_pool, x_test, y_pool, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_pool, y_pool, test_size=val_size, random_state=seed, stratify=y_pool
    )

    n_features = x_train.shape[1]
    n_classes = len(np.unique(labels))

    valid_modes = n_features
    while math.comb(valid_modes, 2) % n_classes != 0:
        valid_modes -= 1

    if valid_modes < n_features:
        logger.info(
            "PCA: %s reduced by %d to %d features for LexGrouping.",
            name,
            n_features,
            valid_modes,
        )
        pca = PCA(n_components=valid_modes, random_state=seed).fit(x_train)
        x_train = pca.transform(x_train)
        x_val = pca.transform(x_val)
        x_test = pca.transform(x_test)

    scaler = MinMaxScaler(feature_range=(0.0, feature_max)).fit(x_train)
    x_train, x_val, x_test = (
        scaler.transform(x_train),
        scaler.transform(x_val),
        scaler.transform(x_test),
    )

    if len(y_test) < n_suite_samples:
        raise ValueError(
            f"Test split has {len(y_test)} samples but the mutation suite "
            f"needs {n_suite_samples}."
        )

    suite_indices = _stratified_subsample(y_test, n_suite_samples, seed)

    logger.info(
        "DATASET_READY | name=%s | train=%d | val=%d | test=%d | suite=%d "
        "| features=%d | classes=%d | scaling=minmax[0,%.4g]",
        name,
        len(y_train),
        len(y_val),
        len(y_test),
        len(suite_indices),
        x_train.shape[1],
        len(np.unique(labels)),
        feature_max,
    )

    def to_x(array):
        return torch.tensor(array, dtype=torch.float32)

    def to_y(array):
        return torch.tensor(array, dtype=torch.long)

    return DatasetSplits(
        x_train=to_x(x_train),
        y_train=to_y(y_train),
        x_val=to_x(x_val),
        y_val=to_y(y_val),
        x_test=to_x(x_test),
        y_test=to_y(y_test),
        x_suite=to_x(x_test[suite_indices]),
        y_suite=to_y(y_test[suite_indices]),
        suite_indices=suite_indices,
    )


def _stratified_subsample(labels: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Return indices of a class-balanced subsample of ``labels``."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = np.full(len(classes), n_samples // len(classes))
    per_class[: n_samples % len(classes)] += 1
    per_class = np.minimum(per_class, counts)

    selected: list[int] = []
    for class_label, quota in zip(classes, per_class):
        candidates = np.flatnonzero(labels == class_label)
        selected.extend(rng.permutation(candidates)[:quota].tolist())

    if len(selected) < n_samples:
        remaining = np.setdiff1d(np.arange(len(labels)), np.array(selected))
        selected.extend(
            rng.permutation(remaining)[: n_samples - len(selected)].tolist()
        )

    return np.sort(np.array(selected, dtype=int))


def load_image_splits(
    *,
    name: str = "mnist",
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
    n_suite_samples: int = 20,
) -> DatasetSplits:
    """Load an image dataset, resize to 4x4, and build splits."""

    if name == "mnist":
        ds_train = datasets.MNIST(root="./data", train=True, download=True)
        ds_test = datasets.MNIST(root="./data", train=False, download=True)
    elif name == "fashion_mnist":
        ds_train = datasets.FashionMNIST(root="./data", train=True, download=True)
        ds_test = datasets.FashionMNIST(root="./data", train=False, download=True)
    elif name == "kmnist":
        ds_train = datasets.KMNIST(root="./data", train=True, download=True)
        ds_test = datasets.KMNIST(root="./data", train=False, download=True)
    else:
        raise ValueError(f"Unsupported image dataset: {name}")

    raw_features = torch.cat([ds_train.data, ds_test.data], dim=0)
    raw_targets = torch.cat([ds_train.targets, ds_test.targets], dim=0)

    features = raw_features.float().unsqueeze(1) / 255.0  # Shape: (N, 1, 28, 28)
    features = torch.nn.functional.interpolate(
        features, size=(4, 4), mode="area"
    )  # Shape: (N, 1, 4, 4)
    features = features + 1e-6
    labels = raw_targets

    x_pool, x_test, y_pool, y_test = train_test_split(
        features.numpy(),
        labels.numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=labels.numpy(),
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_pool, y_pool, test_size=val_size, random_state=seed, stratify=y_pool
    )

    suite_indices = _stratified_subsample(y_test, n_suite_samples, seed)

    logger.info(
        "IMAGE_DATASET_READY | name=%s | train=%d | val=%d | test=%d | suite=%d | shape=(4, 4)",
        name,
        len(y_train),
        len(y_val),
        len(y_test),
        len(suite_indices),
    )

    def to_x(array):
        return torch.tensor(array, dtype=torch.float32)

    def to_y(array):
        return torch.tensor(array, dtype=torch.long)

    return DatasetSplits(
        x_train=to_x(x_train),
        y_train=to_y(y_train),
        x_val=to_x(x_val),
        y_val=to_y(y_val),
        x_test=to_x(x_test),
        y_test=to_y(y_test),
        x_suite=to_x(x_test[suite_indices]),
        y_suite=to_y(y_test[suite_indices]),
        suite_indices=suite_indices,
    )
