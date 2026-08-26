"""Load leakage-safe RF-RQKS DCT datasets for the ablation protocol."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .representations import fit_feature_standardization, standardize_features


@dataclass(frozen=True)
class DatasetSplits:
    """Arrays used during model selection and held-out evaluation.

    Parameters
    ----------
    train_features : numpy.ndarray
        Standardized model-selection training features.
    train_labels : numpy.ndarray
        Training labels.
    validation_features : numpy.ndarray
        Standardized model-selection validation features.
    validation_labels : numpy.ndarray
        Validation labels.
    test_features : numpy.ndarray
        Held-out test features standardized with full development statistics.
    test_labels : numpy.ndarray
        Held-out test labels.
    development_features : numpy.ndarray
        Full original training split standardized using all its rows.
    development_labels : numpy.ndarray
        Full original training labels.
    """

    train_features: np.ndarray
    train_labels: np.ndarray
    validation_features: np.ndarray
    validation_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    development_features: np.ndarray
    development_labels: np.ndarray

    @property
    def input_feature_count(self) -> int:
        """Return the flattened DCT feature count.

        Returns
        -------
        int
            Number of input features per sample.
        """
        return int(self.train_features.shape[1])


def _read_pair_indices(metadata_path: Path) -> np.ndarray:
    """Read leakage-safe grouping identifiers from representation metadata.

    Parameters
    ----------
    metadata_path : pathlib.Path
        Metadata CSV containing a ``pair_index`` column and optionally a
        ``source_pair_index`` column for augmented representations.

    Returns
    -------
    numpy.ndarray
        Pair identifier aligned with every feature row.

    Raises
    ------
    ValueError
        If the metadata lacks ``pair_index``.
    """
    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if reader.fieldnames is None or "pair_index" not in reader.fieldnames:
            raise ValueError(f"Metadata must contain pair_index: {metadata_path}")
        grouping_field = (
            "source_pair_index"
            if "source_pair_index" in reader.fieldnames
            else "pair_index"
        )
        return np.asarray([int(row[grouping_field]) for row in reader], dtype=np.int64)


def _grouped_partition(
    pair_indices: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows while keeping each normal/anomalous source pair together.

    Parameters
    ----------
    pair_indices : numpy.ndarray
        Pair identifier for each row.
    validation_fraction : float
        Fraction of unique source pairs assigned to validation.
    seed : int
        Random seed used to shuffle unique pairs.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Training row indices followed by validation row indices.

    Raises
    ------
    ValueError
        If the requested partition would be empty.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    unique_pairs = np.unique(pair_indices)
    validation_pair_count = round(unique_pairs.size * validation_fraction)
    if validation_pair_count <= 0 or validation_pair_count >= unique_pairs.size:
        raise ValueError("validation_fraction produces an empty partition")
    shuffled_pairs = np.random.default_rng(seed).permutation(unique_pairs)
    validation_pairs = shuffled_pairs[:validation_pair_count]
    validation_mask = np.isin(pair_indices, validation_pairs)
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def load_dct_dataset(
    representation_root: Path,
    validation_fraction: float,
    seed: int,
    standardization_batch_size: int,
) -> DatasetSplits:
    """Load and standardize a cached DCT representation without leakage.

    Parameters
    ----------
    representation_root : pathlib.Path
        Directory containing ``train`` and ``test`` representation splits.
    validation_fraction : float
        Fraction of training source pairs reserved for model selection.
    seed : int
        Seed controlling the grouped partition.
    standardization_batch_size : int
        Batch size used to fit feature statistics.

    Returns
    -------
    DatasetSplits
        Model-selection and final-evaluation arrays.

    Raises
    ------
    FileNotFoundError
        If a required representation file is missing.
    ValueError
        If arrays and metadata have inconsistent lengths.
    """
    required_paths = [
        representation_root / split / filename
        for split in ("train", "test")
        for filename in ("features.npy", "labels.npy", "metadata.csv")
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing DCT representation files: {missing_paths}")

    raw_train_features = np.load(
        representation_root / "train" / "features.npy", mmap_mode="r"
    )
    train_labels = np.asarray(
        np.load(representation_root / "train" / "labels.npy"), dtype=np.int64
    )
    raw_test_features = np.load(
        representation_root / "test" / "features.npy", mmap_mode="r"
    )
    test_labels = np.asarray(
        np.load(representation_root / "test" / "labels.npy"), dtype=np.int64
    )
    pair_indices = _read_pair_indices(representation_root / "train" / "metadata.csv")
    if (
        raw_train_features.shape[0] != train_labels.size
        or pair_indices.size != train_labels.size
    ):
        raise ValueError(
            "Training features, labels, and metadata have inconsistent lengths"
        )
    if raw_test_features.shape[0] != test_labels.size:
        raise ValueError("Test features and labels have inconsistent lengths")

    training_indices, validation_indices = _grouped_partition(
        pair_indices, validation_fraction, seed
    )
    selection_mean, selection_std = fit_feature_standardization(
        raw_train_features, training_indices, standardization_batch_size
    )
    full_training_indices = np.arange(raw_train_features.shape[0], dtype=np.int64)
    final_mean, final_std = fit_feature_standardization(
        raw_train_features, full_training_indices, standardization_batch_size
    )

    return DatasetSplits(
        train_features=standardize_features(
            np.asarray(raw_train_features[training_indices]),
            selection_mean,
            selection_std,
        ).astype(np.float32),
        train_labels=train_labels[training_indices],
        validation_features=standardize_features(
            np.asarray(raw_train_features[validation_indices]),
            selection_mean,
            selection_std,
        ).astype(np.float32),
        validation_labels=train_labels[validation_indices],
        test_features=standardize_features(
            np.asarray(raw_test_features), final_mean, final_std
        ).astype(np.float32),
        test_labels=test_labels,
        development_features=standardize_features(
            np.asarray(raw_train_features), final_mean, final_std
        ).astype(np.float32),
        development_labels=train_labels,
    )
