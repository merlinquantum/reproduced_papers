from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

QORC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = QORC_DIR.parents[1]
for import_path in (REPO_ROOT, QORC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

pytest.importorskip("merlin")

from papers.shared.QORC import datasets


def _ten_class_dataset(samples_per_class: int = 1000):
    labels = np.repeat(np.arange(10), samples_per_class)
    data = np.arange(labels.size, dtype=np.float32).reshape(-1, 1)
    return data, labels


def test_balanced_sampling_returns_requested_count_per_class():
    data, labels = _ten_class_dataset()

    sampled_data, sampled_labels = datasets._sample_training_data(
        data, labels, "balanced", sample_count=100, samples_per_class=None, seed=7
    )

    assert sampled_data.shape == (100, 1)
    assert np.array_equal(np.bincount(sampled_labels, minlength=10), np.full(10, 10))


def test_gaussian_sampling_matches_table_percentages():
    data, labels = _ten_class_dataset()

    _, sampled_labels = datasets._sample_training_data(
        data, labels, "gauss", sample_count=1000, samples_per_class=None, seed=7
    )

    expected_counts = datasets._allocate_class_counts(
        datasets.MNIST_GAUSSIAN_PERCENTAGES, 1000
    )
    assert np.array_equal(np.bincount(sampled_labels, minlength=10), expected_counts)


def test_imbalanced_sampling_shuffles_table_percentages_by_seed():
    data, labels = _ten_class_dataset(samples_per_class=1000)

    _, labels_seed_1 = datasets._sample_training_data(
        data, labels, "imbal", sample_count=1000, samples_per_class=None, seed=1
    )
    _, labels_seed_2 = datasets._sample_training_data(
        data, labels, "imbal", sample_count=1000, samples_per_class=None, seed=2
    )

    expected_counts = datasets._allocate_class_counts(
        datasets.MNIST_IMBALANCED_PERCENTAGES, 1000
    )
    counts_seed_1 = np.bincount(labels_seed_1, minlength=10)
    counts_seed_2 = np.bincount(labels_seed_2, minlength=10)
    assert np.array_equal(np.sort(counts_seed_1), np.sort(expected_counts))
    assert np.array_equal(np.sort(counts_seed_2), np.sort(expected_counts))
    assert not np.array_equal(counts_seed_1, counts_seed_2)


def test_medmnist_rejects_mnist_only_sampling_modes():
    with pytest.raises(ValueError, match="full sampling only"):
        datasets.get_qorc_dataset("oct", sampling="balanced")


def test_get_qorc_dataset_applies_sampling_to_mnist_training_only(monkeypatch):
    train_data, train_labels = _ten_class_dataset()
    test_data = np.ones((5, 1), dtype=np.uint8)
    test_labels = np.arange(5)
    monkeypatch.setattr(
        datasets,
        "get_mnist_variant",
        lambda _: (train_data, train_labels, test_data, test_labels),
    )

    sampled_train, sampled_labels, returned_test, returned_test_labels = (
        datasets.get_qorc_dataset(
            "mnist",
            sampling="balanced",
            samples_per_class=3,
            seed=5,
        )
    )

    assert sampled_train.shape == (30, 1)
    assert np.array_equal(np.bincount(sampled_labels, minlength=10), np.full(10, 3))
    assert np.array_equal(returned_test, test_data)
    assert np.array_equal(returned_test_labels, test_labels)
