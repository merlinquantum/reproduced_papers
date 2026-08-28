"""Tests for RF-RQKS DCT representations and split-aware normalization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from lib.representations import (
    DctRepresentationConfig,
    build_dct_representation_cache,
    compute_dct_features,
    fit_feature_standardization,
    standardize_features,
)
from scipy.fft import dctn


@pytest.fixture
def dct_config() -> DctRepresentationConfig:
    """Return a small DCT configuration.

    Returns
    -------
    DctRepresentationConfig
        Valid configuration retaining a 3-by-2 block.
    """
    return DctRepresentationConfig(
        frequency_coefficients=3,
        time_coefficients=2,
        dct_type=2,
        normalization="ortho",
        batch_size=2,
        output_dtype="float32",
        workers=1,
    )


def test_compute_dct_features_retains_upper_left_block(
    dct_config: DctRepresentationConfig,
) -> None:
    spectrograms = np.arange(2 * 5 * 4, dtype=np.float32).reshape(2, 5, 4)
    expected = dctn(spectrograms, type=2, norm="ortho", axes=(-2, -1))[
        :, :3, :2
    ].reshape(2, 6)

    actual = compute_dct_features(spectrograms, dct_config)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_standardization_uses_only_selected_training_rows() -> None:
    features = np.array(
        [[0.0, 2.0], [2.0, 4.0], [100.0, 400.0], [200.0, 800.0]],
        dtype=np.float32,
    )
    training_indices = np.array([0, 1])

    feature_mean, feature_standard_deviation = fit_feature_standardization(
        features, training_indices, batch_size=1
    )
    normalized_training = standardize_features(
        features[training_indices], feature_mean, feature_standard_deviation
    )

    np.testing.assert_allclose(feature_mean, [1.0, 3.0])
    np.testing.assert_allclose(feature_standard_deviation, [1.0, 1.0])
    np.testing.assert_allclose(normalized_training.mean(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(normalized_training.std(axis=0), [1.0, 1.0])


def test_build_dct_representation_cache(
    tmp_path: Path, dct_config: DctRepresentationConfig
) -> None:
    input_root = tmp_path / "processed"
    rng = np.random.default_rng(8)
    for split_name, sample_count in (("train", 3), ("test", 2)):
        split_root = input_root / split_name
        split_root.mkdir(parents=True)
        np.save(
            split_root / "spectrograms.npy",
            rng.standard_normal((sample_count, 5, 4)).astype(np.float32),
        )
        np.save(split_root / "labels.npy", np.arange(sample_count) % 2)
        with (split_root / "metadata.csv").open(
            "w", encoding="utf-8", newline=""
        ) as metadata_file:
            writer = csv.writer(metadata_file)
            writer.writerow(["sample_index"])
            writer.writerows([[index] for index in range(sample_count)])

    output_root = tmp_path / "dct64x64"
    build_dct_representation_cache(input_root, output_root, dct_config)

    train_features = np.load(output_root / "train" / "features.npy")
    test_features = np.load(output_root / "test" / "features.npy")
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert train_features.shape == (3, 6)
    assert test_features.shape == (2, 6)
    assert manifest["normalized"] is False
    assert manifest["representation"] == "dct3x2"
