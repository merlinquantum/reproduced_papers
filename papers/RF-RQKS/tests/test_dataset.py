"""Tests for RF-RQKS dataset synthesis and packaging."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from lib.dataset import (
    DatasetConfig,
    build_dataset,
    generate_anomaly,
    scale_anomaly_to_jsr,
)


@pytest.fixture
def small_config() -> DatasetConfig:
    """Return a fast configuration that preserves production relationships.

    Returns
    -------
    DatasetConfig
        Small valid test configuration.
    """
    return DatasetConfig(
        sampling_rate_hz=64_000.0,
        iq_points_per_sample=4096,
        bandwidth_hz=48_000.0,
        stft_window_length=128,
        stft_nfft=256,
        stft_overlap_ratio=0.25,
        spectrogram_height=24,
        spectrogram_width=20,
        train_source_samples=2,
        test_source_samples=1,
        jsr_db_values=(-10.0, -2.0, 5.0),
        minimum_anomaly_fraction=0.2,
        maximum_anomaly_fraction=0.5,
        hopping_bandwidth_hz=(2_000.0, 6_000.0),
        hopping_dwell_fraction=(0.02, 0.08),
        iq_dtype="float32",
        output_dtype="float32",
        seed=17,
    )


def test_scale_anomaly_matches_requested_jsr() -> None:
    lte_iq = np.full(128, 2.0 + 0.0j)
    anomaly_iq = np.zeros(128, dtype=np.complex128)
    anomaly_iq[32:96] = 1.0 + 0.0j

    scaled = scale_anomaly_to_jsr(lte_iq, anomaly_iq, jsr_db=-6.0)
    active = np.abs(scaled) > 0
    measured_jsr_db = 10.0 * np.log10(
        np.mean(np.abs(scaled[active]) ** 2) / np.mean(np.abs(lte_iq[active]) ** 2)
    )

    assert measured_jsr_db == pytest.approx(-6.0)


@pytest.mark.parametrize("anomaly_type", ["chirp", "barrage", "frequency_hopping"])
def test_each_anomaly_is_finite_and_scaled(
    anomaly_type: str, small_config: DatasetConfig
) -> None:
    rng = np.random.default_rng(5)
    lte_iq = rng.standard_normal(
        small_config.iq_points_per_sample
    ) + 1j * rng.standard_normal(small_config.iq_points_per_sample)
    anomaly_iq, parameters = generate_anomaly(
        lte_iq, anomaly_type, -2.0, rng, small_config
    )
    active = np.abs(anomaly_iq) > 0
    measured_jsr_db = 10.0 * np.log10(
        np.mean(np.abs(anomaly_iq[active]) ** 2) / np.mean(np.abs(lte_iq[active]) ** 2)
    )

    assert np.isfinite(anomaly_iq).all()
    assert measured_jsr_db == pytest.approx(-2.0)
    assert parameters["stop_sample"] > parameters["start_sample"]


def test_build_dataset_writes_paired_disjoint_splits(
    tmp_path: Path, small_config: DatasetConfig
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    rng = np.random.default_rng(3)
    for file_index in range(3):
        iq = rng.standard_normal(
            small_config.iq_points_per_sample
        ) + 1j * rng.standard_normal(small_config.iq_points_per_sample)
        interleaved = np.empty(2 * iq.size, dtype=np.float32)
        interleaved[::2] = iq.real
        interleaved[1::2] = iq.imag
        interleaved.tofile(input_root / f"capture_{file_index}.bin")

    output_root = tmp_path / "dataset"
    build_dataset(input_root, output_root, small_config)

    train_spectrograms = np.load(
        output_root / "train" / "spectrograms.npy", mmap_mode="r"
    )
    test_spectrograms = np.load(
        output_root / "test" / "spectrograms.npy", mmap_mode="r"
    )
    train_labels = np.load(output_root / "train" / "labels.npy")
    test_labels = np.load(output_root / "test" / "labels.npy")
    with (output_root / "train" / "metadata.csv").open(
        newline="", encoding="utf-8"
    ) as metadata_file:
        train_metadata = list(csv.DictReader(metadata_file))
    with (output_root / "test" / "metadata.csv").open(
        newline="", encoding="utf-8"
    ) as metadata_file:
        test_metadata = list(csv.DictReader(metadata_file))

    assert train_spectrograms.shape == (4, 24, 20)
    assert test_spectrograms.shape == (2, 24, 20)
    assert train_labels.tolist() == [0, 1, 0, 1]
    assert test_labels.tolist() == [0, 1]
    assert {row["source_path"] for row in train_metadata}.isdisjoint(
        {row["source_path"] for row in test_metadata}
    )
    assert (output_root / "manifest.json").is_file()
    assert (output_root / "plot_data_examples.py").is_file()


def test_build_dataset_refuses_to_overwrite(
    tmp_path: Path, small_config: DatasetConfig
) -> None:
    output_root = tmp_path / "existing"
    output_root.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_dataset(
            tmp_path,
            output_root,
            replace(small_config, train_source_samples=1, test_source_samples=1),
        )
