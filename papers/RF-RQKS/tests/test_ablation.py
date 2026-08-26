"""Tests for the RF-RQKS ablation data and protocol."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lib.ablation import run_ablation
from lib.ablation import run_readout_comparison
from lib.ablation_data import load_dct_dataset
from lib.qks import DummyRBFSampler, build_sampler


def _write_split(
    root: Path, split: str, pair_count: int, feature_count: int, seed: int
) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(pair_count * 2, feature_count)).astype(np.float32)
    labels = np.tile(np.asarray([0, 1], dtype=np.uint8), pair_count)
    np.save(split_dir / "features.npy", features)
    np.save(split_dir / "labels.npy", labels)
    with (split_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_index", "pair_index"])
        writer.writeheader()
        for sample_index in range(pair_count * 2):
            writer.writerow(
                {"sample_index": sample_index, "pair_index": sample_index // 2}
            )


@pytest.fixture
def representation_root(tmp_path: Path) -> Path:
    root = tmp_path / "dct64x64"
    _write_split(root, "train", pair_count=10, feature_count=8, seed=1)
    _write_split(root, "test", pair_count=4, feature_count=8, seed=2)
    return root


def test_loader_keeps_normal_anomaly_pairs_together(
    representation_root: Path,
) -> None:
    dataset = load_dct_dataset(
        representation_root,
        validation_fraction=0.2,
        seed=42,
        standardization_batch_size=4,
    )
    assert dataset.train_features.shape == (16, 8)
    assert dataset.validation_features.shape == (4, 8)
    assert dataset.test_features.shape == (8, 8)
    assert np.array_equal(np.bincount(dataset.train_labels), [8, 8])
    assert np.array_equal(np.bincount(dataset.validation_labels), [2, 2])
    np.testing.assert_allclose(dataset.train_features.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(dataset.train_features.std(axis=0), 1.0, atol=1e-6)


def test_default_config_uses_file_free_synthetic_smoke_dataset() -> None:
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "defaults.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["data_source"] == "synthetic"


def test_dummy_sampler_output_shape() -> None:
    sampler = DummyRBFSampler(
        photon_count=1,
        mode_count=4,
        depth=1,
        episode_count=3,
        input_feature_count=8,
    )
    assert tuple(sampler(torch.zeros(5, 8)).shape) == (5, 12)


def test_qiskit_sampler_output_shape_and_probability_normalization() -> None:
    sampler = build_sampler(
        "qiskit",
        photon_count=None,
        mode_count=None,
        depth=2,
        episode_count=2,
        input_feature_count=8,
        encoding_strategy="L2",
        entangling_strategy="V1",
        same_haar=True,
        qubit_count=3,
    )
    output = sampler(torch.zeros(4, 8))
    assert tuple(output.shape) == (4, 16)
    episode_probabilities = output.reshape(4, 2, 8)
    assert torch.allclose(
        episode_probabilities.sum(dim=2), torch.ones(4, 2), atol=1e-5
    )


def test_qiskit_sampler_rejects_hardware_execution() -> None:
    with pytest.raises(ValueError, match="simulator-only"):
        build_sampler(
            "qiskit", None, None, 1, 1, 4, "L2", None, False, run_on_hardware=True,
            qubit_count=2,
        )


def test_photonic_sampler_requires_thales_merlin_version(monkeypatch) -> None:
    monkeypatch.setattr("lib.qks.version", lambda _name: "0.1.2")
    with pytest.raises(RuntimeError, match="requires merlinquantum==0.4.1"):
        build_sampler("photonic", 1, 2, 1, 1, 8, "L2", None, False)


def test_five_stage_smoke_run(representation_root: Path, tmp_path: Path) -> None:
    dataset = load_dct_dataset(
        representation_root,
        validation_fraction=0.2,
        seed=42,
        standardization_batch_size=4,
    )
    config = {
        "sampler": "dummy_rbf",
        "device": "cpu",
        "seed": 42,
        "batch_size": 4,
        "readout_c": 1.0,
        "readout_max_iter": 1000,
        "scale_sampled_features": False,
        "kernel_components": 4,
        "kernel_gamma": 1.0,
        "maximum_output_features": 128,
        "initial_depth": 1,
        "encoding_strategy": "L2",
        "entangling_strategy": "V1",
        "same_haar": False,
        "stage_1": {"mode_counts": [2], "episode_counts": [2]},
        "stage_2": {"shortlist_count": 1, "depths": [1]},
        "stage_3": {"depth_episode_pairs": [[1, 2]]},
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = run_ablation(config, dataset, run_dir)
    persisted = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert persisted["status"] == "complete"
    assert len(state["stages"]["stage_1"]["results"]) == 2
    assert state["stages"]["stage_5"]["results"][0]["best_qks_readout"]
    assert (run_dir / "figures" / "stage_4.png").exists()


def test_figure_six_readout_comparison(representation_root: Path, tmp_path: Path) -> None:
    dataset = load_dct_dataset(
        representation_root,
        validation_fraction=0.2,
        seed=42,
        standardization_batch_size=4,
    )
    config = {
        "sampler": "dummy_rbf",
        "device": "cpu",
        "seed": 42,
        "batch_size": 4,
        "readout_c": 1.0,
        "readout_max_iter": 1000,
        "scale_sampled_features": False,
        "kernel_components": 4,
        "kernel_gamma": 1.0,
        "maximum_output_features": 128,
        "model": {
            "photon_count": 1,
            "mode_count": 2,
            "depth": 1,
            "episode_count": 2,
            "encoding_strategy": "L2",
            "entangling_strategy": "V1",
            "same_haar": False,
        },
    }
    result = run_readout_comparison(config, dataset, tmp_path / "figure_6")
    assert result["best_qks_readout"]
    assert (tmp_path / "figure_6" / "figures" / "figure_6.png").exists()


def test_qiskit_ablation_smoke(representation_root: Path, tmp_path: Path) -> None:
    dataset = load_dct_dataset(
        representation_root,
        validation_fraction=0.2,
        seed=42,
        standardization_batch_size=4,
    )
    config = {
        "sampler": "qiskit",
        "device": "cpu",
        "seed": 42,
        "batch_size": 8,
        "readout_c": 1.0,
        "readout_max_iter": 1000,
        "scale_sampled_features": False,
        "kernel_components": 4,
        "kernel_gamma": 1.0,
        "maximum_output_features": 128,
        "initial_depth": 1,
        "encoding_strategy": "L2",
        "entangling_strategy": "V1",
        "same_haar": False,
        "stage_1": {"qubit_counts": [2], "episode_counts": [2]},
        "stage_2": {"shortlist_count": 1, "depths": [1]},
        "stage_3": {"depth_episode_pairs": [[1, 2]]},
        "stage_4": {"qubit_counts": [1, 2]},
    }
    state = run_ablation(config, dataset, tmp_path / "qiskit_ablation")
    assert state["status"] == "complete"
    assert state["best_model"]["configuration"]["qubit_count"] in {1, 2}
