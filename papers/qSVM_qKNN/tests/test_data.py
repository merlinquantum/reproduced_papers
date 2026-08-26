"""Focused tests for qSVM data preparation."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

from lib.data import load_and_prepare  # noqa: E402


def _write_tar_member(archive_path: Path, member_name: str, content: str) -> None:
    data = content.encode("utf-8")
    info = tarfile.TarInfo(member_name)
    info.size = len(data)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(data))


def test_url_loader_splits_before_scaling_and_exposes_quantum_view(tmp_path):
    data_dir = tmp_path / "qSVM_qKNN" / "ISCX-URL-2016"
    data_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "feature_a": [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
            "feature_b": [5, 5, 6, 6, 15, 15, 16, 16, 25, 25, 26, 26],
            "feature_c": list("aaaabbbbcccc"),
            "class": ["benign", "Defacement"] * 6,
        }
    )
    frame.to_csv(data_dir / "toy.csv", index=False)
    cfg = {
        "dataset": "url2016",
        "data_root": str(tmp_path),
        "data_dir": "qSVM_qKNN",
        "dataset_file": "ISCX-URL-2016/toy.csv",
        "label_column": "class",
        "test_size": 0.25,
        "subset_size": 0,
        "feature_limit": 2,
        "balance_classes": True,
        "drop_missing_rows": True,
        "quantum_input_range": [0.0, 1.0],
        "photonic_n_modes": 2,
    }

    prepared = load_and_prepare(cfg, seed=7)

    assert prepared.X_train_classical.shape[1] == 2
    assert prepared.X_train_quantum.shape == prepared.X_train_classical.shape
    assert prepared.X_train_photonic.shape == prepared.X_train_classical.shape
    assert np.allclose(prepared.X_train_classical.mean(axis=0), 0.0)
    assert prepared.X_train_quantum.min() >= 0.0
    assert prepared.X_train_quantum.max() <= 1.0
    assert prepared.X_train_photonic.min() >= 0.0
    assert prepared.X_train_photonic.max() <= 1.0
    assert set(prepared.y_train) == {0, 1}
    assert prepared.class_balance == {0: 0.5, 1: 0.5}


def test_synthetic_loader_requires_no_csv_files():
    cfg = {
        "data_source": "synthetic",
        "dataset": "synthetic",
        "label_column": "class",
        "test_size": 0.25,
        "subset_size": 48,
        "synthetic_n_features": 6,
        "feature_limit": 4,
        "balance_classes": True,
        "drop_missing_rows": True,
        "quantum_input_range": [0.0, 1.0],
        "photonic_n_modes": 4,
    }

    prepared = load_and_prepare(cfg, seed=11)

    assert prepared.source == "synthetic:qSVM_qKNN"
    assert prepared.X_train_quantum.shape[1] == 4
    assert prepared.X_test_quantum.min() >= 0.0
    assert prepared.X_test_quantum.max() <= 1.0
    assert set(prepared.y_test) == {0, 1}


def test_loader_caps_test_size_before_balancing():
    cfg = {
        "data_source": "synthetic",
        "dataset": "synthetic",
        "label_column": "class",
        "test_size": 0.5,
        "max_test_size": 10,
        "subset_size": 80,
        "synthetic_n_features": 6,
        "feature_limit": 4,
        "balance_classes": True,
        "drop_missing_rows": True,
        "quantum_input_range": [0.0, 1.0],
        "photonic_n_modes": 4,
    }

    prepared = load_and_prepare(cfg, seed=11)

    assert len(prepared.y_test) == 10
    assert len(prepared.y_train) == 70
    assert prepared.class_balance == {0: 0.5, 1: 0.5}


def test_ids_loader_extracts_selected_csv_once_and_reuses_cache(tmp_path):
    data_dir = tmp_path / "qSVM_qKNN" / "ISCX-IDS-2012"
    data_dir.mkdir(parents=True)
    archive_path = data_dir / "iscxids2012-master.tar.gz"
    member_name = "iscxids2012-master/data/CSV/ToyFlows.csv"
    frame = pd.DataFrame(
        {
            "duration": np.arange(16),
            "packets": np.arange(16) * 2,
            "Label": ["Normal", "Attack"] * 8,
        }
    )
    _write_tar_member(archive_path, member_name, frame.to_csv(index=False))
    cfg = {
        "dataset": "ids2012",
        "data_root": str(tmp_path),
        "data_dir": "qSVM_qKNN",
        "ids_archive": "ISCX-IDS-2012/iscxids2012-master.tar.gz",
        "ids_csv_files": ["ToyFlows.csv"],
        "label_column": "Label",
        "test_size": 0.25,
        "subset_size": 0,
        "feature_limit": 2,
        "balance_classes": True,
        "drop_missing_rows": True,
        "quantum_input_range": [0.0, 1.0],
        "photonic_n_modes": 2,
    }

    first = load_and_prepare(cfg, seed=3)
    cached_csv = data_dir / member_name
    assert cached_csv.exists()
    assert str(cached_csv) in first.source
    assert str(archive_path) not in first.source

    archive_path.unlink()
    second = load_and_prepare(cfg, seed=3)

    assert second.source == first.source
    assert second.n_rows_raw == first.n_rows_raw == 16
    assert second.X_train_quantum.shape == first.X_train_quantum.shape
