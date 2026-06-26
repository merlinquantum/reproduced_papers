"""Sanity checks for the QCNN-ID preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

from lib.data import load_and_prepare  # noqa: E402


@pytest.fixture
def synthetic_icu_dir(tmp_path):
    """Three tiny CSVs that mimic the ICU dataset layout."""
    rng = np.random.default_rng(0)
    n_per = 200
    cols = [f"f{i}" for i in range(12)]
    files = {
        "Attack.csv": 1,
        "environmentMonitoring.csv": 0,
        "patientMonitoring.csv": 0,
    }
    for name, label in files.items():
        df = pd.DataFrame(rng.normal(size=(n_per, 12)), columns=cols)
        df["class"] = name.replace(".csv", "")
        df["label"] = label
        df.to_csv(tmp_path / name, index=False)
    return tmp_path


def test_pipeline_shapes_and_scale(synthetic_icu_dir):
    cfg = {
        "data_root": str(synthetic_icu_dir.parent),
        "data_dir": synthetic_icu_dir.name,
        "subset_size": 0,
        "test_size": 0.3,
        "n_components": 4,
    }
    prepared = load_and_prepare(cfg, seed=1)
    assert prepared.X_train_quantum.shape[1] == 4
    assert prepared.X_test_quantum.shape[1] == 4
    assert prepared.X_train_classical.shape[1] == 4
    # 3 files * 200 rows each = 600 total samples.
    assert prepared.y_train.shape[0] + prepared.y_test.shape[0] == 600
    # Angle-encoded features must live in [0, pi] (allow a tiny FP slack).
    assert prepared.X_train_quantum.min() >= 0.0 - 1e-6
    assert prepared.X_train_quantum.max() <= np.pi + 1e-6
    assert prepared.X_test_quantum.min() >= 0.0 - 1e-6
    assert prepared.X_test_quantum.max() <= np.pi + 1e-6
    assert set(prepared.class_balance.keys()) == {0, 1}
    # Attack (label=1) is 1/3 of the synthetic dataset.
    assert 0.25 < prepared.class_balance[1] < 0.4


def test_single_csv_compat(tmp_path):
    """The legacy single-CSV path still works."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(100, 6)), columns=[f"f{i}" for i in range(6)])
    df["label"] = rng.integers(0, 2, size=100)
    csv = tmp_path / "merged.csv"
    df.to_csv(csv, index=False)
    cfg = {
        "data_csv": str(csv),
        "subset_size": 0,
        "test_size": 0.3,
        "n_components": 3,
    }
    prepared = load_and_prepare(cfg, seed=2)
    assert prepared.X_train_quantum.shape[1] == 3
    assert prepared.X_train_classical.shape[1] == 3
