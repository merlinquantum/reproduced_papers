from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
from common import PROJECT_DIR

if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))

from QEGM_rare_events.lib.data import build_dataset  # noqa: E402

CSV = (
    PROJECT_DIR.parent.parent
    / "data"
    / "QEGM_rare_events"
    / ("sp500_daily_logreturns_1990_2022.csv")
)

pytestmark = pytest.mark.skipif(not CSV.exists(), reason="packaged CSV missing")


def make_cfg(tail_quantile: float = 0.05) -> dict:
    return {
        "dataset": {
            "name": "sp500",
            "csv_path": str(CSV),
            "tail_quantile": tail_quantile,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
        }
    }


def test_sp500_dataset_standardized_and_split():
    ds = build_dataset(make_cfg(), seed=42)
    full = torch.cat([ds.train, ds.val, ds.test]).numpy().flatten()
    assert full.size == ds.n_samples > 8000
    assert abs(full.mean()) < 1e-6
    assert abs(full.std() - 1.0) < 1e-3
    assert ds.test.shape[0] == int(round(0.15 * ds.n_samples))


def test_sp500_tail_threshold_matches_quantile():
    ds = build_dataset(make_cfg(tail_quantile=0.05), seed=0)
    full = torch.cat([ds.train, ds.val, ds.test]).numpy().flatten()
    tail_fraction = float(np.mean(np.abs(full) > ds.tail_threshold))
    assert tail_fraction == pytest.approx(0.05, abs=0.005)


def test_sp500_same_seed_same_split():
    a = build_dataset(make_cfg(), seed=7)
    b = build_dataset(make_cfg(), seed=7)
    assert torch.equal(a.test, b.test)


def test_unknown_dataset_name_rejected():
    cfg = make_cfg()
    cfg["dataset"]["name"] = "nope"
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataset(cfg, seed=0)
