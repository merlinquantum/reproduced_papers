from __future__ import annotations

from common import REPO_ROOT
from lib.data import load_cv_splits


def _smoke_cfg() -> dict:
    return {
        "seed": 42,
        "data_root": str(REPO_ROOT / "data"),
        "cv": {"n_repeats": 1, "max_splits": 1},
    }


def test_one_split_shapes_and_balance():
    splits = load_cv_splits(_smoke_cfg())
    assert len(splits) == 1
    split = splits[0]

    n_features = split["X_train"].shape[1]
    assert n_features == 29  # V1..V28 + Amount, no Time/Class
    assert split["X_val"].shape[1] == n_features
    assert split["X_analysis"].shape[1] == n_features
    assert split["X_holdout"].shape[1] == n_features

    # Training pool is downsampled to an exact 50/50 class balance.
    y_train = split["y_train"]
    n_fraud = int((y_train == 1).sum())
    n_nonfraud = int((y_train == 0).sum())
    assert n_fraud == n_nonfraud
    assert n_fraud > 0

    # Validation/analysis/holdout are NOT downsampled - natural (heavily
    # skewed) class balance is preserved.
    assert split["y_val"].mean() < 0.05
    assert split["y_analysis"].mean() < 0.05
    assert split["y_holdout"].mean() < 0.05

    # Holdout is roughly 25% of the fold's held-out test pool (50/25/25 split).
    test_pool_size = (
        len(split["y_val"]) + len(split["y_analysis"]) + len(split["y_holdout"])
    )
    holdout_fraction = len(split["y_holdout"]) / test_pool_size
    assert abs(holdout_fraction - 0.25) < 0.02


def test_repeat_and_fold_indices_present():
    cfg = _smoke_cfg()
    cfg["cv"] = {"n_repeats": 1, "max_splits": 3}
    splits = load_cv_splits(cfg)
    assert len(splits) == 3
    assert [s["fold"] for s in splits] == [0, 1, 2]
    assert all(s["repeat"] == 0 for s in splits)
