"""Data checks on the local Yahoo snapshot (requires utils/download_yahoo_data.py to have run)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from common import REPO_ROOT, load_runtime_ready_config
from lib import data

DATA_ROOT = str(REPO_ROOT / "data")


@pytest.fixture(scope="module")
def directional_cfg():
    return load_runtime_ready_config()["directional"]["dataset"]


def test_one_batch_shapes(directional_cfg):
    cfg = load_runtime_ready_config()
    cfg["data_root"] = DATA_ROOT
    inputs, labels = next(iter(data.build_dataloader(cfg, split="train")))
    assert tuple(inputs.shape) == (32, 3) and tuple(labels.shape) == (32,)
    assert set(labels.unique().tolist()) <= {0.0, 1.0}


def test_lagged_3d_features_are_same_day_features_shifted(directional_cfg):
    lagged = data.build_directional_frame(directional_cfg, DATA_ROOT)
    same_day_cfg = {**directional_cfg, "feature_timing": "same_day"}
    same_day = data.build_directional_frame(same_day_cfg, DATA_ROOT)
    common = lagged.index.intersection(same_day.index)[5:50]
    previous = same_day.index.get_indexer(common) - 1
    assert np.allclose(
        lagged.loc[common, "rsi14"].to_numpy(), same_day["rsi14"].to_numpy()[previous]
    )
    assert (lagged.loc[common, "label"] == same_day.loc[common, "label"]).all()


def test_folds_are_chronological_and_disjoint(directional_cfg):
    frame = data.build_directional_frame(directional_cfg, DATA_ROOT)
    for fold in data.walk_forward_folds(frame.index, directional_cfg):
        assert (
            fold.train.max()
            < fold.early_stop.min()
            < fold.early_stop.max()
            < fold.validation.min()
        )
    last = data.walk_forward_folds(frame.index, directional_cfg)[-1]
    assert str(frame.index[last.validation[0]].date()).startswith("2020-01")


def test_seven_d_us_proxies_are_lagged():
    cfg = {
        "feature_set": "7d",
        "ticker": "^GSPC",
        "train_start": "2009-01-07",
        "test_end": "2021-12-31",
    }
    frame = data.build_directional_frame(cfg, DATA_ROOT)
    prices = data.load_aligned_prices(
        ["^GSPC", *data.SAME_DAY_INDICES, *data.LAGGED_US_INDICES], DATA_ROOT
    )
    returns = np.log(prices / prices.shift(1))
    day = frame.index[100]
    previous_day = prices.index[prices.index.get_loc(day) - 1]
    assert np.isclose(frame.loc[day, "^DJI_lag1"], returns.loc[previous_day, "^DJI"])
    assert np.isclose(frame.loc[day, "^N225"], returns.loc[day, "^N225"])


@pytest.mark.parametrize("split_mode", ["history_then_regime", "within_regime"])
def test_trading_split_slices_are_ordered(split_mode):
    import json

    from common import PROJECT_DIR

    trading_cfg = json.loads(
        (PROJECT_DIR / "configs" / "e5_original.json").read_text()
    )["trading"]["dataset"]
    trading_cfg = copy.deepcopy(trading_cfg) | {"split_mode": split_mode}
    frame = data.build_trading_frame(trading_cfg, DATA_ROOT)
    _, _, _, dates = data.trading_windows(frame, trading_cfg["window"])
    split = data.trading_split(dates, trading_cfg)
    assert (
        split.train.max()
        < split.early_stop.min()
        < split.model_select.min()
        < split.threshold_calibration.min()
    )
    if split_mode == "history_then_regime":
        assert split.threshold_calibration.max() < split.test.min()
        assert str(dates[split.test[0]].date()) >= "2008-01-02"


def test_table_v_timing_moves_only_apac_eu_indices_to_same_day():
    base = {
        "feature_set": "64d",
        "ticker": "AAPL",
        "train_start": "2009-01-07",
        "test_end": "2021-12-31",
    }
    lagged = data.build_directional_frame(
        {**base, "feature_timing": "lagged"}, DATA_ROOT
    )
    table_v = data.build_directional_frame(
        {**base, "feature_timing": "table_v"}, DATA_ROOT
    )
    assert lagged.shape[1] == table_v.shape[1] == 65
    assert "^N225_lag0" in table_v and "^N225_lag7" not in table_v
    assert "^DJI_lag0" not in table_v and "^GSPC_lag0" not in table_v
    day = table_v.index[200]
    previous_day = lagged.index[lagged.index.get_loc(day) - 1]
    assert np.isclose(table_v.loc[day, "^FTSE_lag1"], lagged.loc[day, "^FTSE_lag1"])
    assert np.isclose(
        table_v.loc[day, "^FTSE_lag0"],
        lagged.loc[lagged.index[lagged.index.get_loc(day) + 1], "^FTSE_lag1"],
    )
    assert previous_day < day
