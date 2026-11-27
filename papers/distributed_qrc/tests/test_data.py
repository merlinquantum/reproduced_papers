"""IPTO loader: paper window fingerprint, Table II anchors, splits and normalization."""

from __future__ import annotations

import common  # noqa: F401
import numpy as np
import pytest
from common import load_project_defaults
from distributed_qrc.lib.data import (
    PAPER_WINDOW_LENGTH,
    RAW_SUBDIR,
    build_dataloader,
    build_forecast_data,
    dataset_dir,
)

CFG = load_project_defaults()
pytestmark = pytest.mark.skipif(
    not (dataset_dir(CFG) / RAW_SUBDIR).is_dir()
    and not (dataset_dir(CFG) / "processed").is_dir(),
    reason="IPTO data not acquired; see the README Data section",
)


def test_paper_window_reproduces_table_ii():
    data = build_forecast_data(CFG)
    assert data.metadata["length"] == PAPER_WINDOW_LENGTH
    first = data.denormalize("train", data.inputs["train"][:2])
    assert np.round(first).astype(int).tolist() == [
        [4182, 3899, 3932, 3945],
        [3899, 3932, 3945, 3795],
    ]
    assert round(float(data.denormalize("train", data.targets["train"][:2])[1])) == 3911
    last = data.denormalize(
        "test", np.r_[data.inputs["test"][-1], data.targets["test"][-1]]
    )
    assert np.round(last).astype(int).tolist() == [5702, 5703, 5320, 4814, 4414]


def test_split_sizes_and_batch_shape():
    batch = next(iter(build_dataloader(CFG, split="train")))
    assert batch[0].shape == (24539 - 4, 4) and batch[1].shape == (24539 - 4,)
    data = build_forecast_data(CFG)
    assert [data.inputs[s].shape[0] for s in ("train", "eval", "test")] == [
        24535,
        5254,
        5263,
    ]


def test_per_split_normalization_spans_unit_interval():
    cfg = {**CFG, "dataset": {**CFG["dataset"], "normalization": "per_split"}}
    data = build_forecast_data(cfg)
    for split in ("train", "eval", "test"):
        values = np.r_[data.inputs[split][0], data.targets[split]]
        assert values.min() == pytest.approx(0.0) and values.max() == pytest.approx(1.0)


def test_interpolation_removes_non_positive_loads():
    cfg = {**CFG, "dataset": {**CFG["dataset"], "artifact_handling": "interpolate"}}
    data = build_forecast_data(cfg)
    assert len(data.metadata["preprocessing"]["replaced_indices"]) == 8
    assert data.scalers["train"].minimum > 0
