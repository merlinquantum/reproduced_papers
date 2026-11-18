"""Sweep plans expand to the intended runs, including leftover-only plans."""

from __future__ import annotations

import sys

import pytest
from common import PROJECT_DIR

sys.path.insert(0, str(PROJECT_DIR / "utils"))
from run_sweep import expand_plan  # noqa: E402


def _plan(**entry_overrides):
    entry = {
        "name": "qultsf",
        "config": "configs/table1_qultsf_original.json",
        "seeds": [2021, 2022, 2023],
        "sequence_lengths": [336],
        "prediction_horizons": [96, 720],
    }
    entry.update(entry_overrides)
    return {"sweep_id": "test", "entries": [entry]}


def test_full_plan_is_seed_major():
    runs = expand_plan(_plan())
    assert [run["run_id"] for run in runs] == [
        "qultsf_L336_T96_seed2021",
        "qultsf_L336_T720_seed2021",
        "qultsf_L336_T96_seed2022",
        "qultsf_L336_T720_seed2022",
        "qultsf_L336_T96_seed2023",
        "qultsf_L336_T720_seed2023",
    ]
    assert [run["repetition"] for run in runs] == [0, 0, 1, 1, 2, 2]


def test_run_seeds_keeps_the_original_repetition_index():
    runs = expand_plan(_plan(run_seeds=[2023]))
    assert [run["run_id"] for run in runs] == [
        "qultsf_L336_T96_seed2023",
        "qultsf_L336_T720_seed2023",
    ]
    assert {run["repetition"] for run in runs} == {2}


def test_run_seeds_outside_seeds_fails():
    with pytest.raises(ValueError, match="run_seeds"):
        expand_plan(_plan(run_seeds=[2030]))
