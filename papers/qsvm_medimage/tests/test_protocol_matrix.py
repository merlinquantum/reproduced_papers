"""Tests for seed-level protocol W/T/L aggregation."""

import importlib.util
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "aggregate_protocol_matrix.py"
SPEC = importlib.util.spec_from_file_location("aggregate_protocol_matrix", SCRIPT_PATH)
aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate)


def test_all_wins():
    result = aggregate.summarize_deltas(
        np.array([0.8, 0.7, 0.9]),
        np.array([0.2, 0.3, 0.4]),
    )

    assert result["wins"] == 3
    assert result["ties"] == 0
    assert result["losses"] == 0


def test_all_losses():
    result = aggregate.summarize_deltas(
        np.array([0.2, 0.3, 0.4]),
        np.array([0.8, 0.7, 0.9]),
    )

    assert result["wins"] == 0
    assert result["ties"] == 0
    assert result["losses"] == 3


def test_exact_and_tolerance_based_ties():
    result = aggregate.summarize_deltas(
        np.array([0.5, 0.5000004, 0.6]),
        np.array([0.5, 0.5, 0.4]),
        tolerance=1e-6,
    )

    assert result["wins"] == 1
    assert result["ties"] == 2
    assert result["losses"] == 0


def test_win_count_and_mean_delta_capture_asymmetric_magnitudes():
    result = aggregate.summarize_deltas(
        np.array([0.51, 0.51, 0.0]),
        np.array([0.5, 0.5, 1.0]),
    )

    assert result["wins"] == 2
    assert result["losses"] == 1
    assert result["mean_delta"] < 0.0
