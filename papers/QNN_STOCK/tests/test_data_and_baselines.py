from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the repository root on sys.path)
from lib.baselines import ConditionalModel, frequency_table
from lib.data import (
    context_indices,
    joint_distribution,
    price_moves,
    symbolize,
)
from lib.metrics import distribution_metrics, kl_divergence, sequence_metrics


def test_moving_average_then_difference_matches_paper_sample_count():
    prices = pd.Series(np.arange(10033, dtype=float))
    assert len(price_moves(prices, 5)) == 10028  # 10,029 smoothed prices
    assert len(price_moves(prices, 1)) == 10032


def test_sign_symbols_send_zero_moves_to_level_zero():
    symbols, edges = symbolize(np.array([-1.0, 0.0, 2.0]), 2, "sign", np.array([]))
    assert symbols.tolist() == [0, 0, 1]
    assert edges is None


def test_quantile_symbols_are_balanced_on_training_moves():
    moves = np.random.default_rng(0).normal(size=4000)
    symbols, edges = symbolize(moves, 4, "quantile", moves)
    assert len(edges) == 3
    assert np.allclose(np.bincount(symbols) / len(symbols), 0.25, atol=0.01)


def test_context_index_uses_oldest_step_as_most_significant():
    windows = np.array([[1, 0, 0, 1], [0, 0, 1, 0]])
    assert context_indices(windows, 2).tolist() == [4, 1]
    joint = joint_distribution(windows, 2, 2)
    assert joint[4, 1] == 0.5 and joint[1, 0] == 0.5


def test_frequency_table_reproduces_training_target_exactly():
    rng = np.random.default_rng(3)
    windows = rng.integers(0, 2, size=(500, 4))
    target = joint_distribution(windows, 2, 2)
    table = frequency_table(windows, 2, 2, alpha=0.0)
    metrics = distribution_metrics(target, target.sum(1, keepdims=True) * table, table)
    assert metrics["kl_conditional_sum"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["kl_joint"] == pytest.approx(0.0, abs=1e-12)


def test_kl_and_sequence_metrics_on_known_values():
    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == pytest.approx(np.log(2))
    windows = np.array([[0, 0, 0, 1]])
    table = np.full((8, 2), 0.5)
    result = sequence_metrics(windows, 2, table)
    assert result["log_loss"] == pytest.approx(np.log(2))


@pytest.mark.parametrize("hidden, expected", [(0, 4), (3, 16), (6, 31)])
def test_classical_parameter_counts_match_quantum_budgets(hidden, expected):
    assert ConditionalModel(3, 2, hidden).n_params() == expected
