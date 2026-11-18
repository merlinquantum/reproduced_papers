from __future__ import annotations

import numpy as np
from lib.metrics import (
    diebold_mariano,
    positions_from_probabilities,
    qlike,
    strategy_returns,
    trading_metrics,
)


def test_positions_follow_equation_4():
    positions = positions_from_probabilities(
        np.array([0.95, 0.6, 0.3, 0.05]), tau_long=0.7, tau_short=0.2
    )
    assert positions.tolist() == [1.0, 0.0, 0.0, -1.0]


def test_fee_charged_on_each_unit_of_position_change():
    returns = strategy_returns(np.array([1.0, -1.0, -1.0]), np.zeros(3), fee=0.001)
    assert np.allclose(returns, [-0.001, -0.002, 0.0])


def test_trading_metrics_of_constant_growth():
    metrics = trading_metrics(np.full(252, 0.001))
    assert np.isclose(metrics["arc"], 1.001**252 - 1)
    assert metrics["max_drawdown"] == 0.0


def test_qlike_is_minimized_by_the_true_variance():
    realized = np.array([2.0])
    assert qlike(realized, realized) < qlike(realized, np.array([1.5]))
    assert qlike(realized, realized) < qlike(realized, np.array([2.5]))


def test_diebold_mariano_sign_and_symmetry():
    rng = np.random.default_rng(0)
    worse, better = rng.random(500) + 0.2, rng.random(500)
    statistic, p_value = diebold_mariano(worse, better)
    reverse, reverse_p = diebold_mariano(better, worse)
    assert statistic > 0 and p_value < 0.05
    assert np.isclose(statistic, -reverse) and np.isclose(p_value, reverse_p)
