from __future__ import annotations

import numpy as np
from lib.tasks.trading import trading_candidates
from lib.tasks.volatility import garch_forecasts


def test_trading_grid_and_explicit_candidates():
    grid = trading_candidates(
        {
            "qlstm": {"n_qubits": [3, 4], "layers": [2]},
            "lstm": {"hidden_size": [3], "layers": [2, 3]},
        }
    )
    assert [c["name"] for c in grid] == [
        "qlstm_q3_L2",
        "qlstm_q4_L2",
        "lstm_h3_L2",
        "lstm_h3_L3",
    ]
    explicit = trading_candidates(
        {"qlstm": {"configs": [[5, 6]]}, "lstm": {"configs": [[4, 2], [5, 4]]}}
    )
    assert [c["name"] for c in explicit] == ["qlstm_q5_L6", "lstm_h4_L2", "lstm_h5_L4"]


def test_garch_sigma2_variant_drops_the_known_part():
    returns = np.random.default_rng(0).normal(0.0, 0.01, 600)
    known = np.convolve(returns**2, np.ones(4), mode="full")[: len(returns)]
    train, test = np.arange(500), np.arange(500, 600)
    completed = garch_forecasts(returns, known, train, test)
    sigma2 = garch_forecasts(returns, known, train, test, complete_with_known=False)
    assert np.allclose(completed, (known[test] + sigma2) / 5.0)
    assert np.all(sigma2 > 0)
