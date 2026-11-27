"""The Diebold-Mariano test must behave like the QARIMA estimator."""

from __future__ import annotations

import math

import common  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import pytest

from papers.QuLTSF.lib.statistics import diebold_mariano


def _reference_dm(loss_a, loss_b, h):
    """Independent restatement of papers/QARIMA/lib/metrics.py::diebold_mariano."""
    from scipy import stats

    d = loss_a - loss_b
    n, d_bar = d.size, d.mean()
    var = np.mean((d - d_bar) ** 2)
    for k in range(1, h):
        var += 2 * (1 - k / h) * np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
    dm = d_bar / np.sqrt(var / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm, 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))


@pytest.mark.parametrize("horizon", [1, 5, 24])
def test_matches_reference_estimator(horizon):
    rng = np.random.default_rng(0)
    loss_a = rng.gamma(2.0, 1.0, size=500)
    loss_b = loss_a * 0.95 + rng.normal(0, 0.1, size=500)
    statistic, p_value = diebold_mariano(loss_a, loss_b, horizon)
    expected_statistic, expected_p = _reference_dm(loss_a, loss_b, horizon)
    assert statistic == pytest.approx(expected_statistic, rel=1e-12)
    assert p_value == pytest.approx(expected_p, rel=1e-9, abs=1e-15)


def test_sign_points_to_the_better_model():
    rng = np.random.default_rng(1)
    worse = rng.gamma(2.0, 1.0, size=400) + 0.5
    better = worse - 0.5 + rng.normal(0, 0.05, size=400)
    statistic, p_value = diebold_mariano(worse, better, horizon=3)
    assert statistic > 0 and p_value < 1e-6
    statistic_swapped, _ = diebold_mariano(better, worse, horizon=3)
    assert statistic_swapped == pytest.approx(-statistic)


def test_identical_losses_are_undefined():
    losses = np.ones(50)
    assert all(math.isnan(v) for v in diebold_mariano(losses, losses, 4))


def test_misaligned_series_fail():
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(10), np.ones(11), 2)
