"""Unit tests for the QARIMA numeric core (fast, no network, no data files)."""

from __future__ import annotations

import numpy as np
import pytest
from lib.metrics import diebold_mariano, mape, mse
from lib.qarima import (
    LossWeights,
    ar_loss_fn,
    fit_and_forecast,
    forecast_oos,
    quantum_acf,
)
from lib.refiners import _apply_cnot, _apply_ry, _z_expectations, make_refiner
from lib.swaptest import cosine_similarity, swap_test_cosine


def test_swap_test_equals_cosine_in_analytic_limit():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, t = rng.normal(size=5), rng.normal(size=5)
        assert swap_test_cosine(x, t) == pytest.approx(cosine_similarity(x, t))


def test_ry_expectation_is_cosine():
    for th in (0.0, 0.7, np.pi / 2, np.pi):
        s = np.zeros(2, dtype=complex)
        s[0] = 1
        s = _apply_ry(s, 0, th, 1)
        assert _z_expectations(s, 1)[0] == pytest.approx(np.cos(th), abs=1e-9)


def test_bell_state_via_cnot():
    s = np.zeros(4, dtype=complex)
    s[0] = 1
    s = _apply_ry(s, 0, np.pi / 2, 2)
    s = _apply_cnot(s, 0, 1, 2)
    assert np.allclose(np.abs(s) ** 2, [0.5, 0, 0, 0.5], atol=1e-9)


def test_ar_loss_analytic_is_ols_when_unregularised():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    b_true = np.array([0.5, -0.2, 0.1])
    y = X @ b_true + 0.01 * rng.normal(size=200)
    loss = ar_loss_fn(X, y, LossWeights(lambda_cos=0, lambda_ent=0))
    b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    # OLS minimises the (analytic) prediction-error loss
    assert loss(b_ols) <= loss(b_true) + 1e-6


def test_forecast_recovers_linear_recursion_d0():
    # y_t = 0.5 y_{t-1}: AR(1), d=0 -> forecast decays geometrically
    y = np.array([1.0])
    for _ in range(40):
        y = np.append(y, 0.5 * y[-1])
    b = np.array([0.5])
    preds = forecast_oos(y, 5, p=1, d=0, q=0, b=b, theta=np.zeros(0))
    expected = [0.5 * y[-6] * 0.5**k for k in range(5)]
    assert np.allclose(preds, expected, atol=1e-8)


def test_difference_and_integrate_roundtrip_via_forecast():
    # A pure linear trend has a constant first difference (=2). ARIMA(0,1,0)
    # without drift forecasts flat (last value); AR(1) on the differences with
    # coefficient 1 propagates the constant difference and recovers the trend.
    y = np.arange(30, dtype=float) * 2.0 + 5.0
    flat = forecast_oos(y, 5, p=0, d=1, q=0, b=np.zeros(0), theta=np.zeros(0))
    assert np.allclose(flat, y[-6], atol=1e-8)  # flat at last train value
    trend = forecast_oos(y, 5, p=1, d=1, q=0, b=np.array([1.0]), theta=np.zeros(0))
    assert np.allclose(trend, y[-5:], atol=1e-8)  # trend recovered


def test_quantum_acf_lag0_is_one():
    rng = np.random.default_rng(2)
    w = rng.normal(size=100)
    acf = quantum_acf(w, 5)
    assert acf[0] == pytest.approx(1.0)


@pytest.mark.parametrize("refiner", ["classical", "gate", "merlin"])
def test_refiners_run_on_synthetic_ar(refiner):
    rng = np.random.default_rng(3)
    y = [0.0, 0.0]
    for _ in range(120):
        y.append(0.6 * y[-1] - 0.2 * y[-2] + 0.1 * rng.normal())
    y = np.array(y)
    r = make_refiner(refiner, reps=1, max_iter=30, step_frac=0.5)
    fr = fit_and_forecast(
        y,
        20,
        p=2,
        d=0,
        q=0,
        refiner=r,
        weights=LossWeights(lambda_cos=0, lambda_ent=0),
        seed=0,
    )
    assert fr.y_pred.shape == (20,)
    assert np.isfinite(fr.y_pred).all()


def test_dm_symmetry():
    rng = np.random.default_rng(4)
    y = rng.normal(size=50)
    a = y + 0.1 * rng.normal(size=50)
    b = y + 0.1 * rng.normal(size=50)
    dm_ab, p_ab = diebold_mariano(y, a, b, "mse")
    dm_ba, p_ba = diebold_mariano(y, b, a, "mse")
    assert dm_ab == pytest.approx(-dm_ba, abs=1e-9)
    assert p_ab == pytest.approx(p_ba, abs=1e-9)


def test_metrics_basic():
    y = np.array([1.0, 2.0, 4.0])
    assert mse(y, y) == 0.0
    assert mape(y, y) == 0.0
