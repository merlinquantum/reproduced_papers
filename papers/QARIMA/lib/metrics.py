"""Forecast error metrics and the Diebold--Mariano test (paper Sec. 5.1--5.2)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error as a fraction (paper reports e.g. 0.0228)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)))


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    loss: str = "mse",
    h: int = 1,
) -> tuple[float, float]:
    """Diebold--Mariano test comparing forecast ``a`` (e.g. classical) vs ``b``.

    Returns ``(dm_stat, p_value)`` for the two-sided test of equal accuracy.  A
    positive statistic means ``a`` has larger loss (i.e. ``b`` is better).  Uses
    the Harvey--Leybourne--Newbold small-sample correction.

    Returns ``(NaN, NaN)`` if the long-run variance is zero or non-positive
    (degenerate case where the test is undefined), or if sample size < 2.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    ea = y_true - np.asarray(pred_a, dtype=np.float64)
    eb = y_true - np.asarray(pred_b, dtype=np.float64)
    if loss == "mse":
        d = ea**2 - eb**2
    elif loss == "mae":
        d = np.abs(ea) - np.abs(eb)
    else:
        raise ValueError(f"loss must be 'mse' or 'mae', got {loss!r}")

    n = d.size
    d_bar = float(np.mean(d))
    # Newey-West long-run variance up to lag h-1.
    gamma0 = float(np.mean((d - d_bar) ** 2))
    var = gamma0
    for k in range(1, h):
        cov = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        var += 2.0 * (1.0 - k / h) * cov
    if var <= 0 or n < 2:
        # Degenerate case: return NaNs to indicate the test is undefined.
        return float("nan"), float("nan")
    dm = d_bar / np.sqrt(var / n)
    # Harvey-Leybourne-Newbold correction.
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= correction
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)
