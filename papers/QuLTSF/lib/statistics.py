"""Forecast-comparison statistics on per-window test losses."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, horizon: int
) -> tuple[float, float]:
    """Two-sided Diebold-Mariano test of equal forecast accuracy.

    Same estimator as ``papers/QARIMA/lib/metrics.py::diebold_mariano``
    (Newey-West long-run variance up to lag ``horizon - 1`` with Bartlett
    weights, Harvey-Leybourne-Newbold small-sample correction, Student-t
    reference with ``n - 1`` degrees of freedom), but taking per-window losses
    directly. Consecutive LTSF test windows start one step apart and forecast
    ``horizon`` steps, so their losses overlap and are autocorrelated up to
    lag ``horizon - 1``; pass the forecast horizon ``T`` as ``horizon``.

    Parameters
    ----------
    loss_a : numpy.ndarray
        Per-window losses of model A, in time order, shape ``[n]``.
    loss_b : numpy.ndarray
        Per-window losses of model B on the same windows, shape ``[n]``.
    horizon : int
        Forecast horizon ``T``; sets the autocorrelation lag window.

    Returns
    -------
    tuple[float, float]
        ``(dm_statistic, p_value)``. A positive statistic means A has the
        larger loss (B is better). ``(nan, nan)`` when the long-run variance
        is not positive.

    Raises
    ------
    ValueError
        If the loss series differ in shape, are not 1-D, are shorter than two
        windows, or ``horizon`` is not positive.
    """
    loss_a = np.asarray(loss_a, dtype=np.float64)
    loss_b = np.asarray(loss_b, dtype=np.float64)
    if loss_a.shape != loss_b.shape or loss_a.ndim != 1:
        raise ValueError(
            f"Loss series must be 1-D and aligned, got {loss_a.shape} and {loss_b.shape}"
        )
    if loss_a.size < 2:
        raise ValueError("Need at least two windows for a Diebold-Mariano test")
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")

    differential = loss_a - loss_b
    n = differential.size
    mean = float(differential.mean())
    centred = differential - mean
    variance = float(np.mean(centred**2))
    for lag in range(1, min(horizon, n)):
        autocovariance = float(np.mean(centred[lag:] * centred[:-lag]))
        variance += 2.0 * (1.0 - lag / horizon) * autocovariance
    if variance <= 0:
        return math.nan, math.nan
    statistic = mean / math.sqrt(variance / n)
    statistic *= math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    p_value = 2.0 * float(stats.t.sf(abs(statistic), df=n - 1))
    return float(statistic), p_value
