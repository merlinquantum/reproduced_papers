"""Regression metrics of the paper (Sec. IV-A, Eq. 20-23), in AQI units."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE (percent) and R^2.

    Parameters
    ----------
    y_true : numpy.ndarray
        Observed AQI values, shape ``(n,)``.
    y_pred : numpy.ndarray
        Predicted AQI values, shape ``(n,)``.

    Returns
    -------
    dict
        ``rmse``, ``mae``, ``mape`` (Eq. 22, in percent), ``r2`` (Eq. 23) and
        ``n`` (number of scored points).

    Raises
    ------
    ValueError
        If shapes differ, the input is empty, or an observed value is zero
        (MAPE undefined).
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty set")
    if np.any(y_true == 0):
        raise ValueError("MAPE is undefined when an observed value is zero")
    errors = y_true - y_pred
    total_sum_squares = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "mape": float(100.0 * np.mean(np.abs(errors / y_true))),
        "r2": float(1.0 - np.sum(errors**2) / total_sum_squares),
        "n": int(y_true.size),
    }
