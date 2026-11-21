"""Decay-model comparison for circuit-variance curves (claim C3, paper Eq. 13).

The paper conjectures that ``Var_circ(M)`` of a fixed-order PNM decays
polynomially in ``M`` past its peak. On the decaying branch we fit

* power law: ``log Var = a + b log M``;
* exponential: ``log Var = a + c M``;

by least squares in log space and compare them with AIC
(``n log(RSS / n) + 2k``, ``k = 2`` for both). The predeclared rule (LOG.md, Fair
Baseline Plan) is: polynomial decay is preferred when the power-law AIC is lower
than the exponential AIC by more than 2; the exponential is preferred when the
reverse holds; otherwise the comparison is inconclusive.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

AIC_MARGIN = 2.0


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope * x)
    rss = float(np.sum(residuals**2))
    total = float(np.sum((y - y.mean()) ** 2))
    count = len(x)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "rss": rss,
        "r_squared": 1.0 - rss / total if total > 0 else 1.0,
        "aic": count * np.log(max(rss, 1e-300) / count) + 4.0,
    }


def compare_decay_models(
    modes: Sequence[int], variances: Sequence[float], min_points: int
) -> dict[str, object]:
    """Fit power-law and exponential decay past the peak and pick a model.

    Parameters
    ----------
    modes : sequence of int
        Mode counts ``M`` (increasing).
    variances : sequence of float
        Positive circuit variances at those ``M``.
    min_points : int
        Minimum number of points on the decaying branch for a verdict.

    Returns
    -------
    dict
        Peak location, fitted parameters of both models, and ``preferred``
        (``"power_law"``, ``"exponential"``, ``"inconclusive"`` or
        ``"insufficient_points"``).
    """
    modes_array = np.asarray(modes, dtype=np.float64)
    variance_array = np.asarray(variances, dtype=np.float64)
    if np.any(variance_array <= 0):
        raise ValueError("Variances must be positive for a log-space fit")
    peak_index = int(np.argmax(variance_array))
    branch_modes = modes_array[peak_index:]
    branch_values = np.log(variance_array[peak_index:])
    record: dict[str, object] = {
        "peak_modes": int(modes_array[peak_index]),
        "peak_value": float(variance_array[peak_index]),
        "points_on_decaying_branch": int(len(branch_modes)),
    }
    if len(branch_modes) < min_points:
        record["preferred"] = "insufficient_points"
        return record
    power_law = _linear_fit(np.log(branch_modes), branch_values)
    exponential = _linear_fit(branch_modes, branch_values)
    record["power_law"] = {**power_law, "exponent": power_law["slope"]}
    record["exponential"] = {**exponential, "rate": exponential["slope"]}
    difference = exponential["aic"] - power_law["aic"]
    record["aic_exponential_minus_power_law"] = float(difference)
    if difference > AIC_MARGIN:
        record["preferred"] = "power_law"
    elif difference < -AIC_MARGIN:
        record["preferred"] = "exponential"
    else:
        record["preferred"] = "inconclusive"
    return record


def local_loglog_slopes(
    modes: Sequence[int], variances: Sequence[float]
) -> list[dict[str, float]]:
    """Slopes ``d log Var / d log M`` between consecutive points.

    A power law ``Var ~ M^{-p}`` has local slopes tending to the constant
    ``-p``; an exponential ``e^{-cM}`` has slopes ``-cM`` that keep growing in
    magnitude. Over a short range both global fits can look good, so the trend
    of the local slope is the sharper diagnostic of the asymptotic decay.

    Parameters
    ----------
    modes : sequence of int
        Increasing mode counts.
    variances : sequence of float
        Positive variances at those ``M``.

    Returns
    -------
    list of dict
        ``{"modes_from", "modes_to", "slope"}`` per consecutive pair.
    """
    modes_array = np.asarray(modes, dtype=np.float64)
    values = np.asarray(variances, dtype=np.float64)
    slopes = np.diff(np.log(values)) / np.diff(np.log(modes_array))
    return [
        {"modes_from": int(left), "modes_to": int(right), "slope": float(slope)}
        for left, right, slope in zip(modes_array[:-1], modes_array[1:], slopes)
    ]
