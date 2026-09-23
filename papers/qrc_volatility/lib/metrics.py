"""Forecast-evaluation metrics and tests used in paper Tables II and III.

Metric definitions follow the authors' released code rather than the paper text
wherever the two disagree; the disagreement is recorded in ``LOG.md``.

* ``MSE`` is the mean squared error on raw ``log RV`` values (the reference Julia
  code multiplies squared errors on the ``[-1, 0]`` scale by
  ``(max log RV - min log RV) ** 2``, which is identical).
* ``QLIKE`` as *reported* is the code's ``compute_qlike``: a **sum** over the
  out-of-sample period of ``r - log r - 1`` with ``r = |RV| / |RV_hat|``
  evaluated on ``log RV`` values. This is not the paper's Eq. for QLIKE (which is
  a mean over ``log(RV_hat^2) + RV^2 / RV_hat^2``), but it is the quantity that
  reproduces the published table to four decimals.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def mse(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Mean squared error on raw ``log RV`` units."""
    return float(np.mean((np.asarray(forecast) - np.asarray(actual)) ** 2))


def qlike(forecast: np.ndarray, actual: np.ndarray) -> float:
    """QLIKE as computed by the authors' code (summed, on ``log RV`` values)."""
    ratio = np.abs(np.asarray(actual)) / np.abs(np.asarray(forecast))
    return float(np.sum(ratio - np.log(ratio) - 1.0))


def qlike_losses(forecast: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE loss series (for MCS and DM tests)."""
    ratio = np.abs(np.asarray(actual)) / np.abs(np.asarray(forecast))
    return ratio - np.log(ratio) - 1.0


def hit_rate(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of correctly signed month-over-month changes."""
    return float(
        np.mean((np.diff(np.asarray(forecast)) > 0) == (np.diff(np.asarray(actual)) > 0))
    )


def summarise(forecast: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Standard metric bundle reported for every model in this reproduction."""
    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return {
        "mse": mse(forecast, actual),
        "rmse": float(np.sqrt(mse(forecast, actual))),
        "mae": float(np.mean(np.abs(forecast - actual))),
        "qlike": qlike(forecast, actual),
        "hit_rate": hit_rate(forecast, actual),
        "n_observations": int(len(actual)),
    }


def diebold_mariano(actual: np.ndarray, first: np.ndarray, second: np.ndarray):
    """Diebold-Mariano test on squared-error loss differentials.

    Reproduces the authors' ``diebold_mariano_test``: the loss differential
    variance is the plain sample variance divided by ``T`` (no Newey-West
    correction, despite the paper's text) and the reference distribution is
    Student ``t`` with ``T - 1`` degrees of freedom.

    Returns
    -------
    tuple of float
        ``(statistic, p_value)``. A positive statistic favours ``second``.
    """
    from scipy.stats import t as student_t

    differential = (np.asarray(actual) - np.asarray(first)) ** 2 - (
        np.asarray(actual) - np.asarray(second)
    ) ** 2
    variance = np.var(differential, ddof=1)
    if variance == 0.0:
        # Identical forecast paths: the loss differential is degenerate, so the
        # null of equal predictive ability cannot be rejected.
        return 0.0, 1.0
    statistic = float(np.mean(differential) / np.sqrt(variance / len(differential)))
    p_value = float(2.0 * student_t.cdf(-abs(statistic), df=len(differential) - 1))
    return statistic, p_value


def model_confidence_set(
    losses: dict[str, np.ndarray],
    size: float = 0.05,
    reps: int = 10_000,
    seed: int = 0,
    min_observations: int = 60,
) -> dict[str, float]:
    """Hansen-Lunde-Nason Model Confidence Set p-values.

    Parameters
    ----------
    losses : dict of str to numpy.ndarray
        Per-observation loss series, one entry per model.
    size : float
        MCS significance level. Default value is 0.05.
    reps : int
        Stationary-bootstrap replications. Default value is 10000.
    seed : int
        Bootstrap seed. The reference notebook left this unset, so the published
        p-values are not exactly reproducible. Default value is 0.
    min_observations : int
        Below this sample length the stationary bootstrap in ``arch`` cannot
        eliminate any model and raises; the test is then skipped and every
        p-value is ``nan``. Default value is 60.

    Returns
    -------
    dict of str to float
        MCS p-value per model name, or ``nan`` when the test was skipped.
    """
    import pandas as pd
    from arch.bootstrap import MCS

    frame = pd.DataFrame(losses)
    if len(frame) < min_observations:
        logger.warning(
            "MCS_SKIPPED | n_observations=%d < min_observations=%d | reason=sample "
            "too short for the stationary bootstrap",
            len(frame), min_observations,
        )
        return dict.fromkeys(frame.columns, float("nan"))

    mcs = MCS(frame, size=size, reps=reps, method="R", bootstrap="stationary", seed=seed)
    mcs.compute()
    p_values = mcs.pvalues.iloc[:, 0].to_dict()
    logger.info(
        "MCS_COMPLETED | models=%d | reps=%d | included=%s",
        len(frame.columns), reps, sorted(mcs.included),
    )
    return {name: float(p_values[name]) for name in frame.columns}
