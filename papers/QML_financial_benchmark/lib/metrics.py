"""Classification, forecasting and trading metrics of arXiv:2601.03802."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

TRADING_DAYS_PER_YEAR = 252


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC, or NaN when only one class is present."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict:
    """Accuracy, AUC, precision and recall (§III-C4) plus the class balance."""
    predictions = (probabilities > threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": safe_auc(labels, probabilities),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "positive_rate": float(np.mean(labels)),
        "predicted_positive_rate": float(np.mean(predictions)),
        "n": int(len(labels)),
    }


# ---------------------------------------------------------------------------
# Volatility forecasting (§III-E4)
# ---------------------------------------------------------------------------


def qlike(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Per-day QLIKE loss ``log(RV_hat) + RV / RV_hat`` (Eq. 20)."""
    return np.log(forecast) + realized / forecast


def forecast_metrics(
    realized: np.ndarray, forecast: np.ndarray, previous_realized: np.ndarray
) -> dict:
    """QLIKE, MSE, MAE, R^2 and directional accuracy of variance forecasts.

    DirAcc is the share of days where ``sign(RV_hat_{t+1} - RV_t)`` equals
    ``sign(RV_{t+1} - RV_t)``; the paper names the metric without defining it.
    """
    errors = realized - forecast
    total_variance = np.sum((realized - realized.mean()) ** 2)
    return {
        "qlike": float(np.mean(qlike(realized, forecast))),
        "mse": float(np.mean(errors**2)),
        "mae": float(np.mean(np.abs(errors))),
        "r2": float(1.0 - np.sum(errors**2) / total_variance),
        "directional_accuracy": float(
            np.mean(
                np.sign(forecast - previous_realized)
                == np.sign(realized - previous_realized)
            )
        ),
        "n": int(len(realized)),
    }


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1
) -> tuple[float, float]:
    """Two-sided Diebold-Mariano test on per-day losses with the HLN correction.

    Adapted from ``papers/QARIMA/lib/metrics.py::diebold_mariano`` to take loss
    series directly, so QLIKE differentials can be tested. A positive statistic
    means ``a`` has the larger loss (``b`` forecasts better).

    Returns
    -------
    tuple[float, float]
        ``(statistic, p_value)``; NaNs when the long-run variance is not positive.
    """
    differential = np.asarray(loss_a, dtype=np.float64) - np.asarray(
        loss_b, dtype=np.float64
    )
    n = differential.size
    mean = float(np.mean(differential))
    variance = float(np.mean((differential - mean) ** 2))
    for lag in range(1, horizon):
        covariance = float(
            np.mean((differential[lag:] - mean) * (differential[:-lag] - mean))
        )
        variance += 2.0 * (1.0 - lag / horizon) * covariance
    if variance <= 0 or n < 2:
        return float("nan"), float("nan")
    statistic = mean / np.sqrt(variance / n)
    statistic *= np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1))
    return float(statistic), float(p_value)


# ---------------------------------------------------------------------------
# Trading (§III-D4, §III-D6)
# ---------------------------------------------------------------------------


def positions_from_probabilities(
    probabilities: np.ndarray, tau_long: float, tau_short: float
) -> np.ndarray:
    """Eq. 4: +1 above ``tau_long``, -1 below ``tau_short``, 0 otherwise."""
    positions = np.zeros(len(probabilities))
    positions[probabilities > tau_long] = 1.0
    positions[probabilities < tau_short] = -1.0
    return positions


def strategy_returns(
    positions: np.ndarray, next_returns: np.ndarray, fee: float
) -> np.ndarray:
    """Daily net strategy returns ``s_t r_{t+1} - Cost_{t+1}`` (Eq. 5).

    The fee is charged on the absolute position change, with the strategy
    flat before the first day, so a long-to-short flip pays twice.
    """
    previous = np.concatenate([[0.0], positions[:-1]])
    return positions * next_returns - fee * np.abs(positions - previous)


def trading_metrics(
    daily_returns: np.ndarray, positions: np.ndarray | None = None
) -> dict:
    """ARC, ASD, Sharpe, Sortino and maximum drawdown with ``r_f = 0``.

    ARC is the geometric annualised return ``R_tot^(252/T) - 1``; ASD the
    annualised standard deviation of daily returns; Sortino divides ARC by the
    annualised root-mean-square of negative daily returns.
    """
    n_days = len(daily_returns)
    growth = np.prod(1.0 + daily_returns)
    annualized_return = (
        growth ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0 if growth > 0 else -1.0
    )
    annualized_std = float(
        np.std(daily_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    downside = np.sqrt(np.mean(np.minimum(daily_returns, 0.0) ** 2)) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    equity = np.cumprod(1.0 + daily_returns)
    drawdown = equity / np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:] - 1.0
    result = {
        "arc": float(annualized_return),
        "asd": annualized_std,
        "sharpe": float(annualized_return / annualized_std)
        if annualized_std > 0
        else 0.0,
        "sortino": float(annualized_return / downside) if downside > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "n_days": int(n_days),
    }
    if positions is not None:
        previous = np.concatenate([[0.0], positions[:-1]])
        result["n_position_changes"] = int(np.count_nonzero(positions != previous))
        result["share_long"] = float(np.mean(positions > 0))
        result["share_short"] = float(np.mean(positions < 0))
    return result
