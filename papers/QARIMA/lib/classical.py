"""Classical ARIMA baselines (paper's ``pmdarima`` comparator).

Provides the paper's fair reference: a non-seasonal ``auto_arima`` order selected
on the training set, evaluated with the *same* rolling one-step OOS protocol as
QARIMA (fixed parameters, realised past).  Also exposes a seasonal auto_arima for
the baseline-fairness probe on seasonal series (LOG.md C3/F6).
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def auto_order(y_train: np.ndarray, seasonal: bool = False, m: int = 1):
    import pmdarima as pm

    model = pm.auto_arima(
        y_train,
        seasonal=seasonal,
        m=m if seasonal else 1,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=10,
        max_q=6,
        max_d=2,
        information_criterion="aic",
    )
    return model.order, getattr(model, "seasonal_order", (0, 0, 0, 0))


def dynamic_forecast(
    y_full: np.ndarray, n_oos: int, order, seasonal_order=(0, 0, 0, 0)
):
    """Fit ARIMA(order) on the train split, then forecast the whole OOS horizon.

    Multi-step dynamic forecast from a single origin (end of training), matching
    the QARIMA protocol and the paper's error magnitudes.
    """
    from statsmodels.tsa.arima.model import ARIMA

    y_full = np.asarray(y_full, dtype=np.float64)
    y_train = y_full[:-n_oos]
    res = ARIMA(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    preds = np.asarray(res.forecast(n_oos), dtype=np.float64)
    return preds, order, seasonal_order
