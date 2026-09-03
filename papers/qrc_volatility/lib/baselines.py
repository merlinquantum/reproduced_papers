"""Classical benchmark models of paper Table II, plus two extra fair controls.

Paper baselines
---------------
``har``, ``harx``, ``ar1``, ``ar3``, ``armax`` (linear econometric), ``lstm``,
``lstmx`` (2-layer LSTM), ``rc``, ``rcx`` (echo state network). All are
re-estimated at every rolling forecast origin, matching the reference notebooks.

Extra controls added by this reproduction
----------------------------------------
``linear_lag`` and ``esn_iso`` exist because the paper's classical models do not
match the quantum reservoir on the comparison axis its advantage is claimed on:

* ``linear_lag`` applies the *same* rolling ridge readout to the *raw* lagged
  features the quantum reservoir is given. It isolates whether the reservoir's
  nonlinear feature map contributes anything beyond a linear map of the same
  inputs.
* ``esn_iso`` is a classical echo state network with exactly ``n_qubits *
  virtual_nodes`` readout units, driven by the same feature subset and the same
  3-step window, read out by the same rolling ridge. It matches the QRC on
  readout dimension, trainable-parameter count, inputs, and selection protocol.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

HAR_COLUMNS = ("RV_lag1", "RV_quarterly_lag", "RV_annual_lag")
# Exogenous set used by the authors' HARX notebook cell.
HARX_EXOGENOUS = ("diff_DP", "MKT", "STR", "DEF")
# Feature list used by the authors' LSTM / classical-reservoir notebooks.
NEURAL_FEATURES = (
    "RV", "MKT", "diff_DP", "IP", "DEF", "EP", "SMB", "diff_TB", "HML", "INF", "STR",
)


# --------------------------------------------------------------------------- #
# Linear econometric models
# --------------------------------------------------------------------------- #
def _har_design(rv_history: np.ndarray) -> np.ndarray:
    """HAR regressors ``[RV_{t-1}, mean(RV_{t-3..t-1}), mean(RV_{t-12..t-1})]``."""
    return np.array(
        [rv_history[-1], rv_history[-3:].mean(), rv_history[-12:].mean()]
    )


def har_forecasts(
    frame: pd.DataFrame,
    train_slices,
    origin_index,
    horizons=(1,),
    exogenous: tuple[str, ...] = (),
    window_start: int = 0,
    reference_misalignment: bool = False,
) -> dict[int, np.ndarray]:
    """Rolling HAR / HARX forecasts, closed-loop for horizons above one.

    Parameters
    ----------
    frame : pandas.DataFrame
        Regressor frame from :func:`lib.data.build_regressor_frame`.
    train_slices : sequence of (int, int)
        Training ranges, one per forecast origin.
    origin_index : sequence of int
        First row predicted by each path.
    horizons : sequence of int
        Forecast horizons ``S``. Default value is ``(1,)``.
    exogenous : tuple of str
        Extra lag-1 regressors; empty for plain HAR. Default value is ``()``.
    window_start : int
        Row offset of the first training window, kept so the authors' HARX
        window (which starts at row 277) can be reproduced. Default value is 0.
    reference_misalignment : bool
        When True, evaluate the regressor row one month *before* the forecast
        target, reproducing the off-by-one indexing of the authors' notebook
        (``X_test = dff.iloc[rolling_window_end:rolling_window_end + 1]`` while
        ``actual`` starts one row later). This is required to reproduce the
        published HAR/HARX numbers but is a defect, not the intended model;
        see LOG.md claim C6. Default value is False.

    Returns
    -------
    dict of int to numpy.ndarray
        Forecasts per horizon, aligned with ``origin_index``.
    """
    rv = frame["RV"].to_numpy()
    exog = frame[list(exogenous)].to_numpy() if exogenous else None
    columns = list(HAR_COLUMNS) + [f"{name}_lag1" for name in exogenous]
    design_frame = frame.copy()
    for name in exogenous:
        design_frame[f"{name}_lag1"] = design_frame[name].shift(1)
    design_frame = design_frame.fillna(0.0)
    offset = 1 if reference_misalignment else 0

    label = "HARX" if exogenous else "HAR"
    if reference_misalignment:
        label += " (as-published indexing)"
    logger.info("EVALUATION_STARTED | model=%s | regressors=%d | refits=%d",
                label, len(columns) + 1, len(origin_index))
    out = {h: np.empty(len(origin_index)) for h in horizons}
    for position, ((low, high), origin) in enumerate(zip(train_slices, origin_index)):
        low = max(low, window_start)
        train = design_frame.iloc[low:high - offset]
        model = sm.OLS(train["RV"], sm.add_constant(train[columns])).fit()
        params = model.params.to_numpy()

        history = list(rv[: origin - offset])
        for step in range(max(horizons)):
            row = origin + step - offset
            regressors = [1.0, *_har_design(np.asarray(history))]
            if exog is not None:
                regressors.extend(exog[min(row - 1, len(exog) - 1)])
            prediction = float(np.dot(params, regressors))
            history.append(prediction)
            if step + 1 in out:
                out[step + 1][position] = prediction
    return out


def ar_forecasts(frame: pd.DataFrame, train_slices, origin_index, order: int, horizons=(1,)):
    """Rolling AR(``order``) forecasts with closed-loop multi-step extension."""
    rv = frame["RV"].to_numpy()
    logger.info("EVALUATION_STARTED | model=AR(%d) | refits=%d", order, len(origin_index))
    out = {h: np.empty(len(origin_index)) for h in horizons}
    for position, ((low, high), origin) in enumerate(zip(train_slices, origin_index)):
        series = pd.Series(rv[low:high])
        design = np.column_stack(
            [np.ones(high - low - order)]
            + [series.shift(lag).to_numpy()[order:] for lag in range(1, order + 1)]
        )
        params = np.linalg.lstsq(design, series.to_numpy()[order:], rcond=None)[0]
        history = list(rv[:origin])
        for step in range(max(horizons)):
            lags = [history[-lag] for lag in range(1, order + 1)]
            prediction = float(params[0] + np.dot(params[1:], lags))
            history.append(prediction)
            if step + 1 in out:
                out[step + 1][position] = prediction
    return out


def armax_forecasts(
    frame: pd.DataFrame, train_slices, origin_index, horizons=(1,), order=(1, 0, 0),
    log_every: int = 50,
):
    """Rolling ARMAX(1,0,0) with every non-target column entered at lag 1.

    The authors' notebook builds the exogenous block as
    ``dff.drop(columns=['RV']).shift(1)`` *after* the HAR and HARX helper columns
    have been added to ``dff``, so their ARMAX also sees the HAR lag terms and
    the HARX lag-1 regressors. That column set is reproduced here.

    statsmodels performs the multi-step recursion internally, so the closed-loop
    extension needs only ``steps=S``.
    """
    from statsmodels.tsa.arima.model import ARIMA

    exog = frame.drop(columns=["RV"]).shift(1).fillna(0.0)
    rv = frame["RV"]
    out = {h: np.empty(len(origin_index)) for h in horizons}
    max_h = max(horizons)
    logger.info("EVALUATION_STARTED | model=ARMAX | order=%s | n_exog=%d | refits=%d",
                order, exog.shape[1], len(origin_index))
    for position, ((low, high), origin) in enumerate(zip(train_slices, origin_index)):
        fitted = ARIMA(
            rv.iloc[low:high].to_numpy(),
            order=order,
            exog=exog.iloc[low:high].to_numpy(),
            trend="n",
        ).fit(method_kwargs={"warn_convergence": False})
        rows = np.clip(np.arange(origin, origin + max_h), 0, len(exog) - 1)
        path = fitted.forecast(steps=max_h, exog=exog.iloc[rows].to_numpy())
        for h in horizons:
            out[h][position] = float(path[h - 1])
        if (position + 1) % log_every == 0 or position + 1 == len(origin_index):
            logger.info("EVALUATION_PROGRESS | model=ARMAX | refit=%d/%d",
                        position + 1, len(origin_index))
    return out


# --------------------------------------------------------------------------- #
# Window models: ridge / ESN / LSTM readouts over a fixed 3-step window
# --------------------------------------------------------------------------- #
def linear_lag_forecasts(
    windows: np.ndarray, target: np.ndarray, train_slices, origin_index,
    horizons=(1,), rv_column: int = 0, ridge: float = 1e-8
):
    """Rolling ridge on the flattened raw lag window (no reservoir).

    Parameters
    ----------
    windows : numpy.ndarray, shape (T, n_lags, n_features)
        Same tensor the quantum reservoir consumes.
    target : numpy.ndarray, shape (T,)
        Normalised realized volatility.
    train_slices, origin_index : sequence
        Rolling schedule.
    horizons : sequence of int
        Forecast horizons. Default value is ``(1,)``.
    rv_column : int
        Index of ``RV`` inside the feature vector. Default value is 0.
    ridge : float
        Ridge regulariser, matched to the QRC readout. Default value is 1e-8.

    Returns
    -------
    dict of int to numpy.ndarray
    """
    n_times, n_lags, n_features = windows.shape
    flat = windows.reshape(n_times, -1)
    # Intercept column so the model is not forced through the origin; the QRC
    # readout gets its offset for free from the <Z> expectations.
    design = np.column_stack([flat, np.ones(n_times)])
    eye = np.eye(design.shape[1])

    origins = np.asarray(origin_index)
    weights = np.empty((len(origins), design.shape[1]))
    for position, (low, high) in enumerate(train_slices):
        block = design[low:high]
        weights[position] = np.linalg.solve(
            block.T @ block + ridge * eye, block.T @ target[low:high]
        )

    out = {}
    current = windows[origins].copy()
    predictions = np.zeros(len(origins))
    for step in range(max(horizons)):
        if step > 0:
            rows = np.clip(origins + step, 0, n_times - 1)
            current = np.roll(current, -1, axis=1)
            current[:, -1, :] = windows[rows][:, -1, :]
            current[:, -1, rv_column] = predictions
        features = np.column_stack([current.reshape(len(origins), -1), np.ones(len(origins))])
        predictions = np.einsum("pf,pf->p", weights, features)
        if step + 1 in horizons:
            out[step + 1] = predictions.copy()
    return out


class EchoStateReservoir:
    """Leaky-integrator echo state network matching the reference RC settings.

    ``h_t = (1 - lr) h_{t-1} + lr * tanh(W h_{t-1} + W_in x_t)`` with ``W``
    rescaled to the requested spectral radius. Only the final state of a window
    is read out, exactly as the authors' ``reservoirpy`` loop does
    (``reservoir.run(X, reset=True)[-1, :]``).
    """

    def __init__(
        self,
        n_units: int,
        n_inputs: int,
        *,
        leak_rate: float = 0.6,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        recurrent = rng.normal(size=(n_units, n_units))
        radius = np.max(np.abs(np.linalg.eigvals(recurrent)))
        self.recurrent = recurrent * (spectral_radius / radius)
        self.input_weights = rng.uniform(-1.0, 1.0, size=(n_units, n_inputs)) * input_scaling
        self.leak_rate = leak_rate
        self.n_units = n_units

    def evaluate(self, windows: np.ndarray) -> np.ndarray:
        """Final reservoir state for a batch of windows, shape ``(batch, n_units)``."""
        batch, n_lags, _ = windows.shape
        state = np.zeros((batch, self.n_units))
        for lag in range(n_lags):
            drive = windows[:, lag, :] @ self.input_weights.T + state @ self.recurrent.T
            state = (1.0 - self.leak_rate) * state + self.leak_rate * np.tanh(drive)
        return state


def esn_forecasts(
    windows: np.ndarray, target: np.ndarray, train_slices, origin_index,
    *, n_units: int, horizons=(1,), rv_column: int = 0, ridge: float = 1e-7,
    leak_rate: float = 0.6, spectral_radius: float = 0.9, input_scaling: float = 0.1,
    seed: int = 0,
):
    """Classical echo state network with the rolling ridge readout.

    Returns
    -------
    dict of int to numpy.ndarray
    """
    n_times = windows.shape[0]
    n_lags = windows.shape[1]
    reservoir = EchoStateReservoir(
        n_units, windows.shape[2], leak_rate=leak_rate,
        spectral_radius=spectral_radius, input_scaling=input_scaling, seed=seed,
    )
    states = np.zeros((n_times, n_units))
    states[n_lags:] = reservoir.evaluate(windows[n_lags:])

    origins = np.asarray(origin_index)
    eye = np.eye(n_units)
    weights = np.empty((len(origins), n_units))
    for position, (low, high) in enumerate(train_slices):
        block = states[low:high]
        weights[position] = np.linalg.solve(
            block.T @ block + ridge * eye, block.T @ target[low:high]
        )

    out = {}
    current = windows[origins].copy()
    predictions = np.zeros(len(origins))
    for step in range(max(horizons)):
        if step > 0:
            rows = np.clip(origins + step, 0, n_times - 1)
            current = np.roll(current, -1, axis=1)
            current[:, -1, :] = windows[rows][:, -1, :]
            current[:, -1, rv_column] = predictions
        predictions = np.einsum("pu,pu->p", weights, reservoir.evaluate(current))
        if step + 1 in horizons:
            out[step + 1] = predictions.copy()
    return out


def lstm_forecasts(
    windows: np.ndarray, target: np.ndarray, train_slices, origin_index,
    *, hidden_size: int, horizons=(1,), rv_column: int = 0, n_layers: int = 2,
    epochs: int = 100, batch_size: int = 64, lr: float = 1e-3, seed: int = 0,
    log_every: int = 25,
):
    """Rolling 2-layer LSTM refit at every forecast origin (paper Appendix D).

    The reference notebook re-initialises and retrains the network for each of
    the 245 rolling windows, which dominates the runtime of this reproduction.

    Returns
    -------
    dict of int to numpy.ndarray
    """
    import torch
    from torch import nn

    class Model(nn.Module):
        def __init__(self, n_inputs: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(n_inputs, hidden_size, n_layers, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    n_times, n_lags, n_features = windows.shape
    origins = np.asarray(origin_index)
    out = {h: np.empty(len(origins)) for h in horizons}
    torch.manual_seed(seed)

    for position, ((low, high), origin) in enumerate(zip(train_slices, origins)):
        rows = np.arange(max(low, n_lags), high)
        x_train = torch.tensor(windows[rows], dtype=torch.float32)
        y_train = torch.tensor(target[rows], dtype=torch.float32).unsqueeze(1)
        model = Model(n_features)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=batch_size, shuffle=True,
        )
        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimiser.zero_grad()
                loss_fn(model(batch_x), batch_y).backward()
                optimiser.step()

        model.eval()
        window = windows[origin].copy()
        with torch.no_grad():
            for step in range(max(horizons)):
                if step > 0:
                    row = min(origin + step, n_times - 1)
                    window = np.roll(window, -1, axis=0)
                    window[-1] = windows[row][-1]
                    window[-1, rv_column] = prediction
                prediction = float(
                    model(torch.tensor(window[None], dtype=torch.float32)).item()
                )
                if step + 1 in out:
                    out[step + 1][position] = prediction

        if (position + 1) % log_every == 0 or position + 1 == len(origins):
            logger.info(
                "TRAIN_EPOCH_COMPLETED | model=lstm(hidden=%d) | origin=%d/%d | "
                "last_window_loss=%.6f",
                hidden_size, position + 1, len(origins),
                float(loss_fn(model(x_train), y_train).item()),
            )
    return out
