"""Dataset reconstruction for the QRC realized-volatility reproduction.

The authors publish only the *normalised* feature table (``Data.CSV``); the raw
``Data_raw.csv`` / ``dff.csv`` files referenced by their notebooks are absent
from the repository. Every column of ``Data.CSV`` is an independent min-max
rescaling of the corresponding raw series, and the reference Julia code
(``Time_series.jl``) hard-codes the two constants needed to invert the target
transform:

.. code-block:: julia

    Max_RV = -1.2543188032019446
    Min_RV = -4.7722718186046515

so ``log RV_t = (RV_norm_t + 1) * (Max_RV - Min_RV) + Min_RV`` (the target was
mapped to ``[-1, 0]``). This module inverts that map to recover the raw
log-realized-volatility series exactly, and rebuilds the authors' ``dff``
regressor frame by replaying their ADF-based differencing rule. Because the
augmented Dickey-Fuller statistic is invariant under affine rescaling, the
differencing decision recovered here is identical to theirs, and because OLS
predictions are invariant under affine transforms of the regressors, the linear
econometric baselines are unaffected by the missing raw feature scales.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

# Inverse of the authors' target normalisation (Time_series.jl lines 32-36).
MAX_LOG_RV = -1.2543188032019446
MIN_LOG_RV = -4.7722718186046515
LOG_RV_RANGE = MAX_LOG_RV - MIN_LOG_RV

RAW_FEATURE_COLUMNS = (
    "DP", "EP", "MKT", "SMB", "HML", "TB", "DEF", "IP", "INF", "STR",
)
# Columns available to the quantum reservoir (Data.CSV, normalised scale).
QRC_FEATURE_POOL = (
    "RV", "RV_q", "RV_a", "DP", "EP", "MKT", "SMB", "HML", "STR", "TB",
    "INF", "DEF", "IP",
)


def denormalise_log_rv(values: np.ndarray) -> np.ndarray:
    """Map normalised realized volatility back to raw ``log RV`` units."""
    return (np.asarray(values) + 1.0) * LOG_RV_RANGE + MIN_LOG_RV


def load_normalised_table(data_root: str | Path) -> pd.DataFrame:
    """Load the authors' normalised monthly feature table.

    Parameters
    ----------
    data_root : str or pathlib.Path
        Directory holding ``Data.CSV``.

    Returns
    -------
    pandas.DataFrame
        816 monthly rows (1950-01 .. 2017-12) indexed by month-end date.

    Raises
    ------
    FileNotFoundError
        If ``Data.CSV`` is missing from ``data_root``.
    """
    path = Path(data_root) / "Data.CSV"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy Data.CSV from the authors' repository "
            "(see README, Data section)."
        )
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info(
        "DATASET_SOURCE_LOADED | source=%s | rows=%d | start=%s | end=%s",
        path, len(frame), frame.index[0].date(), frame.index[-1].date(),
    )
    return frame


def load_coupling_instances(data_root: str | Path) -> np.ndarray:
    """Load the authors' 100 saved transverse-field Ising coupling matrices.

    Returns
    -------
    numpy.ndarray, shape (100, 10, 10)
        Symmetric, zero-diagonal coupling matrices, each normalised so its
        largest eigenvalue equals 1 (``coeff_matrix`` in Time_series.jl).
        Instance 0 is the QR1 reservoir and instance 1 the QR2 reservoir used
        in the paper.

    Raises
    ------
    FileNotFoundError
        If ``coeff_10.jld2`` is missing from ``data_root``.
    """
    import h5py

    path = Path(data_root) / "coeff_10.jld2"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (authors' coeff_10.jld2).")
    with h5py.File(path, "r") as handle:
        # JLD2 stores the Julia (100, 10, 10) array with reversed axes.
        stored = np.array(handle["ms"])
    return np.transpose(stored, (2, 1, 0))


def sample_coupling_instances(n_instances: int, n_qubits: int, seed: int) -> np.ndarray:
    """Draw fresh coupling matrices following ``coeff_matrix`` in the reference code.

    Parameters
    ----------
    n_instances : int
        Number of reservoir instances to draw.
    n_qubits : int
        Reservoir size.
    seed : int
        Seed for the NumPy generator.

    Returns
    -------
    numpy.ndarray, shape (n_instances, n_qubits, n_qubits)
        Symmetric zero-diagonal matrices with unit largest eigenvalue.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_instances, n_qubits, n_qubits))
    for i in range(n_instances):
        m = rng.random((n_qubits, n_qubits))
        m = 0.5 * (m + m.T)
        np.fill_diagonal(m, 0.0)
        out[i] = m / np.linalg.eigvalsh(m).max()
    return out


def build_regressor_frame(normalised: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the authors' ``dff`` frame: raw log RV plus ADF-differenced features.

    Parameters
    ----------
    normalised : pandas.DataFrame
        Output of :func:`load_normalised_table`.

    Returns
    -------
    pandas.DataFrame
        ``RV`` in raw ``log RV`` units, exogenous columns kept in levels or
        first-differenced according to an ADF test at the 5 % level, plus the
        HAR lag terms used by the econometric baselines.
    """
    frame = pd.DataFrame(index=normalised.index)
    frame["RV"] = denormalise_log_rv(normalised["RV"].values)
    for column in RAW_FEATURE_COLUMNS:
        series = normalised[column]
        p_value = sm.tsa.adfuller(series.dropna())[1]
        if p_value > 0.05:
            frame[f"diff_{column}"] = series.diff()
            logger.debug("ADF | %s p=%.4f -> first-differenced", column, p_value)
        else:
            frame[column] = series
            logger.debug("ADF | %s p=%.4f -> level", column, p_value)

    frame["RV_lag1"] = frame["RV"].shift(1)
    frame["RV_quarterly_lag"] = frame["RV"].rolling(3).mean().shift(1)
    frame["RV_annual_lag"] = frame["RV"].rolling(12).mean().shift(1)
    return frame.fillna(0.0)


def build_lagged_inputs(
    normalised: pd.DataFrame, features: list[str], n_lags: int
) -> np.ndarray:
    """Assemble the reservoir input tensor.

    Parameters
    ----------
    normalised : pandas.DataFrame
        Normalised feature table.
    features : list of str
        Feature columns fed to the input qubits, in encoding order.
    n_lags : int
        Memory depth ``k`` (3 in the paper).

    Returns
    -------
    numpy.ndarray, shape (T, n_lags, len(features))
        ``out[t, j]`` holds the features at time ``t - n_lags + j``, so
        ``out[t, 0]`` is the oldest lag. The first ``n_lags`` rows are zero,
        matching the reference implementation which leaves them unevaluated.
    """
    values = normalised[list(features)].to_numpy(dtype=np.float64)
    total = len(values)
    out = np.zeros((total, n_lags, len(features)))
    for t in range(n_lags, total):
        out[t] = values[t - n_lags:t]
    return out


def rolling_windows(n_total: int, n_out_of_sample: int, window_start: int = 0):
    """Rolling one-step-ahead re-estimation schedule of the reference code.

    Parameters
    ----------
    n_total : int
        Number of observations (816).
    n_out_of_sample : int
        Number of forecasts (245).
    window_start : int
        Index of the first training row. Default value is 0.

    Returns
    -------
    tuple
        ``(train_slices, predict_index)`` where ``train_slices[j]`` is the
        half-open ``(lo, hi)`` training range for forecast ``j`` and
        ``predict_index[j]`` the row being forecast.
    """
    window_length = n_total - n_out_of_sample - window_start
    train_slices = [
        (window_start + j, window_start + j + window_length)
        for j in range(n_out_of_sample)
    ]
    predict_index = [window_start + window_length + j for j in range(n_out_of_sample)]
    return train_slices, predict_index
