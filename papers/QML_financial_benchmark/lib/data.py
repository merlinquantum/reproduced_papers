"""Price loading, cleaning, feature construction and splits for all three tasks.

The raw snapshot lives in ``<data_root>/QML_financial_benchmark/yahoo_daily_prices/csv``
and is produced by ``utils/download_yahoo_data.py``. ``data_root`` comes from the
shared runtime (``--data-root`` flag or the ``DATA_DIR`` environment variable).

Date conventions: every series is indexed by its exchange-local trading date.
Multi-asset feature sets inner-join prices on date before computing returns,
following §III-B ("we remove any day with missing observations").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from runtime_lib.data_paths import paper_data_dir

from . import features

logger = logging.getLogger(__name__)

PAPER_NAME = "QML_financial_benchmark"
SNAPSHOT_DIRNAME = "yahoo_daily_prices"

# Table V: APAC/EU indices close before the U.S. cash close and enter the 7-D
# set with same-day returns; the two U.S. proxies enter lagged by one day.
SAME_DAY_INDICES = ["^N225", "^HSI", "^AORD", "^GDAXI", "^FTSE"]
LAGGED_US_INDICES = ["^DJI", "^NYA"]
# The 64-D set uses "eight major indices" that the paper does not list
# (assumption R6): Table V's seven plus the S&P 500.
HIGH_DIM_INDICES = SAME_DAY_INDICES + LAGGED_US_INDICES + ["^GSPC"]
INDEX_SYMBOLS = frozenset(HIGH_DIM_INDICES)


def snapshot_csv_dir(data_root: str | Path | None) -> Path:
    """Directory holding one CSV per symbol of the downloaded snapshot.

    Raises
    ------
    FileNotFoundError
        If the snapshot has not been downloaded.
    """
    csv_dir = (
        paper_data_dir(PAPER_NAME, data_root, ensure_exists=False)
        / SNAPSHOT_DIRNAME
        / "csv"
    )
    if not csv_dir.is_dir():
        raise FileNotFoundError(
            f"Price snapshot not found at {csv_dir}. Run "
            "papers/QML_financial_benchmark/utils/download_yahoo_data.py first."
        )
    return csv_dir


def load_price_series(symbol: str, data_root: str | Path | None) -> pd.Series:
    """Load one symbol's daily price series, cleaned of non-trading rows.

    Individual equities use the split- and dividend-adjusted close; indices use
    the raw close (§III-B). Rows dropped as non-trading days:

    - rows with a missing price;
    - zero-volume rows whose close repeats the previous close (Yahoo placeholders
      on Borsa Istanbul holidays; user decision 2026-10-21). Indices are exempt
      because Yahoo reports zero volume for several of them on every day.

    Parameters
    ----------
    symbol : str
        Yahoo symbol, e.g. ``"KCHOL.IS"`` or ``"^GSPC"``.
    data_root : str | Path | None
        Shared data root resolved by the runtime.

    Returns
    -------
    pandas.Series
        Prices indexed by ``DatetimeIndex``, named after the symbol.
    """
    csv_path = snapshot_csv_dir(data_root) / f"{symbol}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"No snapshot CSV for symbol {symbol!r} at {csv_path}")
    frame = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    price_column = "close" if symbol in INDEX_SYMBOLS else "adj_close"
    frame = frame.dropna(subset=[price_column])
    if symbol not in INDEX_SYMBOLS:
        repeated_close = frame["close"].eq(frame["close"].shift(1))
        placeholder_rows = frame["volume"].eq(0) & repeated_close
        if placeholder_rows.any():
            logger.debug(
                "Dropping %d zero-volume placeholder rows for %s",
                int(placeholder_rows.sum()),
                symbol,
            )
        frame = frame.loc[~placeholder_rows]
    series = frame[price_column].astype(float)
    series.name = symbol
    return series


def load_aligned_prices(
    symbols: list[str], data_root: str | Path | None
) -> pd.DataFrame:
    """Inner-join several price series on their common trading dates."""
    return pd.concat(
        [load_price_series(symbol, data_root) for symbol in symbols],
        axis=1,
        join="inner",
    )


# ---------------------------------------------------------------------------
# Task 1: directional classification (§III-C)
# ---------------------------------------------------------------------------


def build_directional_frame(
    dataset_cfg: dict, data_root: str | Path | None
) -> pd.DataFrame:
    """Feature matrix and binary next-move label for one ticker and feature set.

    The label is ``y_t = 1[P_t > P_{t-1}]`` (Eq. 1). Feature sets:

    - ``"3d"``: RSI-14, %K-14 and its 3-day SMA on the stock (Fig. 3A).
      ``feature_timing="lagged"`` uses indicators from day ``t-1`` (no look-ahead;
      decision R7); ``"same_day"`` uses day ``t`` and is a leakage diagnostic.
    - ``"7d"``: same-day returns of N225, HSI, AORD, GDAXI, FTSE and the
      previous-day returns of DJI and NYA, predicting the S&P 500 (Table V).
    - ``"64d"``: lags 1-8 of the stock's own return and 7 lags of eight
      indices' returns (Fig. 3C). ``feature_timing="lagged"`` uses lags 1-7
      for every index; ``"table_v"`` uses lags 0-6 for the APAC/EU indices,
      whose closes precede the U.S. close (Table V; gap diagnosis G2).

    Returns
    -------
    pandas.DataFrame
        One row per date with feature columns and a ``label`` column; rows
        with incomplete features are dropped.
    """
    feature_set = dataset_cfg["feature_set"]
    ticker = dataset_cfg["ticker"]
    if feature_set == "3d":
        # One year of warm-up is ample for the 14-day indicators and keeps the
        # window clear of the non-positive pre-2004 Turkish adjusted closes.
        warm_up_start = pd.Timestamp(dataset_cfg["train_start"]) - pd.DateOffset(
            years=1
        )
        prices = load_price_series(ticker, data_root)
        prices = prices.loc[prices.index >= warm_up_start]
        if (prices <= 0).any():
            raise ValueError(
                f"Non-positive prices for {ticker} after {warm_up_start.date()}"
            )
        percent_k = features.stochastic_k(prices, 14)
        indicators = pd.DataFrame(
            {
                "rsi14": features.rsi(prices, 14),
                "pct_k14": percent_k,
                "pct_k14_sma3": percent_k.rolling(3).mean(),
            }
        )
        timing = dataset_cfg["feature_timing"]
        if timing == "lagged":
            indicators = indicators.shift(1)
        elif timing != "same_day":
            raise ValueError(
                f"feature_timing must be 'lagged' or 'same_day', got {timing!r}"
            )
        frame = indicators
        target_prices = prices
    elif feature_set == "7d":
        if ticker != "^GSPC":
            raise ValueError(
                "The 7-D feature set predicts the S&P 500; set ticker to '^GSPC'."
            )
        prices = load_aligned_prices(
            ["^GSPC", *SAME_DAY_INDICES, *LAGGED_US_INDICES], data_root
        )
        returns = np.log(prices / prices.shift(1))
        frame = pd.concat(
            [
                returns[SAME_DAY_INDICES],
                returns[LAGGED_US_INDICES].shift(1).add_suffix("_lag1"),
            ],
            axis=1,
        )
        target_prices = prices["^GSPC"]
    elif feature_set == "64d":
        symbols = [ticker, *[index for index in HIGH_DIM_INDICES if index != ticker]]
        prices = load_aligned_prices(symbols, data_root)
        returns = np.log(prices / prices.shift(1))
        columns = {
            f"{ticker}_lag{lag}": returns[ticker].shift(lag) for lag in range(1, 9)
        }
        timing = dataset_cfg["feature_timing"]
        if timing not in ("lagged", "table_v"):
            raise ValueError(
                f"64-D feature_timing must be 'lagged' or 'table_v', got {timing!r}"
            )
        for index in HIGH_DIM_INDICES:
            # "table_v": APAC/EU closes precede the U.S. close (Table V), so
            # their window is lags 0-6; U.S. indices stay at lags 1-7.
            first_lag = 0 if timing == "table_v" and index in SAME_DAY_INDICES else 1
            for lag in range(first_lag, first_lag + 7):
                columns[f"{index}_lag{lag}"] = returns[index].shift(lag)
        frame = pd.DataFrame(columns)
        if frame.shape[1] != 64:
            raise ValueError(f"64-D feature set has {frame.shape[1]} columns")
        target_prices = prices[ticker]
    else:
        raise ValueError(f"Unknown feature_set {feature_set!r}")

    frame = frame.copy()
    frame["label"] = (target_prices > target_prices.shift(1)).astype(float)
    frame = frame.iloc[1:].dropna()
    start, end = (
        pd.Timestamp(dataset_cfg["train_start"]),
        pd.Timestamp(dataset_cfg["test_end"]),
    )
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


@dataclass(frozen=True)
class Fold:
    """Index arrays of one walk-forward fold, all positions into the same frame."""

    number: int
    train: np.ndarray
    early_stop: np.ndarray
    validation: np.ndarray


def walk_forward_folds(dates: pd.DatetimeIndex, dataset_cfg: dict) -> list[Fold]:
    """Expanding-window folds of Fig. 4 with a chronological early-stop slice.

    Fold ``k`` trains on ``[train_start, boundary_k)`` and validates on
    ``[boundary_k, boundary_k + validation_years)``. The last
    ``early_stop_fraction`` of each training window is held out for early
    stopping, so no fold ever stops on its validation window (decision R4).
    """
    folds = []
    validation_years = int(dataset_cfg["validation_years"])
    early_stop_fraction = float(dataset_cfg["early_stop_fraction"])
    for number, boundary in enumerate(dataset_cfg["fold_boundaries"], start=1):
        start = pd.Timestamp(boundary)
        end = start + pd.DateOffset(years=validation_years)
        before = np.flatnonzero(dates < start)
        validation = np.flatnonzero((dates >= start) & (dates < end))
        n_early_stop = max(1, int(round(early_stop_fraction * len(before))))
        if len(validation) == 0 or len(before) <= n_early_stop:
            raise ValueError(
                f"Fold {number} starting {boundary} has no usable train or validation rows"
            )
        folds.append(
            Fold(number, before[:-n_early_stop], before[-n_early_stop:], validation)
        )
    return folds


class MinMaxScaler:
    """Per-feature min-max scaling to ``[0, 1]`` fitted on training rows only."""

    def fit(self, values: np.ndarray) -> MinMaxScaler:
        self.minimum = values.min(axis=0)
        span = values.max(axis=0) - self.minimum
        self.span = np.where(span > 0, span, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        # Test rows outside the training range are not clipped, matching
        # sklearn's MinMaxScaler default used for "fit on train, apply to test".
        return (values - self.minimum) / self.span


# ---------------------------------------------------------------------------
# Task 2: live trading (§III-D)
# ---------------------------------------------------------------------------


def build_trading_frame(
    dataset_cfg: dict, data_root: str | Path | None
) -> pd.DataFrame:
    """Daily S&P 500 indicators, next-day label and next-day return.

    Columns at date ``t``: ``log_return`` (``r_t``), ``macd``, ``macd_signal``,
    ``rsi14`` (§III-D2), ``label = 1[r_{t+1} > 0]`` and ``next_return``, the
    simple return from ``t`` to ``t+1`` earned by a position held over that day.
    """
    prices = load_price_series(dataset_cfg["ticker"], data_root)
    macd_line, signal_line = features.macd(prices)
    frame = pd.DataFrame(
        {
            "log_return": features.log_returns(prices),
            "macd": macd_line,
            "macd_signal": signal_line,
            "rsi14": features.rsi(prices, 14),
        }
    )
    frame["next_return"] = prices.shift(-1) / prices - 1.0
    frame["label"] = (frame["next_return"] > 0).astype(float)
    return frame.dropna()


TRADING_FEATURES = ["log_return", "macd", "macd_signal", "rsi14"]


def trading_windows(
    frame: pd.DataFrame, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Stack rolling windows ``x_t in R^{window x 4}`` ending at each date ``t``.

    Returns
    -------
    tuple
        ``(windows, labels, next_returns, dates)`` where ``windows`` has shape
        ``(n, window, 4)`` with rows ordered oldest to newest.
    """
    values = frame[TRADING_FEATURES].to_numpy(dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(
        values, (window, values.shape[1])
    )[:, 0]
    end_positions = np.arange(window - 1, len(frame))
    return (
        np.ascontiguousarray(windows),
        frame["label"].to_numpy()[end_positions],
        frame["next_return"].to_numpy()[end_positions],
        frame.index[end_positions],
    )


@dataclass(frozen=True)
class TradingSplit:
    """Chronological slices of one regime fold (Fig. 8 roles)."""

    regime: str
    train: np.ndarray
    early_stop: np.ndarray
    model_select: np.ndarray
    threshold_calibration: np.ndarray
    test: np.ndarray


def trading_split(dates: pd.DatetimeIndex, dataset_cfg: dict) -> TradingSplit:
    """Split windows of one regime into train/early-stop/model-select/tau/test.

    ``split_mode="history_then_regime"`` (Fig. 6 reading): the history from
    ``history_start`` to the day before the regime is cut 70/10/10/10 and the
    whole regime is the out-of-sample backtest. ``"within_regime"`` (text
    reading, R10): the regime itself is cut 70/10/10/10 and the backtest is
    reported on the threshold-calibration slice, which is therefore not
    held out from threshold selection.
    """
    regime = dataset_cfg["regime"]
    regime_start, regime_end = (
        pd.Timestamp(day) for day in dataset_cfg["regimes"][regime]
    )
    in_regime = np.flatnonzero((dates >= regime_start) & (dates <= regime_end))
    mode = dataset_cfg["split_mode"]
    if mode == "history_then_regime":
        history = np.flatnonzero(
            (dates >= pd.Timestamp(dataset_cfg["history_start"]))
            & (dates < regime_start)
        )
        to_split, test = history, in_regime
    elif mode == "within_regime":
        to_split, test = in_regime, None
    else:
        raise ValueError(
            f"split_mode must be 'history_then_regime' or 'within_regime', got {mode!r}"
        )

    fractions = np.cumsum(dataset_cfg["split_fractions"])
    if not np.isclose(fractions[-1], 1.0):
        raise ValueError("split_fractions must sum to 1")
    cuts = np.round(fractions[:-1] * len(to_split)).astype(int)
    train, early_stop, model_select, threshold_calibration = np.split(to_split, cuts)
    if test is None:
        test = threshold_calibration
    for name, part in {
        "train": train,
        "early_stop": early_stop,
        "model_select": model_select,
        "threshold_calibration": threshold_calibration,
        "test": test,
    }.items():
        if len(part) == 0:
            raise ValueError(f"Trading split {regime}/{mode} has an empty {name} slice")
    return TradingSplit(
        regime, train, early_stop, model_select, threshold_calibration, test
    )


# ---------------------------------------------------------------------------
# Task 3: volatility forecasting (§III-E)
# ---------------------------------------------------------------------------


def build_volatility_frame(
    dataset_cfg: dict, data_root: str | Path | None
) -> pd.DataFrame:
    """Lagged returns and realized variances with the next-day RV target (Eq. 6-8).

    Feature row at ``t``: ``[r_t, ..., r_{t-p+1}, RV_t, ..., RV_{t-q+1}]``;
    target ``RV_{t+1}``. Extra columns ``rv_now`` (``RV_t``), ``rv_month``
    (22-day mean squared return, for HAR-RV), ``return_now`` and
    ``squared_return_sum4`` are kept for the baselines and are not model inputs.
    """
    prices = load_price_series(dataset_cfg["ticker"], data_root)
    start = pd.Timestamp(dataset_cfg["start"])
    prices = prices.loc[prices.index >= start - pd.DateOffset(days=60)]
    returns = features.log_returns(prices)
    variance = features.realized_variance(returns, 5)
    p_lags, q_lags = int(dataset_cfg["return_lags"]), int(dataset_cfg["rv_lags"])
    columns = {f"r_lag{lag}": returns.shift(lag) for lag in range(p_lags)}
    columns |= {f"rv_lag{lag}": variance.shift(lag) for lag in range(q_lags)}
    frame = pd.DataFrame(columns)
    frame["rv_now"] = variance
    frame["rv_month"] = (returns**2).rolling(22).mean()
    frame["return_now"] = returns
    # Sum of the four most recent squared returns: the part of RV_{t+1} already
    # known at t, which the GARCH baseline completes with its sigma^2_{t+1}.
    frame["squared_return_sum4"] = (returns**2).rolling(4).sum()
    frame["target"] = variance.shift(-1)
    frame = frame.dropna()
    end = pd.Timestamp(dataset_cfg["end"])
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def volatility_feature_columns(dataset_cfg: dict) -> list[str]:
    """Model input columns ``[r lags..., RV lags...]`` in Eq. 8 order."""
    return [f"r_lag{lag}" for lag in range(int(dataset_cfg["return_lags"]))] + [
        f"rv_lag{lag}" for lag in range(int(dataset_cfg["rv_lags"]))
    ]


# ---------------------------------------------------------------------------
# Shared DataLoader entry point
# ---------------------------------------------------------------------------


def build_dataloader(cfg: dict, split: str) -> DataLoader:
    """DataLoader of ``(features, label)`` tensors for one task split.

    Used by the one-batch check and tests; the task modules index the frames
    directly because they train many candidates on the same arrays.

    Parameters
    ----------
    cfg : dict
        Resolved run config with ``task``, the task section and ``data_root``.
    split : str
        Directional: ``train``, ``early_stop`` or ``validation`` of fold
        ``dataset.loader_fold`` (1-based). Trading: ``train``, ``early_stop``,
        ``model_select``, ``threshold_calibration`` or ``test``. Volatility:
        ``train`` (first ``initial_train_days`` rows) or ``test``.

    Returns
    -------
    torch.utils.data.DataLoader
        Unshuffled loader; directional and trading features are min-max scaled
        on the training rows.
    """
    task = cfg["task"]
    dataset_cfg = cfg[task]["dataset"]
    data_root = cfg["data_root"]
    if task == "directional":
        frame = build_directional_frame(dataset_cfg, data_root)
        fold = walk_forward_folds(frame.index, dataset_cfg)[
            int(dataset_cfg["loader_fold"]) - 1
        ]
        values = frame.drop(columns="label").to_numpy(dtype=np.float64)
        scaler = MinMaxScaler().fit(values[fold.train])
        rows = {
            "train": fold.train,
            "early_stop": fold.early_stop,
            "validation": fold.validation,
        }[split]
        inputs, labels = scaler.transform(values[rows]), frame["label"].to_numpy()[rows]
    elif task == "trading":
        frame = build_trading_frame(dataset_cfg, data_root)
        windows, labels_all, _, dates = trading_windows(
            frame, int(dataset_cfg["window"])
        )
        parts = trading_split(dates, dataset_cfg)
        scaler = MinMaxScaler().fit(windows[parts.train].reshape(-1, windows.shape[-1]))
        rows = getattr(parts, split)
        inputs, labels = scaler.transform(windows[rows]), labels_all[rows]
    elif task == "volatility":
        frame = build_volatility_frame(dataset_cfg, data_root)
        n_initial = int(dataset_cfg["initial_train_days"])
        rows = slice(0, n_initial) if split == "train" else slice(n_initial, None)
        inputs = frame[volatility_feature_columns(dataset_cfg)].to_numpy(
            dtype=np.float64
        )[rows]
        labels = frame["target"].to_numpy()[rows]
    else:
        raise ValueError(f"Unknown task {task!r}")
    tensors = TensorDataset(
        torch.tensor(np.array(inputs), dtype=torch.float32),
        torch.tensor(np.array(labels), dtype=torch.float32),
    )
    return DataLoader(tensors, batch_size=int(dataset_cfg["batch_size"]), shuffle=False)
