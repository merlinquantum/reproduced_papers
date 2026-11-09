"""Technical indicators and volatility targets used by arXiv:2601.03802.

All functions take a price or return series indexed by date and return a
series on the same index. Values at date ``t`` use prices up to and including
``t``; callers shift them when a feature must only use information up to
``t-1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log return ``r_t = ln(P_t / P_{t-1})``."""
    if (prices <= 0).any():
        bad_dates = prices.index[prices <= 0]
        raise ValueError(
            f"Non-positive prices in {prices.name!r} on {len(bad_dates)} dates, "
            f"first {bad_dates[0].date()}; the requested window must start after them."
        )
    return np.log(prices / prices.shift(1))


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Wilder relative strength index on a 0-100 scale (RSI-14 in §III-C1)."""
    change = prices.diff()
    gains = change.clip(lower=0.0)
    losses = -change.clip(upper=0.0)
    # Wilder smoothing is an exponential average with alpha = 1/window.
    average_gain = gains.ewm(
        alpha=1.0 / window, adjust=False, min_periods=window
    ).mean()
    average_loss = losses.ewm(
        alpha=1.0 / window, adjust=False, min_periods=window
    ).mean()
    relative_strength = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def stochastic_k(prices: pd.Series, window: int = 14) -> pd.Series:
    """Raw stochastic oscillator %K over ``window`` days, on a 0-100 scale.

    The paper builds %K-14 from adjusted-close prices only (Fig. 3A), so the
    14-day high and low are taken over closes rather than intraday highs/lows.
    """
    rolling_low = prices.rolling(window).min()
    rolling_high = prices.rolling(window).max()
    price_range = (rolling_high - rolling_low).replace(0.0, np.nan)
    return 100.0 * (prices - rolling_low) / price_range


def macd(prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    """MACD line (EMA12 - EMA26) and its 9-day EMA signal line (§III-D2)."""
    ema_fast = prices.ewm(span=12, adjust=False).mean()
    ema_slow = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def realized_variance(returns: pd.Series, window: int = 5) -> pd.Series:
    """Rolling realized variance ``RV_t = (1/5) * sum_{i=0..4} r_{t-i}^2`` (Eq. 6)."""
    return (returns**2).rolling(window).mean()
