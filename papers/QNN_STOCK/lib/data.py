"""Data pipeline for the QNN_STOCK reproduction (paper Sec. II.A and VI.A).

Daily prices -> optional moving average (stride 1) -> one-step differences ->
symbols in ``{0, ..., levels - 1}`` -> sliding windows of ``T + 1`` symbols
(``T`` context steps followed by the next step).

Index convention used everywhere in this package: a context
``(s_1, ..., s_T)`` has index ``c = sum_t s_t * levels**(T - t)`` (the oldest
step is the most significant digit) and a joint state ``(context, next)`` has
index ``c * levels + next``. With ``levels = 2**q`` this is the big-endian
computational-basis index of the register ``[step 1 | ... | step T | next]``,
each step using ``q`` qubits, the oldest step on qubit 0.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from runtime_lib.data_paths import resolve_data_root

logger = logging.getLogger(__name__)

PAPER_DATA_DIRNAME = "QNN_STOCK"
_PROJECT_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PROJECT_DIR.parents[1]
# The Zenodo snapshot's final row (2024-06-26) is an intraday snapshot for at
# least AMZN and GOOGL (see data/QNN_STOCK/sp500_daily_prices_zenodo/PROVENANCE.md).
PARTIAL_LAST_DATE = "2024-06-26"


@dataclass(frozen=True)
class TaskData:
    """Symbol windows and provenance for one ticker.

    Attributes
    ----------
    ticker : str
        Ticker symbol.
    levels : int
        Number of quantization levels ``d``.
    context_length : int
        Number of context steps ``T``.
    horizon : int
        Number of predicted steps ``tau``.
    outcome_levels : int
        Number of prediction outcomes, ``levels ** horizon``.
    train_windows, test_windows : numpy.ndarray
        Integer arrays of shape ``(n, T + 1)``, chronological split. The first
        ``T`` columns are the context steps; the last column is the outcome
        index in ``range(outcome_levels)``. For ``horizon > 1`` it encodes the
        next ``horizon`` steps with the earliest step as the most significant
        digit.
    first_date, last_date : str
        Date range of the prices used.
    n_prices, n_moves : int
        Number of prices read and of symbolized moves.
    n_zero_moves : int
        Moves exactly equal to zero (mapped to level 0 under ``sign``).
    bin_edges : list of float or None
        Quantile edges fitted on the training moves, for ``quantile``.
    """

    ticker: str
    levels: int
    context_length: int
    horizon: int
    outcome_levels: int
    train_windows: np.ndarray
    test_windows: np.ndarray
    first_date: str
    last_date: str
    n_prices: int
    n_moves: int
    n_zero_moves: int
    bin_edges: list[float] | None

    @property
    def qubits_per_step(self) -> int:
        return int(math.log2(self.levels))

    def provenance(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "horizon": self.horizon,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "n_prices": self.n_prices,
            "n_moves": self.n_moves,
            "n_zero_moves": self.n_zero_moves,
            "n_train_windows": int(len(self.train_windows)),
            "n_test_windows": int(len(self.test_windows)),
            "bin_edges": self.bin_edges,
        }


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key '{where}.{key}'")
    return mapping[key]


def dataset_dir(cfg: dict[str, Any]) -> Path:
    """Return the directory holding the per-ticker CSV files."""

    dataset_cfg = _require(cfg, "dataset", "<root>")
    source_dir = _require(dataset_cfg, "source_dir", "dataset")
    root = resolve_data_root(cfg.get("data_root"), _PROJECT_DIR)
    if root.resolve() == (_PROJECT_DIR / "data").resolve():
        # runtime_lib.find_repo_root() stops at the first folder holding an
        # implementation.py, which is this paper folder, so the default data
        # root becomes papers/QNN_STOCK/data. The shared root is <repo>/data.
        root = _REPO_ROOT / "data"
    path = root / PAPER_DATA_DIRNAME / source_dir
    if not path.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {path}. Run find-and-download-data for "
            "papers/QNN_STOCK or pass --data-root."
        )
    return path


def load_prices(
    path: Path, ticker: str, price_field: str, end_date: str | None
) -> pd.Series:
    """Load one ticker's daily prices, dropping the partial snapshot row.

    Parameters
    ----------
    path : pathlib.Path
        Directory with ``<TICKER>_stock_data.csv`` files.
    ticker : str
        Ticker symbol.
    price_field : str
        Column to use, ``Close`` or ``Adj Close``.
    end_date : str or None
        Last date to keep (inclusive). ``None`` keeps all complete days.

    Returns
    -------
    pandas.Series
        Prices indexed by date.
    """

    csv_path = path / f"{ticker}_stock_data.csv"
    frame = pd.read_csv(csv_path, parse_dates=["Date"])
    if price_field not in frame.columns:
        raise KeyError(f"{csv_path} has no column '{price_field}'")
    frame = frame[frame["Date"] < pd.Timestamp(PARTIAL_LAST_DATE)]
    if end_date is not None:
        frame = frame[frame["Date"] <= pd.Timestamp(end_date)]
    if not frame["Date"].is_monotonic_increasing:
        raise ValueError(f"{csv_path}: dates are not sorted")
    return frame.set_index("Date")[price_field].astype(float)


def price_moves(prices: pd.Series, moving_average_window: int) -> np.ndarray:
    """Smooth prices with a stride-1 moving average, then take differences.

    The moving average and the difference are both linear, so this equals the
    paper's order (difference, then average). Averaging the prices first gives
    ``n - w + 1`` smoothed values, which matches the paper's sample count
    (10,033 Apple prices -> 10,029 samples for ``w = 5``).
    """

    if moving_average_window < 1:
        raise ValueError("moving_average_window must be >= 1")
    smoothed = prices.rolling(moving_average_window).mean().dropna().to_numpy()
    return np.diff(smoothed)


def symbolize(
    moves: np.ndarray, levels: int, quantization: str, train_moves: np.ndarray
) -> tuple[np.ndarray, list[float] | None]:
    """Map moves to integer levels.

    ``sign`` (d = 2): up (> 0) -> 1, otherwise -> 0. Zero moves go to 0, the
    lowest level, as in Eq. 2. ``quantile``: equal-frequency bins whose edges
    are fitted on the training moves only (Sec. VI.C.3, density-based bins).
    """

    if quantization == "sign":
        if levels != 2:
            raise ValueError("quantization 'sign' requires levels == 2")
        return (moves > 0).astype(np.int64), None
    if quantization == "quantile":
        quantiles = np.arange(1, levels) / levels
        edges = np.quantile(train_moves, quantiles)
        return np.searchsorted(edges, moves, side="right").astype(
            np.int64
        ), edges.tolist()
    raise ValueError(f"Unknown quantization '{quantization}'")


def training_subset(train_windows: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Return a contiguous block of ``dataset.train_size`` training windows.

    ``dataset.train_size`` null keeps the full chronological training split.
    Otherwise the block start is drawn uniformly from the valid offsets with
    ``numpy.random.default_rng(dataset.train_subset_seed)``; the string
    ``"seed"`` uses the run's top-level ``seed``, so paired runs with the same
    seed see the same block. The test split is never subsampled, and the
    quantile bin edges stay fitted on the full training moves.

    Raises
    ------
    ValueError
        If the size is not positive or exceeds the training split.
    """

    dataset_cfg = cfg["dataset"]
    size = _require(dataset_cfg, "train_size", "dataset")
    if size is None:
        return train_windows
    size = int(size)
    if not 0 < size <= len(train_windows):
        raise ValueError(
            f"dataset.train_size {size} outside 1..{len(train_windows)} training windows"
        )
    subset_seed = _require(dataset_cfg, "train_subset_seed", "dataset")
    if subset_seed == "seed":
        subset_seed = cfg["seed"]
    start = int(
        np.random.default_rng(int(subset_seed)).integers(
            0, len(train_windows) - size + 1
        )
    )
    return train_windows[start : start + size]


def build_task(cfg: dict[str, Any], ticker: str) -> TaskData:
    """Build the symbol windows for one ticker from the resolved config."""

    dataset_cfg = cfg["dataset"]
    levels = int(_require(dataset_cfg, "levels", "dataset"))
    if levels < 2 or levels & (levels - 1):
        raise ValueError("dataset.levels must be a power of two >= 2")
    context_length = int(_require(dataset_cfg, "context_length", "dataset"))
    train_fraction = float(_require(dataset_cfg, "train_fraction", "dataset"))
    horizon = int(_require(dataset_cfg, "horizon", "dataset"))
    if horizon < 1:
        raise ValueError("dataset.horizon must be at least 1")

    prices = load_prices(
        dataset_dir(cfg),
        ticker,
        _require(dataset_cfg, "price_field", "dataset"),
        _require(dataset_cfg, "end_date", "dataset"),
    )
    moves = price_moves(
        prices, int(_require(dataset_cfg, "moving_average_window", "dataset"))
    )

    n_windows = len(moves) - context_length - horizon + 1
    n_train = int(math.floor(train_fraction * n_windows))
    # Training windows use moves[0 : n_train + T + horizon - 1]; quantile edges
    # are fitted there.
    symbols, edges = symbolize(
        moves,
        levels,
        _require(dataset_cfg, "quantization", "dataset"),
        moves[: n_train + context_length + horizon - 1],
    )
    windows = outcome_windows(symbols, context_length, horizon, levels)
    train_windows = training_subset(windows[:n_train], cfg)
    return TaskData(
        ticker=ticker,
        levels=levels,
        context_length=context_length,
        horizon=horizon,
        outcome_levels=levels**horizon,
        train_windows=np.ascontiguousarray(train_windows),
        test_windows=np.ascontiguousarray(windows[n_train:]),
        first_date=str(prices.index[0].date()),
        last_date=str(prices.index[-1].date()),
        n_prices=int(len(prices)),
        n_moves=int(len(moves)),
        n_zero_moves=int(np.sum(moves == 0)),
        bin_edges=edges,
    )


def build_tasks(cfg: dict[str, Any]) -> list[TaskData]:
    """Build one :class:`TaskData` per ticker in ``dataset.tickers``."""

    tickers = _require(cfg["dataset"], "tickers", "dataset")
    if not tickers:
        raise ValueError("dataset.tickers must list at least one ticker")
    tasks = [build_task(cfg, ticker) for ticker in tickers]
    for task in tasks:
        logger.info(
            "Data %s: %s..%s, %d prices, %d moves (%d zero), %d train / %d test windows",
            task.ticker,
            task.first_date,
            task.last_date,
            task.n_prices,
            task.n_moves,
            task.n_zero_moves,
            len(task.train_windows),
            len(task.test_windows),
        )
    return tasks


def context_indices(windows: np.ndarray, levels: int) -> np.ndarray:
    """Context index of every window (oldest step most significant)."""

    context = windows[:, :-1]
    weights = levels ** np.arange(context.shape[1] - 1, -1, -1)
    return context @ weights


def outcome_windows(
    symbols: np.ndarray, context_length: int, horizon: int, levels: int
) -> np.ndarray:
    """Sliding windows ``(context steps..., outcome index)`` of shape ``(n, T + 1)``.

    The outcome index encodes the ``horizon`` steps after the context in base
    ``levels``, earliest step most significant; for ``horizon == 1`` it is the
    next symbol itself.
    """

    raw = np.lib.stride_tricks.sliding_window_view(symbols, context_length + horizon)
    weights = levels ** np.arange(horizon - 1, -1, -1)
    outcome = raw[:, context_length:] @ weights
    return np.column_stack([raw[:, :context_length], outcome])


def joint_distribution(
    windows: np.ndarray, levels: int, outcome_levels: int
) -> np.ndarray:
    """Empirical ``P(context, outcome)``, shape ``(levels**T, outcome_levels)``."""

    n_contexts = levels ** (windows.shape[1] - 1)
    counts = np.zeros((n_contexts, outcome_levels))
    np.add.at(counts, (context_indices(windows, levels), windows[:, -1]), 1.0)
    return counts / counts.sum()


def split_validation(
    windows: np.ndarray, fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split of training windows into (fit, validation)."""

    n_validation = int(math.floor(fraction * len(windows)))
    if n_validation < 1:
        raise ValueError("validation fraction leaves no validation windows")
    return windows[:-n_validation], windows[-n_validation:]


def build_dataloader(
    cfg: dict[str, Any], split: str
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return one full-split batch per ticker.

    QBGU (Sec. IV.A) trains on the whole empirical distribution at once, so a
    split is a single batch: ``(windows, joint)`` with ``windows`` of shape
    ``(n, T + 1)`` and ``joint`` of shape ``(levels**T, levels)``.
    """

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    batches = []
    for task in build_tasks(cfg):
        windows = task.train_windows if split == "train" else task.test_windows
        batches.append(
            (
                torch.as_tensor(windows, dtype=torch.long),
                torch.as_tensor(
                    joint_distribution(windows, task.levels, task.outcome_levels)
                ),
            )
        )
    return batches
