"""IPTO Greek hourly load data: loading, windowing, splitting and normalization.

The paper (arXiv:2605.04991, Sec. III-A) uses 35,064 consecutive hourly loads
from the IPTO (ADMIE) portal, a 4-hour sliding window to predict the next hour,
a 24,539 / 5,258 / 5,267 train / evaluation / test split and min-max
normalization. The window 2020-10-22 .. 2024-10-21 (first 24 hourly columns of
each daily file) was identified from the Table II anchors and the reported length (README, *Data*).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PAPER_NAME = "distributed_qrc"
DATASET_SLUG = "ipto_greek_hourly_load"
RAW_SUBDIR = Path("raw") / "RealTimeSCADASystemLoad"
HOURS_PER_DAY = 24
SPLIT_NAMES = ("train", "eval", "test")

# Fingerprint of the paper window read with the first-24-columns rule and the
# artifact values kept as published (README, *Data*).
PAPER_WINDOW = ("2020-10-22", "2024-10-21")
PAPER_WINDOW_LENGTH = 35_064
PAPER_WINDOW_SUM = 165_669_294
PAPER_WINDOW_SHA256 = "eca5bc5197fd4ad0b004b5c8441f092d297213bdecdbf549fd248ba133a0fb99"


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key '{where}.{key}'")
    return mapping[key]


PROJECT_DIR = Path(__file__).resolve().parents[1]


def dataset_dir(cfg: dict[str, Any]) -> Path:
    """Return ``<data_root>/distributed_qrc/ipto_greek_hourly_load``.

    ``cfg["data_root"]`` (``--data-root``) wins, then the ``DATA_DIR`` environment
    variable. A relative root is resolved against this paper folder, so the
    default ``"../../data"`` is the repository-level data root both under the
    shared runtime and in tests. (The shared ``find_repo_root`` would stop at
    this paper folder, which also contains an ``implementation.py``.)
    """

    from runtime_lib.data_paths import resolve_data_root

    root = resolve_data_root(cfg.get("data_root"), PROJECT_DIR)
    return root / PAPER_NAME / DATASET_SLUG


def series_fingerprint(values: np.ndarray) -> str:
    """SHA-256 of the integer values joined by commas (the paper-window fingerprint)."""

    text = ",".join(str(int(v)) for v in values)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _daily_file(raw_dir: Path, day: dt.date) -> Path:
    # Re-published days carry a higher revision suffix (_02); keep the latest.
    candidates = sorted(raw_dir.glob(f"{day:%Y%m%d}_RealTimeSCADASystemLoad_*.xls"))
    if not candidates:
        raise FileNotFoundError(f"No IPTO file for {day.isoformat()} in {raw_dir}")
    return candidates[-1]


def _read_day(path: Path, day: dt.date) -> np.ndarray:
    import xlrd

    sheet = xlrd.open_workbook(path).sheet_by_index(0)
    row = sheet.row_values(2)
    if row[0] != f"{day:%d-%m-%Y}":
        raise ValueError(f"{path.name}: date cell {row[0]!r} does not match {day}")
    # Columns 1..24 are the hourly loads; column 25 is only filled on fall-back
    # DST days and is dropped, which reproduces the paper's 35,064 values.
    return np.asarray(row[1 : 1 + HOURS_PER_DAY], dtype=np.float64)


def load_hourly_series(directory: Path, start: str, end: str) -> np.ndarray:
    """Read the hourly load (MW) between two inclusive dates.

    The parsed series is cached as ``processed/hourly_load_<start>_<end>.npy``
    next to the raw files.
    """

    cache = directory / "processed" / f"hourly_load_{start}_{end}.npy"
    if cache.exists():
        return np.load(cache)
    raw_dir = directory / RAW_SUBDIR
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw IPTO files not found at {raw_dir}. See the README *Data* section "
            "for the source and download."
        )
    first, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    days = [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]
    series = np.concatenate([_read_day(_daily_file(raw_dir, d), d) for d in days])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, series)
    logger.info("Cached parsed series (%d values) at %s", series.size, cache)
    return series


def interpolate_artifacts(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace non-positive loads (DST zeros, negative net loads) by linear interpolation.

    Returns the cleaned series and the indices that were replaced.
    """

    bad = np.flatnonzero(series <= 0)
    cleaned = series.copy()
    good = np.setdiff1d(np.arange(series.size), bad)
    cleaned[bad] = np.interp(bad, good, series[good])
    return cleaned, bad


@dataclass
class SplitScaler:
    """Min-max scaler of one split: ``x_norm = (x - minimum) / (maximum - minimum)``."""

    minimum: float
    maximum: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.minimum) / (self.maximum - self.minimum)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * (self.maximum - self.minimum) + self.minimum


def make_windows(values: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows ``[t-lags+1 .. t] -> t+1`` with oldest lag first (Table II order)."""

    n_windows = values.size - lags
    if n_windows <= 0:
        raise ValueError(f"Series of length {values.size} is too short for {lags} lags")
    index = np.arange(lags)[None, :] + np.arange(n_windows)[:, None]
    return values[index], values[lags:]


@dataclass
class ForecastData:
    """Windowed, normalized splits plus the metadata needed for evidence files."""

    inputs: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    scalers: dict[str, SplitScaler]
    metadata: dict[str, Any] = field(default_factory=dict)

    def denormalize(self, split: str, values: np.ndarray) -> np.ndarray:
        return self.scalers[split].inverse(values)


def build_forecast_data(cfg: dict[str, Any]) -> ForecastData:
    """Build the paper's windowed train / eval / test data from ``cfg["dataset"]``.

    Required keys: ``name``, ``start``, ``end``, ``split_sizes``, ``lags``,
    ``normalization`` (``"train"`` or ``"per_split"``) and ``artifact_handling``
    (``"keep"`` or ``"interpolate"``).
    """

    data_cfg = _require(cfg, "dataset", "<root>")
    name = _require(data_cfg, "name", "dataset")
    if name != DATASET_SLUG:
        raise ValueError(f"Unsupported dataset '{name}'; only '{DATASET_SLUG}' exists")
    start = _require(data_cfg, "start", "dataset")
    end = _require(data_cfg, "end", "dataset")
    split_sizes = [int(v) for v in _require(data_cfg, "split_sizes", "dataset")]
    lags = int(_require(data_cfg, "lags", "dataset"))
    normalization = _require(data_cfg, "normalization", "dataset")
    artifact_handling = _require(data_cfg, "artifact_handling", "dataset")

    directory = dataset_dir(cfg)
    series = load_hourly_series(directory, start, end)
    if (start, end) == PAPER_WINDOW:
        if series.size != PAPER_WINDOW_LENGTH or int(series.sum()) != PAPER_WINDOW_SUM:
            raise ValueError(
                "Paper window does not match the expected length/sum fingerprint"
            )
        if series_fingerprint(series) != PAPER_WINDOW_SHA256:
            raise ValueError(
                "Paper window does not match the expected SHA-256 fingerprint"
            )

    replaced = np.array([], dtype=int)
    if artifact_handling == "interpolate":
        series, replaced = interpolate_artifacts(series)
    elif artifact_handling != "keep":
        raise ValueError(
            f"artifact_handling must be 'keep' or 'interpolate', got {artifact_handling!r}"
        )

    if sum(split_sizes) != series.size:
        raise ValueError(
            f"split_sizes {split_sizes} do not sum to the series length {series.size}"
        )
    # Chronological split on raw values; windows are built inside each split.
    bounds = np.cumsum([0, *split_sizes])
    raw_splits = {
        split: series[bounds[i] : bounds[i + 1]] for i, split in enumerate(SPLIT_NAMES)
    }

    if normalization == "train":
        train = raw_splits["train"]
        shared = SplitScaler(float(train.min()), float(train.max()))
        scalers = dict.fromkeys(SPLIT_NAMES, shared)
    elif normalization == "per_split":
        # Paper-faithful reading of Eq. 9 ("in each of our datasets").
        scalers = {
            split: SplitScaler(float(v.min()), float(v.max()))
            for split, v in raw_splits.items()
        }
    else:
        raise ValueError(
            f"normalization must be 'train' or 'per_split', got {normalization!r}"
        )

    inputs, targets = {}, {}
    for split in SPLIT_NAMES:
        x, y = make_windows(scalers[split].transform(raw_splits[split]), lags)
        inputs[split], targets[split] = x, y

    first_day = dt.date.fromisoformat(start)
    hour_stamps = [first_day + dt.timedelta(hours=int(h)) for h in bounds[:-1]]
    metadata = {
        "name": DATASET_SLUG,
        "source": "IPTO RealTimeSCADASystemLoad",
        "window": [start, end],
        "length": int(series.size),
        "fingerprint_sha256": series_fingerprint(series)
        if artifact_handling == "keep"
        else None,
        "splits": {
            split: {
                "raw_values": int(split_sizes[i]),
                "windows": int(inputs[split].shape[0]),
                "first_hour_offset": int(bounds[i]),
                "first_day": hour_stamps[i].isoformat(),
                "scaler_min": scalers[split].minimum,
                "scaler_max": scalers[split].maximum,
            }
            for i, split in enumerate(SPLIT_NAMES)
        },
        "preprocessing": {
            "lags": lags,
            "normalization": normalization,
            "artifact_handling": artifact_handling,
            "replaced_indices": replaced.tolist(),
        },
    }
    return ForecastData(
        inputs=inputs, targets=targets, scalers=scalers, metadata=metadata
    )


class SplitLoader:
    """Minimal iterable over ``(inputs, targets)`` batches of one split.

    All readouts in this paper are closed-form, so the default batch is the whole
    split (``batch_size: null``).
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray, batch_size: int | None):
        self.inputs, self.targets = inputs, targets
        self.batch_size = batch_size or inputs.shape[0]

    def __len__(self) -> int:
        return -(-self.inputs.shape[0] // self.batch_size)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, self.inputs.shape[0], self.batch_size):
            stop = start + self.batch_size
            yield self.inputs[start:stop], self.targets[start:stop]


def build_dataloader(cfg: dict[str, Any], split: str) -> SplitLoader:
    """Return a batch iterable for ``split`` in ``{"train", "eval", "test"}``."""

    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    data = build_forecast_data(cfg)
    batch_size = cfg["dataset"].get("batch_size")
    return SplitLoader(data.inputs[split], data.targets[split], batch_size)
