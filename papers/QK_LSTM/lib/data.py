"""CPCB Bengaluru AQI data pipeline (arXiv:2412.08851, Sec. II).

Source file: ``<data_root>/QK_LSTM/cpcb_aqi_india_2015_2020/city_day.csv``
(Kaggle "Air Quality Data in India (2015-2020)"; provenance in LOG.md, Data
Acquisition Log). The data root follows the shared convention: ``--data-root``
on the CLI, then the ``DATA_DIR`` environment variable, then ``<repo>/data``.

Preprocessing, in order (LOG.md, Q10; the paper states only the Z-score rule
with gamma = 3 and linear interpolation):

1. Select ``City == city`` and the 11 pollutant inputs (Xylene dropped, Sec. II-A).
2. Fit per-feature mean and standard deviation on the training-fit days only.
   Input values with ``|Z| > outlier_z`` become missing (Eq. 2). The AQI target
   is not outlier-filtered.
3. Fill missing values (``dataset.fill``). ``"interpolate"`` (default, paper Eq. 3):
   linear interpolation in time, which can use later observations to fill a gap.
   ``"causal"``: carry the last observation forward, so a value never depends on
   later days. In both modes, values before a column's first observation are
   filled with that first observation (the only non-causal step left).
4. Keep a mask of the days whose AQI was originally observed. Test metrics are
   computed on those days only (filled targets are not scored).
5. Min-max scale inputs and target to ``[0, 1]`` with training-fit statistics.

A sample is a window of ``sequence_length`` consecutive days of inputs
``x_{t-L+1..t}`` and the target ``AQI_{t+horizon}``. Windows are assigned to a
split by their target day, chronologically, so no target of a later split is
seen in training.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from runtime_lib.data_paths import resolve_data_root

SPLITS = ("train", "validation", "test")


@dataclass
class PreparedSeries:
    """Preprocessed daily series and split assignment.

    Attributes
    ----------
    dates : numpy.ndarray
        ``(T,)`` daily dates (``datetime64[D]``).
    features : numpy.ndarray
        ``(T, F)`` scaled inputs.
    target : numpy.ndarray
        ``(T,)`` scaled AQI.
    target_aqi : numpy.ndarray
        ``(T,)`` AQI in original units after interpolation.
    target_observed : numpy.ndarray
        ``(T,)`` bool, True where AQI was present in the raw file.
    target_min, target_max : float
        Min-max statistics used to scale the target.
    window_end : dict
        Split name to ``(n,)`` array of window end days ``t``; the target day is
        ``t + horizon``.
    info : dict
        Dataset identity, preprocessing and split description for run status.
    """

    dates: np.ndarray
    features: np.ndarray
    target: np.ndarray
    target_aqi: np.ndarray
    target_observed: np.ndarray
    target_min: float
    target_max: float
    sequence_length: int
    horizon: int
    window_end: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def to_aqi(self, scaled: np.ndarray) -> np.ndarray:
        """Invert the target min-max scaling to AQI units."""
        return (
            np.asarray(scaled) * (self.target_max - self.target_min) + self.target_min
        )


def _required(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise KeyError(f"Missing required config key dataset.{key}")
    return section[key]


def resolve_csv_path(cfg: dict[str, Any]) -> Path:
    """Resolve the raw CSV path from ``cfg['data_root']`` and the dataset section."""
    dataset_cfg = cfg["dataset"]
    data_root = resolve_data_root(cfg.get("data_root"))
    path = data_root / _required(dataset_cfg, "subdir") / _required(dataset_cfg, "file")
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Place city_day.csv as described in "
            "papers/QK_LSTM/README.md (Data) or pass --data-root."
        )
    return path


def _split_boundaries(
    dates: pd.DatetimeIndex, split_cfg: dict[str, Any]
) -> tuple[int, int]:
    """Return ``(validation_start, test_start)`` day indices of the target days."""
    n_days = len(dates)
    scheme = split_cfg["scheme"]
    if scheme == "ratio":
        test_start = int(n_days * (1.0 - float(split_cfg["test_fraction"])))
    elif scheme == "date":
        test_start = int(
            np.searchsorted(dates.values, np.datetime64(split_cfg["test_start"]))
        )
    else:
        raise ValueError(f"Unknown split scheme {scheme!r}; expected 'ratio' or 'date'")
    validation_days = int(test_start * float(split_cfg["validation_fraction"]))
    validation_start = test_start - validation_days
    if not 0 < validation_start <= test_start < n_days:
        raise ValueError(
            f"Invalid split boundaries: validation {validation_start}, test {test_start}"
        )
    return validation_start, test_start


def fill_missing(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Fill missing values of a date-indexed frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        Daily series indexed by date; NaN marks missing values.
    mode : str
        ``"interpolate"``: time-linear interpolation (paper Eq. 3), which can use
        later observations. ``"causal"``: forward-fill, so a filled value never
        depends on later days.

    Returns
    -------
    pandas.DataFrame
        Filled frame. In both modes, gaps before a column's first observation are
        filled with that first observation (``bfill``), the only non-causal step.

    Raises
    ------
    ValueError
        If ``mode`` is unknown.
    """
    if mode == "interpolate":
        return frame.interpolate(method="time").ffill().bfill()
    if mode == "causal":
        return frame.ffill().bfill()
    raise ValueError(f"Unknown fill mode {mode!r}; expected 'interpolate' or 'causal'")


def prepare_series(cfg: dict[str, Any]) -> PreparedSeries:
    """Load, clean, scale and split the configured city's daily series.

    Parameters
    ----------
    cfg : dict
        Resolved run configuration with a ``dataset`` section.

    Returns
    -------
    PreparedSeries
        Preprocessed series with split assignments.
    """
    dataset_cfg = cfg["dataset"]
    csv_path = resolve_csv_path(cfg)
    city = _required(dataset_cfg, "city")
    feature_columns = list(_required(dataset_cfg, "features"))
    target_column = _required(dataset_cfg, "target")
    sequence_length = int(_required(dataset_cfg, "sequence_length"))
    horizon = int(_required(dataset_cfg, "horizon"))
    outlier_z = float(_required(dataset_cfg, "outlier_z"))
    split_cfg = _required(dataset_cfg, "split")
    fill = _required(dataset_cfg, "fill")
    if fill not in ("interpolate", "causal"):
        raise ValueError(
            f"Unknown dataset.fill {fill!r}; expected 'interpolate' or 'causal'"
        )

    raw = pd.read_csv(csv_path, parse_dates=["Date"])
    # dict.fromkeys de-duplicates a target that is also an input (known-answer test).
    columns = list(dict.fromkeys(["Date", *feature_columns, target_column]))
    frame = raw.loc[raw["City"] == city, columns]
    if frame.empty:
        raise ValueError(f"No rows for city {city!r} in {csv_path}")
    frame = frame.sort_values("Date").set_index("Date")
    expected_days = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    if len(frame) != len(expected_days) or frame.index.has_duplicates:
        raise ValueError(f"{city} series is not one row per calendar day")

    validation_start, test_start = _split_boundaries(frame.index, split_cfg)
    # Statistics are fitted on the days the model trains on: everything before
    # the validation block (or before the test block if validation is empty).
    fit_days = slice(0, validation_start)

    features = frame[feature_columns].astype(float)
    fit_mean = features.iloc[fit_days].mean()
    fit_std = features.iloc[fit_days].std()
    z_scores = (features - fit_mean) / fit_std
    outlier_mask = z_scores.abs() > outlier_z
    features = features.mask(outlier_mask)

    target_observed = frame[target_column].notna().to_numpy()
    # The target is carried as its own column: it is not outlier-filtered even
    # when the same variable is also an input feature.
    target_internal = "__target__"
    filled = pd.concat(
        [features, frame[target_column].astype(float).rename(target_internal)], axis=1
    )
    filled = fill_missing(filled, fill)

    fit_block = filled.iloc[fit_days]
    feature_min = fit_block[feature_columns].min()
    feature_range = (fit_block[feature_columns].max() - feature_min).replace(0.0, 1.0)
    target_min = float(fit_block[target_internal].min())
    target_max = float(fit_block[target_internal].max())

    scaled_features = (
        (filled[feature_columns] - feature_min) / feature_range
    ).to_numpy(np.float32)
    target_aqi = filled[target_internal].to_numpy(np.float64)
    scaled_target = ((target_aqi - target_min) / (target_max - target_min)).astype(
        np.float32
    )

    window_ends = np.arange(sequence_length - 1, len(frame) - horizon)
    target_days = window_ends + horizon
    window_end = {
        "train": window_ends[target_days < validation_start],
        "validation": window_ends[
            (target_days >= validation_start) & (target_days < test_start)
        ],
        "test": window_ends[target_days >= test_start],
    }
    dates = frame.index.values.astype("datetime64[D]")

    def _date_range(ends: np.ndarray) -> list[str] | None:
        if ends.size == 0:
            return None
        return [str(dates[ends[0] + horizon]), str(dates[ends[-1] + horizon])]

    info = {
        "name": dataset_cfg["name"],
        "file": str(csv_path),
        "file_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "subset": f"City == {city}; {len(frame)} days {dates[0]}..{dates[-1]}",
        "splits": {
            name: {
                "windows": int(ends.size),
                "target_dates": _date_range(ends),
                "scored_targets": int(target_observed[ends + horizon].sum())
                if ends.size
                else 0,
            }
            for name, ends in window_end.items()
        },
        "preprocessing": (
            f"inputs {feature_columns}; |Z|>{outlier_z} input outliers (train-fit stats) -> NaN "
            f"({int(outlier_mask.to_numpy().sum())} values); "
            f"{'time-linear interpolation' if fill == 'interpolate' else 'causal forward-fill'} + edge fill; "
            "min-max [0,1] fitted on train-fit days; "
            f"windows of {sequence_length} days -> {target_column}(t+{horizon}); "
            f"split {split_cfg}"
        ),
    }
    return PreparedSeries(
        dates=dates,
        features=scaled_features,
        target=scaled_target,
        target_aqi=target_aqi,
        target_observed=target_observed,
        target_min=target_min,
        target_max=target_max,
        sequence_length=sequence_length,
        horizon=horizon,
        window_end=window_end,
        info=info,
    )


class WindowDataset(Dataset):
    """Sliding windows of one split.

    Each item is ``(x, y, target_day)`` with ``x`` of shape
    ``(sequence_length, n_features)``, ``y`` of shape ``(1,)`` (scaled AQI at the
    target day) and ``target_day`` the integer day index of the target.
    """

    def __init__(
        self, series: PreparedSeries, split: str, max_windows: int | None = None
    ):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
        ends = series.window_end[split]
        if max_windows is not None:
            # Keep the most recent windows so the subset stays chronological.
            ends = ends[-int(max_windows) :]
        self.series = series
        self.window_end = ends

    def __len__(self) -> int:
        return int(self.window_end.size)

    def __getitem__(self, index: int):
        end = int(self.window_end[index])
        start = end - self.series.sequence_length + 1
        target_day = end + self.series.horizon
        x = torch.from_numpy(self.series.features[start : end + 1])
        y = torch.tensor([self.series.target[target_day]], dtype=torch.float32)
        return x, y, target_day


def build_dataloader(
    cfg: dict[str, Any],
    split: str,
    series: PreparedSeries | None = None,
) -> DataLoader:
    """Build the data loader of one split.

    Parameters
    ----------
    cfg : dict
        Resolved configuration. Uses ``dataset.batch_size``,
        ``dataset.shuffle_train``, ``dataset.max_train_windows`` and ``seed``.
    split : str
        ``"train"``, ``"validation"`` or ``"test"``.
    series : PreparedSeries | None
        Already prepared series, to avoid reloading. If omitted it is built
        from ``cfg``. Default value is None.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader yielding ``(x, y, target_day)`` batches. Only the training split
        is shuffled (each window is an independent sequence).
    """
    if series is None:
        series = prepare_series(cfg)
    dataset_cfg = cfg["dataset"]
    max_windows = dataset_cfg.get("max_train_windows") if split == "train" else None
    dataset = WindowDataset(series, split, max_windows=max_windows)
    shuffle = bool(dataset_cfg["shuffle_train"]) and split == "train"
    generator = torch.Generator().manual_seed(int(cfg["seed"])) if shuffle else None
    batch_size = int(dataset_cfg["batch_size"]) if split == "train" else 256
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )
