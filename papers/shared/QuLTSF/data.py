"""LTSF benchmark data pipeline used by the QuLTSF reproduction.

This module reimplements ``Dataset_Custom`` and ``data_provider`` from the
upstream QuLTSF code (``data_provider/data_loader.py`` and
``data_provider/data_factory.py``, themselves taken from LTSF-Linear), so that
splits, scaling and window counts match the paper's pipeline exactly:

- chronological split on rows: ``num_train = int(0.7 n)``,
  ``num_test = int(0.2 n)``, validation takes the remaining rows;
- the validation and test windows start ``seq_len`` rows before their split
  boundary, so every target row of a split is predicted from a full lookback;
- a standard scaler (population std, ``ddof=0`` as in scikit-learn) is fitted
  on the training rows only and applied to the whole series;
- window ``i`` of a split uses rows ``[i, i + seq_len)`` as input and rows
  ``[i + seq_len, i + seq_len + pred_len)`` as target, so a split of ``r`` rows
  has ``r - seq_len - pred_len + 1`` windows.

The time-feature columns and ``label_len`` decoder tokens of the upstream
loader are not built: none of the models in this reproduction read them.

Feature modes follow the benchmark convention: ``M`` (all channels in, all
channels out), ``S`` (target channel only) and ``MS`` (all channels in, target
channel scored). Column order is ``date``, the other features, then the target
last, as upstream.

A deterministic synthetic series with the same file layout is available for
smoke runs (``dataset.source = "synthetic"``). It carries no scientific
meaning and must never be reported as a Weather result.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from runtime_lib.data_paths import paper_data_dir

logger = logging.getLogger(__name__)

PAPER_NAME = "QuLTSF"
SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}
FEATURE_MODES = ("M", "S", "MS")


@dataclass(frozen=True)
class SeriesSplits:
    """Scaled series and the row borders of each split.

    Attributes
    ----------
    values : numpy.ndarray
        Scaled series of shape ``[n_rows, n_channels]`` (float64).
    border_starts : tuple[int, int, int]
        First row of the train, val and test segments (lookback included).
    border_ends : tuple[int, int, int]
        End row (exclusive) of the train, val and test segments.
    columns : tuple[str, ...]
        Channel names in model order (target last for ``M`` and ``MS``).
    scaler_mean : numpy.ndarray
        Per-channel mean fitted on the training rows.
    scaler_std : numpy.ndarray
        Per-channel population standard deviation fitted on the training rows.
    source : dict
        Dataset identity for ``run_status.json`` (name, path, sha256, rows).
    """

    values: np.ndarray
    border_starts: tuple[int, int, int]
    border_ends: tuple[int, int, int]
    columns: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    source: dict


class ForecastWindows(Dataset):
    """Sliding forecast windows over one split of a scaled series.

    Parameters
    ----------
    segment : numpy.ndarray
        Scaled rows of the split, lookback rows included, ``[rows, channels]``.
    seq_len : int
        Lookback length L.
    pred_len : int
        Forecast horizon T.
    dtype : torch.dtype
        Floating dtype of the returned tensors.
    """

    def __init__(
        self, segment: np.ndarray, seq_len: int, pred_len: int, dtype: torch.dtype
    ) -> None:
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.data = torch.as_tensor(segment, dtype=dtype)
        if len(self) <= 0:
            raise ValueError(
                f"Split has {segment.shape[0]} rows, too few for "
                f"seq_len={seq_len} and pred_len={pred_len}."
            )

    def __len__(self) -> int:
        return self.data.shape[0] - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_end = index + self.seq_len
        return (
            self.data[index:input_end],
            self.data[input_end : input_end + self.pred_len],
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_weather_frame(n_rows: int, n_channels: int, seed: int) -> pd.DataFrame:
    """Build a deterministic multichannel series shaped like ``weather.csv``.

    Each channel is a sum of a daily (144-step) and a slower sinusoid with a
    channel-specific phase and amplitude, plus Gaussian noise.

    Parameters
    ----------
    n_rows : int
        Number of 10-minute time steps.
    n_channels : int
        Number of channels; the last one is named ``OT``.
    seed : int
        Seed of the noise generator.

    Returns
    -------
    pandas.DataFrame
        Frame with a ``date`` column followed by the channels.
    """
    rng = np.random.default_rng(seed)
    steps = np.arange(n_rows)[:, None]
    channel = np.arange(n_channels)[None, :]
    daily = np.sin(2 * np.pi * steps / 144 + channel)
    slow = 0.5 * np.sin(2 * np.pi * steps / 1008 + 0.3 * channel)
    values = (1 + 0.1 * channel) * (daily + slow) + 0.1 * rng.standard_normal(
        (n_rows, n_channels)
    )
    names = [f"x{index}" for index in range(n_channels - 1)] + ["OT"]
    frame = pd.DataFrame(values, columns=names)
    frame.insert(
        0, "date", pd.date_range("2020-01-01 00:10:00", periods=n_rows, freq="10min")
    )
    return frame


def _require(dataset_cfg: dict, key: str):
    if key not in dataset_cfg:
        raise KeyError(f"Missing required config key dataset.{key}")
    return dataset_cfg[key]


def _load_frame(cfg: dict) -> tuple[pd.DataFrame, dict]:
    dataset_cfg = cfg["dataset"]
    source = _require(dataset_cfg, "source")
    if source == "synthetic":
        synthetic_cfg = _require(dataset_cfg, "synthetic")
        frame = synthetic_weather_frame(
            n_rows=int(synthetic_cfg["n_rows"]),
            n_channels=int(synthetic_cfg["n_channels"]),
            seed=int(synthetic_cfg["seed"]),
        )
        identity = {
            "name": "synthetic_weather_like",
            "path": None,
            "sha256": None,
            "rows": len(frame),
            "generator": dict(synthetic_cfg),
        }
        return frame, identity
    if source != "csv":
        raise ValueError(f"dataset.source must be 'csv' or 'synthetic', got {source!r}")

    relative_path = Path(_require(dataset_cfg, "path"))
    data_dir = paper_data_dir(PAPER_NAME, cfg.get("data_root"), ensure_exists=False)
    csv_path = (
        relative_path if relative_path.is_absolute() else data_dir / relative_path
    )
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. See the README Data section "
            "for the download command."
        )
    # The Weather header contains U+FFFD in three unit names; the file is UTF-8.
    frame = pd.read_csv(csv_path, encoding="utf-8")
    identity = {
        "name": str(_require(dataset_cfg, "name")),
        "path": str(relative_path),
        "sha256": _file_sha256(csv_path),
        "rows": len(frame),
    }
    expected_sha256 = dataset_cfg.get("sha256")
    if expected_sha256 and identity["sha256"] != expected_sha256:
        raise ValueError(
            f"{csv_path} sha256 {identity['sha256']} does not match the "
            f"configured dataset.sha256 {expected_sha256}."
        )
    return frame, identity


def load_series(cfg: dict) -> SeriesSplits:
    """Load, order, split and scale the series exactly as upstream.

    Parameters
    ----------
    cfg : dict
        Resolved configuration with a ``dataset`` section.

    Returns
    -------
    SeriesSplits
        Scaled values, split borders, channel names and dataset identity.

    Raises
    ------
    KeyError
        If a required ``dataset`` key is missing.
    ValueError
        If the feature mode is unknown or the series is too short.
    """
    dataset_cfg = cfg["dataset"]
    seq_len = int(_require(dataset_cfg, "sequence_length"))
    features_mode = str(_require(dataset_cfg, "features_mode"))
    target = str(_require(dataset_cfg, "target"))
    time_column = str(_require(dataset_cfg, "time_column"))
    if features_mode not in FEATURE_MODES:
        raise ValueError(f"features_mode must be one of {FEATURE_MODES}")

    frame, identity = _load_frame(cfg)
    other_columns = [c for c in frame.columns if c not in (time_column, target)]
    frame = frame[[time_column, *other_columns, target]]

    n_rows = len(frame)
    num_train = int(n_rows * float(_require(dataset_cfg, "train_ratio")))
    num_test = int(n_rows * float(_require(dataset_cfg, "test_ratio")))
    num_val = n_rows - num_train - num_test
    border_starts = (0, num_train - seq_len, n_rows - num_test - seq_len)
    border_ends = (num_train, num_train + num_val, n_rows)

    if features_mode in ("M", "MS"):
        columns = tuple(frame.columns[1:])
    else:
        columns = (target,)
    raw = frame[list(columns)].to_numpy(dtype=np.float64)

    train_rows = raw[border_starts[0] : border_ends[0]]
    scaler_mean = train_rows.mean(axis=0)
    scaler_std = train_rows.std(axis=0)
    if np.any(scaler_std == 0):
        raise ValueError("A channel is constant over the training rows.")
    values = (raw - scaler_mean) / scaler_std

    identity = {
        **identity,
        "splits": {
            name: {
                "rows": [border_starts[i], border_ends[i]],
                "target_rows": [
                    border_starts[i] + (seq_len if i else 0),
                    border_ends[i],
                ],
            }
            for name, i in SPLIT_INDEX.items()
        },
    }
    return SeriesSplits(
        values=values,
        border_starts=border_starts,
        border_ends=border_ends,
        columns=columns,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        source=identity,
    )


def build_dataset(
    series: SeriesSplits, cfg: dict, split: str, dtype: torch.dtype
) -> ForecastWindows:
    """Return the forecast windows of one split.

    Parameters
    ----------
    series : SeriesSplits
        Output of :func:`load_series`.
    cfg : dict
        Resolved configuration.
    split : str
        ``"train"``, ``"val"`` or ``"test"``.
    dtype : torch.dtype
        Floating dtype of the tensors.

    Returns
    -------
    ForecastWindows
        Dataset of ``(input [L, C], target [T, C])`` pairs.
    """
    index = SPLIT_INDEX[split]
    segment = series.values[series.border_starts[index] : series.border_ends[index]]
    return ForecastWindows(
        segment,
        seq_len=int(cfg["dataset"]["sequence_length"]),
        pred_len=int(cfg["dataset"]["prediction_horizon"]),
        dtype=dtype,
    )


class WindowBatches:
    """Batches of forecast windows gathered directly on the target device.

    Same batches, in the same order, as
    ``DataLoader(dataset, batch_size, shuffle, drop_last)`` over
    :class:`ForecastWindows`, but each batch is one index gather on a tensor
    that already lives on the device, without per-window Python calls and
    host-to-device copies.

    The global torch RNG is consumed exactly as ``DataLoader`` consumes it, so
    a seeded run draws the same shuffles as before: every pass draws one int64
    (the iterator's ``_base_seed``), and a shuffled pass then draws a second
    int64 that seeds a fresh generator for ``torch.randperm`` (``RandomSampler``).
    ``tests/test_data.py`` checks the equivalence.

    Parameters
    ----------
    dataset : ForecastWindows
        Windows of one split.
    batch_size : int
        Windows per batch.
    shuffle : bool
        Reshuffle the windows at every pass.
    drop_last : bool
        Drop the last incomplete batch.
    device : torch.device | str
        Device holding the series and the batches.
    """

    def __init__(
        self,
        dataset: ForecastWindows,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        device: torch.device | str,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.data = dataset.data.to(device)
        self.window_offsets = torch.arange(
            dataset.seq_len + dataset.pred_len, device=self.data.device
        )

    def __len__(self) -> int:
        n_windows = len(self.dataset)
        if self.drop_last:
            return n_windows // self.batch_size
        return -(-n_windows // self.batch_size)

    def _order(self) -> torch.Tensor:
        n_windows = len(self.dataset)
        torch.empty((), dtype=torch.int64).random_()  # DataLoader iterator _base_seed
        if not self.shuffle:
            return torch.arange(n_windows)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randperm(n_windows, generator=generator)

    def __iter__(self):
        order = self._order().to(self.data.device)
        seq_len = self.dataset.seq_len
        for batch in range(len(self)):
            starts = order[batch * self.batch_size : (batch + 1) * self.batch_size]
            windows = self.data[starts[:, None] + self.window_offsets]
            yield windows[:, :seq_len], windows[:, seq_len:]


def build_dataloader(
    series: SeriesSplits,
    cfg: dict,
    split: str,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> WindowBatches:
    """Return the batches of one split with the upstream batching rules.

    Train batches are shuffled; train and validation drop the last incomplete
    batch, as upstream. The test batches keep every window
    (``dataset.test_drop_last`` is false in all paper configs); the upstream
    ``drop_last=True`` test metric is computed from the per-window errors by
    the evaluation code instead, so one pass yields both numbers.

    Parameters
    ----------
    series : SeriesSplits
        Output of :func:`load_series`.
    cfg : dict
        Resolved configuration.
    split : str
        ``"train"``, ``"val"`` or ``"test"``.
    dtype : torch.dtype
        Floating dtype of the tensors.
    device : torch.device | str
        Device of the batches. Default value is ``"cpu"``.

    Returns
    -------
    WindowBatches
        Iterable over ``(input [B, L, C], target [B, T, C])`` batches.
    """
    dataset_cfg = cfg["dataset"]
    batch_size = int(_require(dataset_cfg, "batch_size"))
    drop_last = {
        "train": True,
        "val": bool(_require(dataset_cfg, "val_drop_last")),
        "test": bool(_require(dataset_cfg, "test_drop_last")),
    }[split]
    return WindowBatches(
        build_dataset(series, cfg, split, dtype),
        batch_size=batch_size,
        shuffle=split == "train",
        drop_last=drop_last,
        device=device,
    )


__all__ = [
    "FEATURE_MODES",
    "ForecastWindows",
    "SeriesSplits",
    "WindowBatches",
    "build_dataloader",
    "build_dataset",
    "load_series",
    "synthetic_weather_frame",
]
