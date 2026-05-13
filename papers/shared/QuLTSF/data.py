"""Shared QuLTSF data loader utilities.

Notes on the Weather benchmark schema
-------------------------------------
The canonical `weather.csv` used by QuLTSF comes from the same long-term time
series forecasting benchmark lineage as Autoformer/Informer/Time-Series-Library.
That benchmark convention uses:

- `date` as the timestamp column
- all non-timestamp columns as candidate features
- `OT` as the default target column

The QuLTSF paper itself describes the Weather dataset dimensions and mentions
CO₂ concentration, but does not explicitly name `OT` in the text. We therefore
follow the upstream benchmark code convention rather than inferring a custom
target definition. This was verified against:

- `chariharasuthan/QuLTSF`, which instructs users to download `weather.csv`
  from the Autoformer benchmark bundle
- THUML benchmark loaders (`Dataset_Custom`), whose default target is `OT`

If we later discover the QuLTSF authors used a different target/task mode, this
module is the right place to update the default mapping.

Benchmark task modes
--------------------
We follow the common benchmark task-mode naming:

- `M`: multivariate -> multivariate
- `S`: univariate -> univariate
- `MS`: multivariate -> univariate

This mirrors the conventions used by the upstream LTSF benchmark loaders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from runtime_lib.data_paths import resolve_data_root
from runtime_lib.dtypes import dtype_torch

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[2] / "QuLTSF"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _shared_root(cfg: dict | None = None) -> Path:
    data_root = None if cfg is None else cfg.get("data_root")
    root = resolve_data_root(data_root, project_dir=REPO_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _paper_data_dir(cfg: dict | None = None) -> Path:
    path = (_shared_root(cfg) / "QuLTSF").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _shared_time_series_data_dir(cfg: dict | None = None) -> Path:
    path = (_shared_root(cfg) / "time_series").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_dataset_path(raw_path: str | None, *, cfg: dict | None = None) -> Path:
    if raw_path is None:
        raise ValueError("dataset.path must be provided for QuLTSF")

    path_value = Path(raw_path).expanduser()
    if path_value.is_absolute():
        return path_value

    candidate = (_paper_data_dir(cfg) / path_value).resolve()
    if candidate.exists():
        return candidate

    shared_candidate = (_shared_time_series_data_dir(cfg) / path_value).resolve()
    if shared_candidate.exists():
        return shared_candidate

    repo_shared_candidate = (REPO_ROOT / "data" / "time_series" / path_value).resolve()
    if repo_shared_candidate.exists():
        return repo_shared_candidate

    repo_paper_candidate = (REPO_ROOT / "data" / "QuLTSF" / path_value).resolve()
    if repo_paper_candidate.exists():
        return repo_paper_candidate

    legacy_candidate = (PROJECT_DIR / path_value).resolve()
    if legacy_candidate.exists():
        LOGGER.warning(
            "Using legacy project-local dataset path %s. Prefer data/QuLTSF/ or data/time_series/.",
            legacy_candidate,
        )
        return legacy_candidate

    return candidate


def _normalize_columns(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    shift: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = torch.tensor(frame[columns].to_numpy(), dtype=dtype)
    return (values - shift) / scale


@dataclass(frozen=True)
class WindowMetadata:
    input_size: int
    output_size: int
    input_columns: list[str]
    target_columns: list[str]
    sequence_length: int
    prediction_horizon: int
    rows: int
    feature_shift: list[float]
    feature_scale: list[float]
    target_shift: list[float]
    target_scale: list[float]


class ForecastWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        sequence_length: int,
        prediction_horizon: int,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.sequence_length = int(sequence_length)
        self.prediction_horizon = int(prediction_horizon)
        self._size = max(
            len(inputs) - self.sequence_length - self.prediction_horizon + 1,
            0,
        )

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        stop = start + self.sequence_length
        target_stop = stop + self.prediction_horizon
        x = self.inputs[start:stop]
        y = self.targets[stop:target_stop]
        return x, y


def _build_dataframe(cfg: dict) -> pd.DataFrame:
    dataset_cfg = cfg.get("dataset", {})
    path = _resolve_dataset_path(dataset_cfg.get("path"), cfg=cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset path {path} does not exist. Place the CSV under data/QuLTSF/ or data/time_series/."
        )

    frame = pd.read_csv(path)
    time_column = dataset_cfg.get("time_column")
    if time_column and time_column in frame.columns:
        frame = frame.sort_values(time_column).reset_index(drop=True)
    max_rows = dataset_cfg.get("max_rows")
    if max_rows is not None:
        frame = frame.iloc[: int(max_rows)].copy()
    return frame


def _split_frame(
    frame: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_cfg = cfg.get("dataset", {})
    train_ratio = float(dataset_cfg.get("train_ratio", 0.7))
    val_ratio = float(dataset_cfg.get("val_ratio", 0.1))
    test_ratio = float(dataset_cfg.get("test_ratio", 0.2))
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Dataset split ratios must sum to 1.0")

    n_rows = len(frame)
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:val_end].copy(),
        frame.iloc[val_end:].copy(),
    )


def _compute_shift_scale(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True).clamp_min(1e-6)
    return shift, scale


def _resolve_columns(frame: pd.DataFrame, cfg: dict) -> tuple[list[str], list[str], str]:
    dataset_cfg = cfg.get("dataset", {})
    time_column = str(dataset_cfg.get("time_column", "date"))
    target = str(dataset_cfg.get("target", "OT"))
    features_mode = str(dataset_cfg.get("features_mode", "")).strip().upper()

    non_time_columns = [column for column in frame.columns if column != time_column]
    if target not in frame.columns:
        raise ValueError(
            f"Configured target column {target!r} is not present in dataset columns"
        )

    if features_mode:
        if features_mode not in {"M", "S", "MS"}:
            raise ValueError("dataset.features_mode must be one of M, S, or MS")
        if features_mode == "M":
            return non_time_columns, non_time_columns, features_mode
        if features_mode == "S":
            return [target], [target], features_mode
        return non_time_columns, [target], features_mode

    input_columns = list(dataset_cfg.get("feature_columns", []))
    target_columns = list(dataset_cfg.get("target_columns", input_columns))
    return input_columns, target_columns, "custom"


def build_dataloaders(cfg: dict):
    dataset_cfg = cfg.get("dataset", {})
    dtype = dtype_torch(cfg.get("dtype")) or torch.float32
    frame = _build_dataframe(cfg)

    input_columns, target_columns, features_mode = _resolve_columns(frame, cfg)
    # For the canonical LTSF Weather benchmark:
    # - `features_mode="M"`  means forecast all 21 variables
    # - `features_mode="S"`  means only forecast the target column
    # - `features_mode="MS"` means use all variables to forecast `target` (default `OT`)
    if not input_columns:
        raise ValueError("dataset.feature_columns must not be empty")
    if not target_columns:
        raise ValueError("dataset.target_columns must not be empty")

    train_frame, val_frame, test_frame = _split_frame(frame, cfg)
    train_features = torch.tensor(train_frame[input_columns].to_numpy(), dtype=dtype)
    train_targets = torch.tensor(train_frame[target_columns].to_numpy(), dtype=dtype)
    feature_shift, feature_scale = _compute_shift_scale(train_features)
    target_shift, target_scale = _compute_shift_scale(train_targets)

    sequence_length = int(dataset_cfg.get("sequence_length", 96))
    prediction_horizon = int(dataset_cfg.get("prediction_horizon", 24))
    batch_size = int(dataset_cfg.get("batch_size", 32))

    def _make_dataset(local_frame: pd.DataFrame) -> ForecastWindowDataset:
        return ForecastWindowDataset(
            _normalize_columns(
                local_frame,
                columns=input_columns,
                shift=feature_shift,
                scale=feature_scale,
                dtype=dtype,
            ),
            _normalize_columns(
                local_frame,
                columns=target_columns,
                shift=target_shift,
                scale=target_scale,
                dtype=dtype,
            ),
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
        )

    train_dataset = _make_dataset(train_frame)
    val_dataset = _make_dataset(val_frame)
    test_dataset = _make_dataset(test_frame)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = WindowMetadata(
        input_size=len(input_columns),
        output_size=len(target_columns),
        input_columns=input_columns,
        target_columns=target_columns,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        rows=len(frame),
        feature_shift=feature_shift.squeeze(0).tolist(),
        feature_scale=feature_scale.squeeze(0).tolist(),
        target_shift=target_shift.squeeze(0).tolist(),
        target_scale=target_scale.squeeze(0).tolist(),
    ).__dict__
    metadata["splits"] = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }
    metadata["features_mode"] = features_mode
    metadata["target"] = str(dataset_cfg.get("target", "OT"))
    metadata["dataset_path"] = str(_resolve_dataset_path(dataset_cfg.get("path"), cfg=cfg))
    return train_loader, val_loader, test_loader, metadata


__all__ = ["build_dataloaders"]
