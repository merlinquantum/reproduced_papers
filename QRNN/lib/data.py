from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from runtime_lib.dtypes import dtype_torch

LOGGER = logging.getLogger(__name__)


def _find_first_csv(directory: Path) -> Path:
    candidates = sorted(directory.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found under downloaded dataset directory: {directory}"
        )
    return candidates[0]


def resolve_dataset_path(dataset_cfg: dict) -> Path:
    """Resolve the dataset path, optionally downloading via kagglehub."""

    path = Path(dataset_cfg.get("path", "")).expanduser().resolve()
    if path.exists():
        return path

    if dataset_cfg.get("use_kagglehub"):
        import kagglehub

        dataset_id = dataset_cfg.get("kaggle_dataset", "thedevastator/weather-prediction")
        LOGGER.info("Downloading dataset '%s' via kagglehub", dataset_id)
        dataset_dir = Path(kagglehub.dataset_download(dataset_id)).resolve()
        csv_path = _find_first_csv(dataset_dir)
        LOGGER.info("Using CSV file discovered at: %s", csv_path)
        return csv_path

    raise FileNotFoundError(
        f"Dataset path {path} does not exist and use_kagglehub is disabled"
    )


def _select_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df[list(columns)].copy()


class WeatherSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window time-series dataset for weather forecasting."""

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str,
        sequence_length: int,
        prediction_horizon: int,
        dtype: torch.dtype,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")

        self.features = _select_columns(data, feature_columns).to_numpy(dtype=float)
        self.targets = _select_columns(data, [target_column]).to_numpy(dtype=float).ravel()
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.dtype = dtype

    def __len__(self) -> int:
        return max(0, len(self.targets) - self.sequence_length - self.prediction_horizon + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.sequence_length
        target_idx = end + self.prediction_horizon - 1
        sequence = torch.as_tensor(self.features[start:end], dtype=self.dtype)
        target = torch.as_tensor(self.targets[target_idx], dtype=self.dtype)
        return sequence, target


def _normalize_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    normalized = (tensor - mean) / std
    return normalized, mean, std


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, dict]:
    dataset_cfg = cfg.get("dataset", {})
    dtype = dtype_torch(cfg.get("dtype")) or torch.float32

    csv_path = resolve_dataset_path(dataset_cfg)
    raw_df = pd.read_csv(csv_path)
    feature_columns = dataset_cfg.get("feature_columns") or []
    target_column = dataset_cfg.get("target_column")
    if not target_column:
        raise ValueError("dataset.target_column must be provided")

    sequence_length = int(dataset_cfg.get("sequence_length", 8))
    prediction_horizon = int(dataset_cfg.get("prediction_horizon", 1))

    dataset = WeatherSequenceDataset(
        raw_df,
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        dtype=dtype,
    )

    total_len = len(dataset)
    if total_len == 0:
        raise ValueError("Dataset is too small for the requested sequence and horizon lengths")

    train_ratio = float(dataset_cfg.get("train_ratio", 0.7))
    val_ratio = float(dataset_cfg.get("val_ratio", 0.15))
    train_cutoff = int(total_len * train_ratio)
    val_cutoff = int(total_len * (train_ratio + val_ratio))
    train_indices = list(range(0, train_cutoff))
    val_indices = list(range(train_cutoff, val_cutoff))

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)

    batch_size = int(dataset_cfg.get("batch_size", 16))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    shuffle = bool(dataset_cfg.get("shuffle", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    sample_input, _ = dataset[0]
    _, mean, std = _normalize_tensor(sample_input)

    metadata = {
        "input_size": sample_input.shape[-1],
        "sequence_length": sequence_length,
        "prediction_horizon": prediction_horizon,
        "feature_mean": mean.squeeze().tolist(),
        "feature_std": std.squeeze().tolist(),
        "dataset_path": str(csv_path),
    }
    return train_loader, val_loader, metadata


__all__ = ["WeatherSequenceDataset", "build_dataloaders", "resolve_dataset_path"]
