from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from runtime_lib.data_paths import paper_data_dir
from runtime_lib.dtypes import dtype_torch

from .preprocess import PREPROCESSORS

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]


def _compute_feature_shift_scale(
    values: torch.Tensor,
    *,
    scheme: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute affine feature normalization parameters.

    We keep the historical names feature_mean/feature_std in datasets, but they
    semantically represent an affine transform:

        x_norm = (x - shift) / scale

    where (shift, scale) depend on the chosen scheme.
    """

    scheme_norm = str(scheme or "standard").strip().lower()
    if scheme_norm in {"standard", "zscore", "z-score"}:
        shift = values.mean(dim=0, keepdim=True)
        scale = values.std(dim=0, keepdim=True)
    elif scheme_norm in {"none", "identity"}:
        shift = torch.zeros((1, values.shape[1]), dtype=dtype)
        scale = torch.ones((1, values.shape[1]), dtype=dtype)
    elif scheme_norm in {"minmax_0_1", "minmax01", "minmax"}:
        vmin = values.amin(dim=0, keepdim=True)
        vmax = values.amax(dim=0, keepdim=True)
        shift = vmin
        scale = vmax - vmin
    elif scheme_norm in {"minmax_-1_1", "minmax11", "minmax_signed"}:
        vmin = values.amin(dim=0, keepdim=True)
        vmax = values.amax(dim=0, keepdim=True)
        shift = (vmin + vmax) / 2.0
        scale = (vmax - vmin) / 2.0
    else:
        raise ValueError(
            "Unsupported dataset.feature_normalization scheme "
            f"{scheme!r} (expected: standard, none, minmax_0_1, minmax_-1_1)"
        )

    scale = scale.to(dtype=dtype).clamp_min(1e-6)
    shift = shift.to(dtype=dtype)
    return shift, scale


def _paper_data_dir(cfg: dict | None = None) -> Path:
    data_root = None
    if cfg is not None:
        data_root = cfg.get("data_root")
    return paper_data_dir("QRNN", data_root=data_root, ensure_exists=True)


def _shared_time_series_data_dir(cfg: dict | None = None) -> Path:
    data_root = None
    if cfg is not None:
        data_root = cfg.get("data_root")
    return paper_data_dir("time_series", data_root=data_root, ensure_exists=True)


def _find_first_csv(directory: Path) -> Path:
    candidates = sorted(directory.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found under downloaded dataset directory: {directory}"
        )
    return candidates[0]


def _resolve_dataset_path(
    raw_path: str | Path | None, *, cfg: dict | None = None
) -> Path | None:
    if raw_path is None:
        return None
    path_value = Path(raw_path).expanduser()
    if not path_value.is_absolute():
        # New convention: paths in configs are interpreted relative to the
        # per-paper shared data directory: <data_root>/QRNN/.
        # For backwards compatibility, accept legacy prefixes like "data/...".
        path_str = path_value.as_posix()
        if path_str.startswith("data/"):
            path_value = Path(path_str[len("data/") :])

        candidate = (_paper_data_dir(cfg) / path_value).resolve()
        if candidate.exists():
            path_value = candidate
        else:
            # New shared convention: allow generic time-series datasets to live
            # under <data_root>/time_series/.
            ts_candidate = (_shared_time_series_data_dir(cfg) / path_value).resolve()
            if ts_candidate.exists():
                path_value = ts_candidate
                return path_value

            # Backwards compatibility: older runs stored datasets under
            # papers/QRNN/data/ (or relative to the project directory).
            legacy = (PROJECT_DIR / Path(raw_path)).resolve()
            if legacy.exists():
                LOGGER.warning(
                    "Using legacy dataset path under project dir (%s). "
                    "Consider moving data under $DATA_DIR/QRNN or using --data-root.",
                    legacy,
                )
                path_value = legacy
            else:
                path_value = candidate
    return path_value


def _derive_preprocessed_path(raw_path: Path) -> Path:
    return raw_path.with_name(f"{raw_path.stem}.preprocess{raw_path.suffix}")


def _maybe_preprocess(csv_path: Path, dataset_cfg: dict) -> Path:
    preprocess_name = dataset_cfg.get("preprocess")
    if not preprocess_name:
        return csv_path

    if csv_path.name.endswith(".preprocess.csv") and csv_path.exists():
        return csv_path

    try:
        preprocessor = PREPROCESSORS[preprocess_name]
    except KeyError as exc:  # pragma: no cover - guardrail
        raise ValueError(f"Unknown dataset preprocess '{preprocess_name}'") from exc

    preprocessed_path = _derive_preprocessed_path(csv_path)
    return preprocessor(csv_path, preprocessed_path)


def resolve_dataset_path(dataset_cfg: dict, *, cfg: dict | None = None) -> Path:
    """Resolve the dataset path, downloading via kagglehub only when needed."""

    configured_path = _resolve_dataset_path(dataset_cfg.get("path"), cfg=cfg)
    if configured_path and configured_path.exists():
        return _maybe_preprocess(configured_path, dataset_cfg)

    kaggle_dataset = dataset_cfg.get("kaggle_dataset")
    if kaggle_dataset:
        import kagglehub

        LOGGER.info("Downloading dataset '%s' via kagglehub", kaggle_dataset)
        dataset_dir = Path(kagglehub.dataset_download(kaggle_dataset)).resolve()
        csv_path = _find_first_csv(dataset_dir)

        destination = configured_path or (_paper_data_dir(cfg) / csv_path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            LOGGER.info("Using existing dataset at %s", destination)
            return _maybe_preprocess(destination, dataset_cfg)

        shutil.copyfile(csv_path, destination)
        LOGGER.info("Copied dataset to %s", destination)
        return _maybe_preprocess(destination, dataset_cfg)

    msg_path = configured_path or (_paper_data_dir(cfg) / "dataset.csv")
    raise FileNotFoundError(
        f"Dataset path {msg_path} does not exist and no kaggle_dataset is configured"
    )


def _sanitize_column_name(name: str) -> str:
    no_paren = re.sub(r"\([^)]*\)", "", name)
    return re.sub(r"[^a-z0-9]", "", no_paren.lower())


def _select_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    alias_map: dict[str, str] = {}
    for existing in df.columns:
        sanitized = _sanitize_column_name(existing)
        alias_map.setdefault(sanitized, existing)

    resolved: list[str] = []
    missing: list[str] = []
    for requested in columns:
        if requested in df.columns:
            resolved.append(requested)
            continue
        alias = alias_map.get(_sanitize_column_name(requested))
        if alias:
            resolved.append(alias)
        else:
            missing.append(requested)

    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df[resolved].copy()


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
        self.targets = (
            _select_columns(data, [target_column]).to_numpy(dtype=float).ravel()
        )
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.dtype = dtype
        self.feature_normalization = "standard"
        # Normalization stats (populated later from the training split)
        self.feature_mean = torch.zeros((1, self.features.shape[1]), dtype=self.dtype)
        self.feature_std = torch.ones((1, self.features.shape[1]), dtype=self.dtype)
        self.target_mean = torch.tensor(0.0, dtype=self.dtype)
        self.target_std = torch.tensor(1.0, dtype=self.dtype)

    def __len__(self) -> int:
        return max(
            0, len(self.targets) - self.sequence_length - self.prediction_horizon + 1
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.sequence_length
        target_idx = end + self.prediction_horizon - 1
        raw_seq = torch.as_tensor(self.features[start:end], dtype=self.dtype)
        sequence = (raw_seq - self.feature_mean) / self.feature_std
        raw_target = torch.as_tensor(self.targets[target_idx], dtype=self.dtype)
        target = (raw_target - self.target_mean) / self.target_std
        return sequence, target

    def set_normalization(
        self,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        *,
        feature_normalization: str = "standard",
    ) -> None:
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.feature_normalization = str(feature_normalization)

    def compute_stats_for_indices(
        self, indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not indices:
            raise ValueError("indices must be non-empty")

        # Each sample is a sliding window. For contiguous splits, the union of
        # windows covers a contiguous time span; we approximate the covered range
        # by [min_idx, max_idx + sequence_length).
        start = int(min(indices))
        end = int(max(indices)) + int(self.sequence_length)
        end = min(end, int(self.features.shape[0]))

        feature_tensor = torch.as_tensor(self.features[start:end], dtype=self.dtype)
        targets_tensor = torch.as_tensor(self.targets[indices], dtype=self.dtype)

        feat_mean, feat_std = _compute_feature_shift_scale(
            feature_tensor,
            scheme=self.feature_normalization,
            dtype=self.dtype,
        )
        tgt_mean = targets_tensor.mean()
        tgt_std = targets_tensor.std().clamp_min(1e-6)
        return feat_mean, feat_std, tgt_mean, tgt_std


class TensorSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple time-series dataset backed by precomputed tensors.

    Expected input shape is (N, seq_len, input_size); targets shape is (N,).
    Normalization stats are computed on the training split and applied at access
    time to avoid leakage.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype) -> None:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape (N, seq_len, input_size), got {tuple(x.shape)}"
            )
        if y.ndim != 1:
            raise ValueError(f"Expected y with shape (N,), got {tuple(y.shape)}")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        self.x = x.to(dtype=dtype)
        self.y = y.to(dtype=dtype)
        self.dtype = dtype

        self.feature_normalization = "standard"

        input_size = int(self.x.shape[-1])
        self.feature_mean = torch.zeros((1, input_size), dtype=self.dtype)
        self.feature_std = torch.ones((1, input_size), dtype=self.dtype)
        self.target_mean = torch.tensor(0.0, dtype=self.dtype)
        self.target_std = torch.tensor(1.0, dtype=self.dtype)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raw_seq = self.x[idx]
        seq = (raw_seq - self.feature_mean) / self.feature_std
        raw_target = self.y[idx]
        target = (raw_target - self.target_mean) / self.target_std
        return seq, target

    def set_normalization(
        self,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        *,
        feature_normalization: str = "standard",
    ) -> None:
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.feature_normalization = str(feature_normalization)

    def compute_stats_for_indices(
        self, indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        subset_x = self.x[indices]
        flattened = subset_x.reshape(-1, subset_x.shape[-1])
        feat_mean, feat_std = _compute_feature_shift_scale(
            flattened,
            scheme=self.feature_normalization,
            dtype=self.dtype,
        )
        subset_y = self.y[indices]
        tgt_mean = subset_y.mean()
        tgt_std = subset_y.std().clamp_min(1e-6)
        return feat_mean, feat_std, tgt_mean, tgt_std


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    dataset_cfg = cfg.get("dataset", {})
    dtype = dtype_torch(cfg.get("dtype")) or torch.float32

    feature_normalization = str(dataset_cfg.get("feature_normalization", "minmax_-1_1"))

    generator_cfg = dataset_cfg.get("generator")
    if generator_cfg is not None:
        from papers.shared.time_series.generators import data as ts_data

        generator_name = str(generator_cfg.get("name") or "").strip()
        if not generator_name:
            raise ValueError("dataset.generator.name must be provided")
        generator_params = generator_cfg.get("params") or {}
        if not isinstance(generator_params, dict):
            raise ValueError("dataset.generator.params must be an object")

        sequence_length = int(dataset_cfg.get("sequence_length", 8))
        max_samples = dataset_cfg.get("max_rows")
        if max_samples is not None:
            max_samples = int(max_samples)

        generator = ts_data.get(generator_name, **generator_params)
        x, y = generator.get_data(
            seq_len=sequence_length, target_dim=1, max_samples=max_samples
        )
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        feature_dim = generator_cfg.get("feature_dim")
        if feature_dim is None:
            feature_dim = dataset_cfg.get("feature_dim")
        if feature_dim is not None:
            feature_dim = int(feature_dim)
            if feature_dim <= 0:
                raise ValueError("feature_dim must be positive")
            if x.shape[-1] == 1 and feature_dim != 1:
                padded = torch.zeros(
                    x.shape[0],
                    x.shape[1],
                    feature_dim,
                    dtype=x.dtype,
                    device=x.device,
                )
                padded[:, :, 0] = x[:, :, 0]
                x = padded
            elif x.shape[-1] != feature_dim:
                raise ValueError(
                    f"Generator produced input_size={x.shape[-1]} but feature_dim={feature_dim} was requested"
                )
        dataset = TensorSequenceDataset(x, y, dtype=dtype)
        dataset.feature_normalization = feature_normalization

        total_len = len(dataset)
        if total_len == 0:
            raise ValueError("Generated dataset is empty")

        train_ratio = float(dataset_cfg.get("train_ratio", 0.7))
        val_ratio = float(dataset_cfg.get("val_ratio", 0.15))
        train_cutoff = int(total_len * train_ratio)
        val_cutoff = int(total_len * (train_ratio + val_ratio))
        train_indices = list(range(0, train_cutoff))
        val_indices = list(range(train_cutoff, val_cutoff))
        test_indices = list(range(val_cutoff, total_len))

        feat_mean, feat_std, tgt_mean, tgt_std = dataset.compute_stats_for_indices(
            train_indices
        )
        dataset.set_normalization(
            feat_mean,
            feat_std,
            tgt_mean,
            tgt_std,
            feature_normalization=feature_normalization,
        )

        train_ds = torch.utils.data.Subset(dataset, train_indices)
        val_ds = torch.utils.data.Subset(dataset, val_indices)
        test_ds = torch.utils.data.Subset(dataset, test_indices)

        batch_size = int(dataset_cfg.get("batch_size", 16))
        shuffle = bool(dataset_cfg.get("shuffle", True))
        sample_input, _ = dataset[0]

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        metadata = {
            "input_size": sample_input.shape[-1],
            "sequence_length": sequence_length,
            "prediction_horizon": 1,
            "feature_normalization": feature_normalization,
            "feature_mean": dataset.feature_mean.squeeze().tolist(),
            "feature_std": dataset.feature_std.squeeze().tolist(),
            "target_mean": float(dataset.target_mean),
            "target_std": float(dataset.target_std),
            "target_abs_mean": float(torch.mean(torch.abs(dataset.y))),
            "dataset_path": f"generator:{generator_name}",
            "generator": {"name": generator_name, "params": generator_params},
            "splits": {
                "train": len(train_ds),
                "val": len(val_ds),
                "test": len(test_ds),
            },
        }
        return train_loader, val_loader, test_loader, metadata

    csv_path = resolve_dataset_path(dataset_cfg, cfg=cfg)
    raw_df = pd.read_csv(csv_path)
    max_rows = dataset_cfg.get("max_rows")
    if max_rows is not None:
        raw_df = raw_df.head(int(max_rows))
    feature_columns = dataset_cfg.get("feature_columns") or []
    target_column = dataset_cfg.get("target_column")
    if not target_column:
        raise ValueError("dataset.target_column must be provided")

    time_column = dataset_cfg.get("time_column")
    if not feature_columns:
        excluded = {target_column}
        if time_column:
            excluded.add(str(time_column))
        inferred = [col for col in raw_df.columns if col not in excluded]
        feature_columns = inferred or [target_column]

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
    dataset.feature_normalization = feature_normalization

    total_len = len(dataset)
    if total_len == 0:
        raise ValueError(
            "Dataset is too small for the requested sequence and horizon lengths"
        )

    train_ratio = float(dataset_cfg.get("train_ratio", 0.7))
    val_ratio = float(dataset_cfg.get("val_ratio", 0.15))
    train_cutoff = int(total_len * train_ratio)
    val_cutoff = int(total_len * (train_ratio + val_ratio))
    train_indices = list(range(0, train_cutoff))
    val_indices = list(range(train_cutoff, val_cutoff))

    feat_mean, feat_std, tgt_mean, tgt_std = dataset.compute_stats_for_indices(
        train_indices
    )
    dataset.set_normalization(
        feat_mean,
        feat_std,
        tgt_mean,
        tgt_std,
        feature_normalization=feature_normalization,
    )

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    test_indices = list(range(val_cutoff, total_len))
    test_ds = torch.utils.data.Subset(dataset, test_indices)

    batch_size = int(dataset_cfg.get("batch_size", 16))
    shuffle = bool(dataset_cfg.get("shuffle", True))

    sample_input, _ = dataset[0]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    metadata = {
        "input_size": sample_input.shape[-1],
        "sequence_length": sequence_length,
        "prediction_horizon": prediction_horizon,
        "feature_normalization": feature_normalization,
        "feature_mean": dataset.feature_mean.squeeze().tolist(),
        "feature_std": dataset.feature_std.squeeze().tolist(),
        "target_mean": float(dataset.target_mean),
        "target_std": float(dataset.target_std),
        "target_abs_mean": float(
            torch.mean(torch.abs(torch.as_tensor(dataset.targets)))
        ),
        "dataset_path": str(csv_path),
        "splits": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
    }
    return train_loader, val_loader, test_loader, metadata


__all__ = [
    "TensorSequenceDataset",
    "WeatherSequenceDataset",
    "build_dataloaders",
    "resolve_dataset_path",
]
