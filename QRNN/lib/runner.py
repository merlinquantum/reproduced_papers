from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from runtime_lib.dtypes import describe_dtype, dtype_torch

from .data import build_dataloaders
from .model import RNNRegressor
from .training import fit


def _build_model(cfg: dict, input_size: int) -> RNNRegressor:
    model_cfg = cfg.get("model", {})
    params = model_cfg.get("params", {})
    hidden_dim = int(params.get("hidden_dim", 64))
    layers = int(params.get("layers", 1))
    dropout = float(params.get("dropout", 0.0))
    model = RNNRegressor(input_size=input_size, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    return model


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    dtype_label = describe_dtype(cfg.get("dtype"))
    logger.info("Using dtype: %s", dtype_label)

    train_loader, val_loader, metadata = build_dataloaders(cfg)
    model = _build_model(cfg, metadata["input_size"])

    dtype = dtype_torch(cfg.get("dtype"))
    if dtype is not None:
        model = model.to(dtype=dtype)

    metrics = fit(model, train_loader, val_loader, cfg, run_dir)

    model_path = run_dir / "rnn_baseline.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model checkpoint to %s", model_path)

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved metadata to %s", metadata_path)

    done_marker = run_dir / "done.txt"
    done_marker.write_text("baseline training complete\n", encoding="utf-8")
    logger.info("Wrote completion marker to %s", done_marker)


__all__ = ["train_and_evaluate"]
