"""Shared-runtime entry point for RF-RQKS ablation experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .ablation import run_ablation
from .ablation_data import load_dct_dataset


def _require(config: dict, key: str):
    if key not in config:
        raise KeyError(f"Missing required RF-RQKS configuration key: {key}")
    return config[key]


def _resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is unavailable")
    return str(device)


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Load the DCT dataset and execute the five-stage ablation.

    Parameters
    ----------
    cfg : dict
        Resolved shared-runtime configuration.
    run_dir : pathlib.Path
        Timestamped output directory.

    Raises
    ------
    KeyError
        If a required configuration entry is missing.
    """
    logger = logging.getLogger(__name__)
    data_root = Path(_require(cfg, "data_root")) / "RF-RQKS"
    representation_root = data_root / str(_require(cfg, "representation_path"))
    cfg = dict(cfg)
    cfg["device"] = _resolve_device(str(_require(cfg, "device")))
    dataset = load_dct_dataset(
        representation_root=representation_root,
        validation_fraction=float(_require(cfg, "validation_fraction")),
        seed=int(_require(cfg, "seed")),
        standardization_batch_size=int(_require(cfg, "standardization_batch_size")),
    )
    logger.info(
        "Loaded %s: train=%d, validation=%d, test=%d, features=%d",
        representation_root,
        dataset.train_labels.size,
        dataset.validation_labels.size,
        dataset.test_labels.size,
        dataset.input_feature_count,
    )
    state = run_ablation(cfg, dataset, run_dir)
    summary = {
        "status": state["status"],
        "dataset": state["dataset"],
        "sampler": state["sampler"],
        "best_configuration": state["best_model"]["configuration"],
        "best_validation_metrics": state["best_model"]["metrics"],
        "stage_5": state["stages"]["stage_5"]["results"][0],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

