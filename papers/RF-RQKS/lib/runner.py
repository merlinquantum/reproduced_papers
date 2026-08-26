"""Shared-runtime entry point for RF-RQKS ablation experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .ablation import run_ablation, run_readout_comparison
from .ablation_data import DatasetSplits, load_dct_dataset


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


def _build_synthetic_smoke_dataset(config: dict) -> DatasetSplits:
    """Build a deterministic feature dataset for file-free smoke tests.

    Parameters
    ----------
    config : dict
        Resolved configuration containing the synthetic dataset dimensions and
        seed.

    Returns
    -------
    DatasetSplits
        Small balanced dataset suitable for exercising the ablation pipeline.
    """
    feature_count = int(_require(config, "synthetic_feature_count"))
    train_sample_count = int(_require(config, "synthetic_train_samples"))
    validation_sample_count = int(_require(config, "synthetic_validation_samples"))
    test_sample_count = int(_require(config, "synthetic_test_samples"))
    if feature_count <= 0:
        raise ValueError("synthetic_feature_count must be positive")
    if any(
        sample_count <= 1
        or sample_count % 2
        for sample_count in (
            train_sample_count,
            validation_sample_count,
            test_sample_count,
        )
    ):
        raise ValueError("Synthetic sample counts must be positive even values greater than one")

    rng = np.random.default_rng(int(_require(config, "seed")))

    def make_split(sample_count: int) -> tuple[np.ndarray, np.ndarray]:
        labels = np.tile(np.asarray([0, 1], dtype=np.int64), sample_count // 2)
        features = rng.normal(size=(sample_count, feature_count)).astype(np.float32)
        features[labels == 1, 0] += 1.0
        return features, labels

    train_features, train_labels = make_split(train_sample_count)
    validation_features, validation_labels = make_split(validation_sample_count)
    test_features, test_labels = make_split(test_sample_count)
    development_features = np.concatenate((train_features, validation_features), axis=0)
    development_labels = np.concatenate((train_labels, validation_labels), axis=0)
    return DatasetSplits(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        test_features=test_features,
        test_labels=test_labels,
        development_features=development_features,
        development_labels=development_labels,
    )


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
    cfg = dict(cfg)
    cfg["device"] = _resolve_device(str(_require(cfg, "device")))
    data_source = str(_require(cfg, "data_source"))
    if data_source == "synthetic":
        dataset = _build_synthetic_smoke_dataset(cfg)
        dataset_description = "deterministic synthetic smoke dataset"
    elif data_source == "representation":
        data_root = Path(_require(cfg, "data_root")) / "RF-RQKS"
        representation_root = data_root / str(_require(cfg, "representation_path"))
        dataset = load_dct_dataset(
            representation_root=representation_root,
            validation_fraction=float(_require(cfg, "validation_fraction")),
            seed=int(_require(cfg, "seed")),
            standardization_batch_size=int(_require(cfg, "standardization_batch_size")),
        )
        dataset_description = str(representation_root)
    else:
        raise ValueError(f"Unsupported RF-RQKS data_source: {data_source}")
    logger.info(
        "Loaded %s: train=%d, validation=%d, test=%d, features=%d",
        dataset_description,
        dataset.train_labels.size,
        dataset.validation_labels.size,
        dataset.test_labels.size,
        dataset.input_feature_count,
    )
    experiment = str(cfg.get("experiment", "ablation"))
    if experiment == "readout_comparison":
        result = run_readout_comparison(cfg, dataset, run_dir)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "experiment": experiment,
                    "configuration": result["configuration"],
                    "best_qks_readout": result["best_qks_readout"],
                    "direct_readouts": result["direct_readouts"],
                    "qks_readouts": result["qks_readouts"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    if experiment != "ablation":
        raise ValueError(f"Unsupported RF-RQKS experiment: {experiment}")
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
