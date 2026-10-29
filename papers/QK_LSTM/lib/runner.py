"""Runtime entry point for the QK-LSTM reproduction.

Artifacts written to ``run_dir`` (the reproduction's public contract):

- ``run.log``: configured by the shared runtime; lifecycle events follow the
  experiment-logging contract (RUN_STARTED, DATASET_READY, TRAINING_STARTED,
  TRAIN_EPOCH_COMPLETED, EVALUATION_*, METRICS_WRITTEN, RUN_COMPLETED/FAILED).
- ``run_status.json``: structured run state (RUNNING -> COMPLETED | FAILED).
- ``config_snapshot.json``: resolved configuration (written by the runtime, or
  here when the runner is called directly).
- ``metrics.json``: RMSE/MAE/MAPE/R^2 per split, in AQI units, parameter count
  and training history.
- ``predictions.csv``: one row per window with date, observed and predicted AQI.
- ``model_state.pt``: final-epoch weights (trainable models only).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import (
    SPLITS,
    PreparedSeries,
    WindowDataset,
    build_dataloader,
    prepare_series,
)
from .experiment_logging import git_state, sha256, utc_now, write_json
from .metrics import regression_metrics
from .models import (
    CLOSED_FORM_MODEL_TYPES,
    KERNEL_MODEL_TYPES,
    TORCH_MODEL_TYPES,
    RidgeWindowRegressor,
    build_torch_model,
    count_trainable_parameters,
    sample_reference_inputs,
)
from .training import predict, train_model

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if hasattr(value, "label"):  # runtime_lib DtypeSpec
        return value.label
    return str(value)


def _initial_status(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    run_cfg = cfg.get("run", {})
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_dir.name,
        "sweep_id": run_cfg.get("sweep_id"),
        "candidate": run_cfg.get("candidate", cfg["model"]["type"]),
        "seed": cfg["seed"],
        "repetition": run_cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(_REPO_ROOT),
        "dataset": {
            "name": cfg["dataset"]["name"],
            "splits": None,
            "preprocessing": None,
            "subset": None,
        },
        "selection": {
            "name": "final_epoch",
            "direction": "minimize",
            "split": "validation",
            "checkpoint": "final_epoch (no checkpoint selection; validation only monitored)",
        },
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": None,
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }


def _split_metrics(
    series: PreparedSeries, predictions_scaled: np.ndarray, target_days: np.ndarray
) -> dict[str, Any]:
    predicted_aqi = series.to_aqi(predictions_scaled)
    observed = series.target_observed[target_days]
    true_aqi = series.target_aqi[target_days]
    return {
        # Primary: only days whose AQI was observed (LOG.md, Q10).
        "observed_targets": regression_metrics(
            true_aqi[observed], predicted_aqi[observed]
        ),
        # Sensitivity: interpolated targets included.
        "all_targets": regression_metrics(true_aqi, predicted_aqi),
    }


def _predictions_frame(
    series: PreparedSeries,
    split: str,
    predictions_scaled: np.ndarray,
    target_days: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split,
            "date": series.dates[target_days].astype(str),
            "target_day": target_days,
            "aqi_true": series.target_aqi[target_days],
            "aqi_pred": series.to_aqi(predictions_scaled),
            "target_observed": series.target_observed[target_days],
        }
    )


def _window_arrays(
    series: PreparedSeries, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = WindowDataset(series, split)
    ends = dataset.window_end
    offsets = np.arange(-series.sequence_length + 1, 1)
    windows = series.features[ends[:, None] + offsets[None, :]]
    target_days = ends + series.horizon
    return windows, series.target[target_days], target_days


def _fit_and_predict(
    cfg: dict[str, Any], series: PreparedSeries, run_dir: Path
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Train (or fit) the configured model and predict every split."""
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]
    splits = [split for split in SPLITS if series.window_end[split].size > 0]
    extra: dict[str, Any] = {"history": [], "train_seconds": 0.0}

    if model_type == "persistence":
        # y_hat(t + h) = AQI(t): the last input day's (interpolated) AQI.
        extra["trainable_parameters"] = 0
        outputs = {}
        for split in splits:
            ends = series.window_end[split]
            outputs[split] = (series.target[ends], ends + series.horizon)
        return outputs, extra

    if model_type == "ridge":
        windows, targets, _ = _window_arrays(series, "train")
        logger.info(
            "TRAINING_STARTED | model=ridge | alpha=%s | train_windows=%d",
            model_cfg["params"]["alpha"],
            len(targets),
        )
        started = time.perf_counter()
        ridge = RidgeWindowRegressor(alpha=model_cfg["params"]["alpha"]).fit(
            windows, targets
        )
        extra["train_seconds"] = time.perf_counter() - started
        logger.info(
            "TRAIN_FIT_COMPLETED | model=ridge | parameters=%d", ridge.n_parameters
        )
        extra["trainable_parameters"] = ridge.n_parameters
        outputs = {}
        for split in splits:
            windows, _, target_days = _window_arrays(series, split)
            outputs[split] = (ridge.predict(windows), target_days)
        return outputs, extra

    if model_type not in TORCH_MODEL_TYPES:
        raise ValueError(
            f"Unknown model type {model_type!r}; expected {TORCH_MODEL_TYPES + CLOSED_FORM_MODEL_TYPES}"
        )

    device = torch.device(cfg["device"])
    # Each run is tiny (4 qubits, hidden 16); one thread avoids oversubscribing
    # the CPU when several seeds run in parallel.
    torch.set_num_threads(int(cfg["num_threads"]))
    logger.info("TORCH_THREADS | num_threads=%d", torch.get_num_threads())
    reference_inputs = None
    if model_type in KERNEL_MODEL_TYPES:
        train_days = np.unique(series.window_end["train"])
        reference_inputs = sample_reference_inputs(
            series.features[train_days],
            int(model_cfg["params"]["n_references"]),
            seed=int(cfg["seed"]),
        )
    model = build_torch_model(model_cfg, series.features.shape[1], reference_inputs).to(
        device
    )
    extra["trainable_parameters"] = count_trainable_parameters(model)
    photonic_kernel = getattr(getattr(model, "cell", None), "photonic_kernel", None)
    if photonic_kernel is not None:
        extra["photonic"] = photonic_kernel.describe()
        logger.info(
            "PHOTONIC_CONFIG | %s", json.dumps(extra["photonic"], sort_keys=True)
        )
    logger.info(
        "MODEL_BUILT | type=%s | params=%s | trainable_parameters=%d",
        model_type,
        json.dumps(model_cfg["params"], sort_keys=True),
        extra["trainable_parameters"],
    )

    train_loader = build_dataloader(cfg, "train", series)
    validation_loader = (
        build_dataloader(cfg, "validation", series) if "validation" in splits else None
    )

    def validation_rmse(
        predictions_scaled: np.ndarray, target_days: np.ndarray
    ) -> float:
        observed = series.target_observed[target_days]
        errors = (
            series.to_aqi(predictions_scaled)[observed]
            - series.target_aqi[target_days][observed]
        )
        return float(np.sqrt(np.mean(errors**2)))

    started = time.perf_counter()
    extra["history"] = train_model(
        model,
        train_loader,
        cfg["training"],
        device,
        validation_loader=validation_loader,
        validation_rmse_fn=validation_rmse,
    )
    extra["train_seconds"] = time.perf_counter() - started
    model_path = run_dir / "model_state.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(
        "CHECKPOINT_WRITTEN | path=%s | checkpoint=final_epoch", model_path.name
    )

    outputs = {}
    for split in splits:
        # Unshuffled loaders over every window of the split, for evaluation.
        loader = torch.utils.data.DataLoader(
            WindowDataset(series, split), batch_size=256, shuffle=False
        )
        outputs[split] = predict(model, loader, device)
    return outputs, extra


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run one configured experiment end to end.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (``dataset``, ``model``, ``training``, ``seed``,
        ``device``).
    run_dir : pathlib.Path
        Output directory created by the shared runtime.

    Returns
    -------
    dict
        The metrics written to ``metrics.json``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "run_status.json"
    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(cfg, indent=2, default=_json_default), encoding="utf-8"
        )
    status = _initial_status(cfg, run_dir)
    status["config_sha256"] = sha256(config_path)
    write_json(status_path, status)
    logger.info(
        "RUN_STARTED | run_id=%s | seed=%s | model=%s",
        status["run_id"],
        cfg["seed"],
        cfg["model"]["type"],
    )
    try:
        series = prepare_series(cfg)
        status["dataset"].update(
            {
                "splits": series.info["splits"],
                "preprocessing": series.info["preprocessing"],
                "subset": series.info["subset"],
                "file_sha256": series.info["file_sha256"],
            }
        )
        write_json(status_path, status)
        logger.info(
            "DATASET_READY | %s | splits=%s",
            series.info["subset"],
            json.dumps(series.info["splits"]),
        )

        outputs, extra = _fit_and_predict(cfg, series, run_dir)

        split_metrics, frames = {}, []
        for split, (predictions_scaled, target_days) in outputs.items():
            logger.info(
                "EVALUATION_STARTED | split=%s | windows=%d", split, len(target_days)
            )
            split_metrics[split] = _split_metrics(
                series, predictions_scaled, target_days
            )
            scored = split_metrics[split]["observed_targets"]
            logger.info(
                "EVALUATION_COMPLETED | split=%s | rmse=%.4f | mae=%.4f | mape=%.4f | r2=%.4f | n=%d",
                split,
                scored["rmse"],
                scored["mae"],
                scored["mape"],
                scored["r2"],
                scored["n"],
            )
            frames.append(
                _predictions_frame(series, split, predictions_scaled, target_days)
            )

        predictions_path = run_dir / "predictions.csv"
        pd.concat(frames, ignore_index=True).to_csv(predictions_path, index=False)
        logger.info("ARTIFACT_WRITTEN | path=%s", predictions_path.name)

        metrics = {
            "model_type": cfg["model"]["type"],
            "model_params": cfg["model"].get("params", {}),
            "trainable_parameters": extra["trainable_parameters"],
            "photonic": extra.get("photonic"),
            "seed": cfg["seed"],
            "metric_units": "AQI",
            "primary_scoring": "observed_targets",
            "splits": split_metrics,
            "history": extra["history"],
            "train_seconds": extra["train_seconds"],
            "target_scaling": {"min": series.target_min, "max": series.target_max},
        }
        metrics_path = run_dir / "metrics.json"
        write_json(metrics_path, metrics)
        logger.info("METRICS_WRITTEN | path=%s", metrics_path.name)

        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(metrics_path),
        )
        write_json(status_path, status)
        logger.info("RUN_COMPLETED | run_id=%s", status["run_id"])
        return metrics
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise
