"""QuLTSF reproduction entry point for the shared runtime.

One run trains one model (``cfg["model"]["name"]``) on one (L, T) setting with
one seed and writes, inside ``run_dir``:

- ``run.log`` and ``config_snapshot.json`` (written by the shared runtime);
- ``run_status.json``: run identity, dataset identity, selection rule, Git
  state and terminal status (logging contract of the reproduction workflow);
- ``metrics.json``: parameter count, training history, best epoch, and
  validation/test MSE and MAE on standardised data. Test metrics are given
  for every window and for the upstream ``drop_last`` subset;
- ``test_errors.npz``: per-window test squared and absolute errors, in time
  order, for Diebold-Mariano tests between models;
- ``best_model.pt``: best-validation ``state_dict`` (trained models only).
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from .data import build_dataloader, load_series, model_dtype
from .experiment_logging import git_state, sha256, utc_now, write_json
from .models import build_model, count_trainable_parameters
from .training import fit, per_window_errors, summarise_errors, validation_loss

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1


def _run_id(cfg: dict) -> str:
    # Inside a sweep the coordinator names each planned run
    # "<candidate>_seed<seed>"; sweep validation requires the same ID here.
    candidate = cfg.get("sweep", {}).get("candidate")
    if candidate:
        return f"{candidate}_seed{cfg['seed']}"
    dataset = cfg["dataset"]
    return (
        f"{cfg['model']['name']}-L{dataset['sequence_length']}-"
        f"T{dataset['prediction_horizon']}-seed{cfg['seed']}-{uuid.uuid4().hex[:8]}"
    )


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Train and evaluate one forecasting model.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (defaults, config overlay and CLI flags).
    run_dir : pathlib.Path
        Timestamped run directory created by the shared runtime.

    Raises
    ------
    Exception
        Any failure is logged as ``RUN_FAILED``, recorded in
        ``run_status.json`` and re-raised.
    """
    run_dir = Path(run_dir)
    status_path = run_dir / "run_status.json"
    sweep_cfg = cfg.get("sweep", {})
    status = {
        "schema_version": SCHEMA_VERSION,
        "run_id": _run_id(cfg),
        "sweep_id": sweep_cfg.get("sweep_id"),
        "candidate": sweep_cfg.get("candidate"),
        "seed": cfg["seed"],
        "repetition": sweep_cfg.get("repetition"),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": {
            "name": cfg["dataset"].get("name"),
            "splits": None,
            "preprocessing": (
                "standard scaling fitted on train rows (ddof=0); features_mode="
                f"{cfg['dataset']['features_mode']}; L={cfg['dataset']['sequence_length']}, "
                f"T={cfg['dataset']['prediction_horizon']}"
            ),
            "subset": cfg["training"].get("max_batches_per_epoch"),
        },
        "selection": {
            "name": "val_mse",
            "direction": "minimize",
            "split": "val",
            "checkpoint": "best validation epoch (early stopping)",
        },
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(run_dir / "config_snapshot.json"),
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    write_json(status_path, status)
    logger.info(
        "RUN_STARTED | run_id=%s | model=%s | seed=%s | device=%s | dtype=%s",
        status["run_id"],
        cfg["model"]["name"],
        cfg["seed"],
        cfg["device"],
        cfg["dtype"],
    )
    try:
        started = time.perf_counter()
        device = torch.device(cfg["device"])
        dtype = model_dtype(cfg)

        series = load_series(cfg)
        loaders = {
            split: build_dataloader(cfg, split, series)
            for split in ("train", "val", "test")
        }
        status["dataset"]["splits"] = series.source["splits"]
        status["dataset"]["identity"] = {
            key: series.source[key] for key in ("name", "path", "sha256", "rows")
        }
        write_json(status_path, status)
        logger.info(
            "DATASET_READY | name=%s | sha256=%s | channels=%d | windows train/val/test=%d/%d/%d",
            series.source["name"],
            series.source["sha256"],
            len(series.columns),
            len(loaders["train"].dataset),
            len(loaders["val"].dataset),
            len(loaders["test"].dataset),
        )

        model = build_model(cfg, dtype).to(device)
        n_parameters = count_trainable_parameters(model)
        logger.info(
            "MODEL_READY | name=%s | trainable_parameters=%d",
            cfg["model"]["name"],
            n_parameters,
        )

        features_mode = str(cfg["dataset"]["features_mode"])
        if n_parameters > 0:
            training = fit(model, loaders["train"], loaders["val"], cfg, device)
            checkpoint_path = run_dir / "best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("CHECKPOINT_WRITTEN | path=%s", checkpoint_path.name)
        else:
            training = {
                "history": [],
                "epochs_run": 0,
                "stop_reason": "no_trainable_parameters",
            }
            logger.info("TRAINING_SKIPPED | reason=no_trainable_parameters")

        logger.info("EVALUATION_STARTED | split=val")
        val_mse = validation_loss(model, loaders["val"], device, features_mode)
        logger.info("EVALUATION_COMPLETED | split=val | mse=%.6f", val_mse)

        logger.info("EVALUATION_STARTED | split=test")
        test_errors = per_window_errors(model, loaders["test"], device, features_mode)
        test = summarise_errors(test_errors, int(cfg["dataset"]["batch_size"]))
        logger.info(
            "EVALUATION_COMPLETED | split=test | mse=%.6f | mae=%.6f | windows=%d | "
            "mse_drop_last=%.6f | mae_drop_last=%.6f",
            test["mse"],
            test["mae"],
            test["n_windows"],
            test["mse_drop_last"],
            test["mae_drop_last"],
        )
        errors_path = run_dir / "test_errors.npz"
        np.savez_compressed(
            errors_path,
            squared=test_errors["squared"],
            absolute=test_errors["absolute"],
        )
        logger.info("ARTIFACT_WRITTEN | path=%s", errors_path.name)

        metrics = {
            "schema_version": SCHEMA_VERSION,
            "run_id": status["run_id"],
            "model": cfg["model"]["name"],
            "model_params": cfg["model"].get("params", {}),
            "seq_len": int(cfg["dataset"]["sequence_length"]),
            "pred_len": int(cfg["dataset"]["prediction_horizon"]),
            "features_mode": features_mode,
            "seed": cfg["seed"],
            "device": cfg["device"],
            "dtype": str(cfg["dtype"]),
            "trainable_parameters": n_parameters,
            "metric_space": "standardised (train-fitted scaler)",
            "training": training,
            "val": {"mse": val_mse},
            "test": test,
            "test_errors_path": errors_path.name,
            "wall_clock_seconds": time.perf_counter() - started,
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
        logger.info(
            "RUN_COMPLETED | run_id=%s | test_mse=%.6f | test_mae=%.6f",
            status["run_id"],
            test["mse"],
            test["mae"],
        )
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise
