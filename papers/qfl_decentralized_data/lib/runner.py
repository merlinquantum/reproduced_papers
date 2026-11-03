"""Runtime entry point: train and evaluate one QFL experiment.

One call runs one experiment (E1/E2 MNIST, E3 QAOA or E4 VQE, federated or
centralised, quantum or classical) for one seed, and writes ``run.log``,
``run_status.json``, ``config_snapshot.json`` (the runtime writes this one) and
``metrics.json`` into ``run_dir``.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import torch

from runtime_lib.dtypes import coerce_dtype_spec

from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .federated import run_training
from .task import Task

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_KEYS = {
    "dataset": ["name"],
    "model": ["name"],
    "federation": [
        "mode",
        "rounds",
        "local_epochs",
        "n_clients",
        "clients_per_round",
        "server_lr",
        "partition",
    ],
    "training": ["optimizer", "lr", "batch_size"],
}
DATASET_PREPROCESSING = {
    "mnist": "28x28 -> 4x4 area interpolation, pixels/255 in [0, 1]; classes relabelled "
    "0..C-1; Dirichlet non-i.i.d. unbalanced client partition",
    "random_graphs": "random 3-regular graphs, MaxCut cost diagonal precomputed",
    "h2": "H2 STO-3G 2-qubit Hamiltonian coefficients from PySCF integrals",
}


def _build_task(cfg: dict[str, Any], dtype: torch.dtype, device: torch.device) -> Task:
    name = cfg["dataset"]["name"]
    if name == "mnist":
        from .mnist_task import build_mnist_task

        return build_mnist_task(cfg, dtype, device)
    if name == "random_graphs":
        from .qaoa import build_qaoa_task

        return build_qaoa_task(cfg, dtype, device)
    if name == "h2":
        from .vqe import build_vqe_task

        return build_vqe_task(cfg, dtype, device)
    raise ValueError(f"Unknown dataset.name {name!r}")


def _validate_config(cfg: dict[str, Any]) -> None:
    for section, keys in REQUIRED_KEYS.items():
        for key in keys:
            if key not in cfg.get(section, {}):
                raise KeyError(f"Missing required config key {section}.{key}")
    if cfg["training"]["optimizer"] != "adam":
        raise ValueError("Only Adam is implemented (the paper's optimiser)")
    if cfg["dataset"]["name"] not in DATASET_PREPROCESSING:
        raise ValueError(f"Unknown dataset.name {cfg['dataset']['name']!r}")


def _torch_dtype(cfg: dict[str, Any]) -> torch.dtype:
    spec = coerce_dtype_spec(cfg.get("dtype") or "float32")
    if spec is None or spec.torch is None:
        raise ValueError(f"Cannot resolve dtype from {cfg.get('dtype')!r}")
    return spec.torch


def _initial_status(cfg: dict[str, Any], run_id: str, run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config_snapshot.json"
    dataset_name = cfg.get("dataset", {}).get("name")
    has_validation = float(cfg.get("dataset", {}).get("validation_fraction", 0)) > 0
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": {
            "experiment": cfg.get("experiment"),
            "mode": cfg.get("federation", {}).get("mode"),
            "n_clients": cfg.get("federation", {}).get("n_clients"),
            **cfg.get("candidate_overrides", {}),
        },
        "seed": cfg.get("seed"),
        "repetition": cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": {
            "name": dataset_name,
            "splits": None,
            "preprocessing": DATASET_PREPROCESSING.get(dataset_name),
            "subset": {
                "classes": cfg.get("dataset", {}).get("classes"),
                "validation_fraction": cfg.get("dataset", {}).get(
                    "validation_fraction"
                ),
            },
        },
        "selection": {
            "name": "final_round_validation_metric"
            if has_validation
            else "final_round_test_metric",
            "direction": cfg.get("metric_direction", "maximize"),
            "split": "validation" if has_validation else "test",
            "checkpoint": "last",
        },
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(config_path) if config_path.exists() else None,
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """Run one QFL experiment and write its evidence into ``run_dir``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (see ``configs/defaults.json``).
    run_dir : pathlib.Path
        Output directory created by the shared runtime.

    Raises
    ------
    KeyError
        If a required configuration key is missing.
    ValueError
        If the dataset, optimiser or federation mode is not supported.
    """
    run_dir = Path(run_dir)
    configure_logging(
        run_dir / "run.log",
        level=logging.DEBUG
        if cfg.get("logging", {}).get("level") == "debug"
        else logging.INFO,
    )
    run_id = cfg.get("run_id") or f"{run_dir.name}-{uuid.uuid4().hex[:8]}"
    status = _initial_status(cfg, run_id, run_dir)
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "RUN_STARTED | run_id=%s | seed=%s | experiment=%s",
        run_id,
        cfg.get("seed"),
        cfg.get("experiment"),
    )
    try:
        _validate_config(cfg)
        device = torch.device(cfg.get("device", "cpu"))
        dtype = _torch_dtype(cfg)
        torch.manual_seed(int(cfg["seed"]))

        task = _build_task(cfg, dtype, device)
        status["dataset"]["splits"] = task.summary
        write_json(run_dir / "run_status.json", status)
        logger.info("DATASET_READY | %s", task.summary)

        parameter_count = sum(p.numel() for p in task.model.parameters())
        logger.info(
            "MODEL_READY | model=%s | trainable_parameters=%d | clients=%d",
            cfg["model"]["name"],
            parameter_count,
            len(task.client_data),
        )

        federation, training = cfg["federation"], cfg["training"]
        rounds = int(federation["rounds"])
        logger.info(
            "TRAINING_STARTED | mode=%s | rounds=%d | optimizer=adam | lr=%g | batch_size=%d",
            federation["mode"],
            rounds,
            training["lr"],
            training["batch_size"],
        )

        def log_round(entry: dict[str, Any]) -> None:
            logger.info(
                "TRAIN_ROUND_COMPLETED | round=%d/%d | loss=%.6f | lr=%g",
                entry["round"],
                rounds,
                entry["train_loss"],
                training["lr"],
            )
            logger.info("EVALUATION_STARTED | split=test | round=%d", entry["round"])
            logger.info(
                "EVALUATION_COMPLETED | split=test | round=%d | %s=%.6f",
                entry["round"],
                task.primary_metric,
                entry[task.primary_metric],
            )

        started = time.perf_counter()
        history = run_training(cfg, task, device, log_round)
        wall_seconds = time.perf_counter() - started

        with torch.no_grad():
            report = task.final_report(task.model)
        metrics = {
            "experiment": cfg.get("experiment"),
            "label": cfg.get("label"),
            "mode": federation["mode"],
            "n_clients": len(task.client_data),
            "primary_metric": task.primary_metric,
            "history": history,
            "final_round": history[-1]["round"],
            "final_primary_metric": history[-1][task.primary_metric],
            "final_report": report,
            "trainable_parameters": parameter_count,
            "wall_clock_seconds": wall_seconds,
        }
        write_json(run_dir / "metrics.json", metrics)
        logger.info(
            "METRICS_WRITTEN | path=metrics.json | %s=%.6f",
            task.primary_metric,
            metrics["final_primary_metric"],
        )
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(run_dir / "metrics.json"),
        )
        write_json(run_dir / "run_status.json", status)
        logger.info(
            "RUN_COMPLETED | run_id=%s | wall_clock_seconds=%.1f", run_id, wall_seconds
        )
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.exception("RUN_FAILED | run_id=%s", run_id)
        raise
