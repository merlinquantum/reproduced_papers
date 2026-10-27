"""Runtime entry point for the trainable-frequency QNN reproduction.

``cfg["task"]`` selects the experiment family:

- ``toy_regression``: cosine-series fits of paper Sec. V (E1-E5);
- ``navier_stokes``: DQC solve of the cylinder wake of paper Sec. VI (E6-E7).

Every run writes the evidence files of the logging contract next to the task
artifacts: ``run.log``, ``run_status.json``, ``config_snapshot.json`` and
``metrics.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from runtime_lib.dtypes import DtypeSpec

from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .navier_stokes import run_navier_stokes
from .toy_task import run_toy_regression

logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TASKS = {"toy_regression": run_toy_regression, "navier_stokes": run_navier_stokes}
STATUS_SCHEMA_VERSION = 1


def _log_file_attached(log_path: Path) -> bool:
    target = str(log_path.resolve())
    return any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == target
        for handler in logging.getLogger().handlers
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, DtypeSpec):
        return value.label
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _dataset_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    if dataset_cfg["name"] == "cosine_series":
        return {
            "name": "cosine_series",
            "splits": ["train", "dense"],
            "preprocessing": "none",
            "subset": {
                "frequencies": dataset_cfg["frequencies"],
                "domain_pi": dataset_cfg["domain_pi"],
            },
        }
    return {
        "name": f"cylinder_wake:{dataset_cfg['file']}",
        "splits": ["collocation", "data", "evaluation"],
        "preprocessing": f"coordinate_transform={dataset_cfg['coordinate_transform']}",
        "subset": {
            "time_window": dataset_cfg["time_window"],
            "data_stride": dataset_cfg["data_stride"],
        },
    }


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run one configured experiment and write its artifacts to ``run_dir``.

    Parameters
    ----------
    cfg : dict
        Resolved experiment configuration.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Returns
    -------
    dict
        The metrics written to ``metrics.json``.

    Raises
    ------
    ValueError
        If ``cfg["task"]`` is not a known task.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    if not _log_file_attached(log_path):
        configure_logging(log_path)

    task = cfg["task"]
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASKS)}.")

    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        # The shared runtime writes this file; direct calls (tests) need it too.
        config_path.write_text(json.dumps(_json_safe(cfg), indent=2), encoding="utf-8")

    sweep = cfg.get("sweep") or {}
    status_path = run_dir / "run_status.json"
    status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "sweep_id": sweep.get("sweep_id"),
        "candidate": sweep.get("candidate", {}),
        "seed": cfg["seed"],
        "repetition": sweep.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPOSITORY_ROOT),
        "dataset": _dataset_identity(cfg),
        "selection": {
            "name": "final_training_loss",
            "direction": "minimize",
            "split": "train",
            "checkpoint": "final_iteration",
        },
        "log_path": log_path.name,
        "config_path": config_path.name,
        "config_sha256": sha256(config_path),
        "metrics_path": None,
        "metrics_sha256": None,
        "error": None,
    }
    write_json(status_path, status)
    logger.info(
        "RUN_STARTED | run_id=%s | task=%s | seed=%s", run_dir.name, task, cfg["seed"]
    )

    try:
        metrics = TASKS[task](cfg, run_dir)
        metrics_path = run_dir / "metrics.json"
        write_json(metrics_path, metrics)
        logger.info("METRICS_WRITTEN | path=%s", metrics_path.name)
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_path=metrics_path.name,
            metrics_sha256=sha256(metrics_path),
        )
        write_json(status_path, status)
        logger.info("RUN_COMPLETED | run_id=%s", run_dir.name)
        return metrics
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("RUN_FAILED | run_id=%s", run_dir.name)
        raise
