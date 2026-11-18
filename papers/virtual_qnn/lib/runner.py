"""Shared-runtime entry point for the virtual QNN reproduction.

One call of :func:`train_and_evaluate` performs exactly one training (or one
SDP solve) at one configuration point and writes, in ``run_dir``::

    run.log               lifecycle, per-iteration progress, evaluations
    run_status.json       structured status (experiment-logging contract)
    config_snapshot.json  resolved configuration (written by the runtime)
    metrics.json          final metrics of the run
    losses.csv            per-iteration history (training tasks)
    model_state.pt, virtual_state.npz, dataset.npz   task-specific artifacts

Budget, noise-level, bond-length and seed grids are produced by running this
entry point once per point (see ``utils/``), never by looping inside a run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .tasks import TASKS, describe_dataset, describe_selection

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _candidate(cfg: dict) -> dict:
    """Configuration values that identify this run inside a sweep."""
    task = cfg["experiment"]["task"]
    explicit = cfg["experiment"].get("candidate")
    if explicit:
        return dict(explicit)
    section = {
        "vqem": "vqem",
        "vqc": "vqc",
        "vqc_merlin": "vqc_merlin",
        "vvqe": "vvqe",
        "classical_classifier": "classical",
        "min_overhead_sdp": "sdp",
        "equal_shot_eval": "equal_shot",
    }[task]
    return {
        "task": task,
        **{k: v for k, v in cfg[section].items() if not isinstance(v, dict)},
    }


def _ensure_config_snapshot(cfg: dict, run_dir: Path) -> Path:
    """Return the runtime's snapshot, writing one when called outside the runtime."""
    path = run_dir / "config_snapshot.json"
    if not path.exists():
        path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    return path


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run one configured virtual QNN experiment.

    Parameters
    ----------
    cfg : dict
        Resolved experiment configuration. ``experiment.task`` selects one of
        ``min_overhead_sdp``, ``vqem``, ``vqc``, ``classical_classifier`` or
        ``vvqe``; ``seed`` seeds the parameter initialization.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Raises
    ------
    ValueError
        If ``experiment.task`` is not an implemented task.
    """
    run_dir = Path(run_dir)
    task = cfg["experiment"]["task"]
    if task not in TASKS:
        raise ValueError(
            f"Unsupported experiment.task={task!r}; implemented tasks: {tuple(TASKS)}"
        )
    level = getattr(logging, str(cfg["logging"]["level"]).upper())
    configure_logging(run_dir / "run.log", level=level)
    config_path = _ensure_config_snapshot(cfg, run_dir)
    status_path = run_dir / "run_status.json"
    run_id = cfg["experiment"].get("run_id") or run_dir.name
    status = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "sweep_id": cfg["experiment"].get("sweep_id"),
        "candidate": _candidate(cfg),
        "seed": cfg["seed"],
        "repetition": cfg["experiment"].get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(_REPO_ROOT),
        "dataset": describe_dataset(cfg),
        "selection": describe_selection(cfg),
        "log_path": "run.log",
        "config_path": config_path.name,
        "config_sha256": sha256(config_path),
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    write_json(status_path, status)
    logger.info(
        "RUN_STARTED | run_id=%s | task=%s | experiment=%s | seed=%s | torch=%s",
        run_id,
        task,
        cfg["experiment"].get("id"),
        cfg["seed"],
        torch.__version__,
    )
    try:
        generator = torch.Generator().manual_seed(int(cfg["seed"]))
        metrics = TASKS[task](cfg, run_dir, generator)
        metrics_path = run_dir / "metrics.json"
        write_json(metrics_path, metrics)
        logger.info("METRICS_WRITTEN | path=%s", metrics_path.name)
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(metrics_path),
        )
        write_json(status_path, status)
        logger.info("RUN_COMPLETED | run_id=%s", run_id)
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("RUN_FAILED | run_id=%s", run_id)
        raise
