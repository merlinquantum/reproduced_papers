"""Runtime entry point: one run = one task configuration (see ``configs/``).

Artifacts written to ``run_dir`` besides the runtime's ``run.log`` and
``config_snapshot.json``:

- ``run_status.json``: structured state under the experiment-logging contract;
- ``metrics.json``: the task's test metrics and selections;
- task artifacts (``candidates.csv``, ``test_predictions.csv``, ...).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .experiment_logging import git_state, sha256, utc_now, write_json

logger = logging.getLogger(__name__)

STATUS_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]


def _torch_dtype(value) -> torch.dtype:
    """Resolve the runtime's dtype entry (string, ``torch.dtype`` or ``DtypeSpec``)."""
    if isinstance(value, torch.dtype):
        return value
    if hasattr(value, "torch") and isinstance(value.torch, torch.dtype):
        return value.torch
    if isinstance(value, str) and isinstance(getattr(torch, value, None), torch.dtype):
        return getattr(torch, value)
    raise ValueError(f"Config 'dtype' must name a torch floating dtype, got {value!r}")


def _task_function(task: str):
    if task == "directional":
        from .tasks.directional import run_directional

        return run_directional
    if task == "trading":
        from .tasks.trading import run_trading

        return run_trading
    if task == "volatility":
        from .tasks.volatility import run_volatility

        return run_volatility
    raise ValueError(
        f"Unknown task {task!r}; expected 'directional', 'trading' or 'volatility'"
    )


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run one configured experiment and write its evidence files.

    Parameters
    ----------
    cfg : dict
        Resolved config. Required top-level keys: ``task``, ``seed``,
        ``dtype``, ``device``, ``num_threads``, ``data_root`` and a section
        named after the task (``directional``, ``trading`` or ``volatility``)
        holding its ``dataset``, ``models`` and training settings.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Raises
    ------
    Exception
        Any failure is recorded in ``run_status.json`` and re-raised.
    """
    run_dir = Path(run_dir)
    status_path = run_dir / "run_status.json"
    config_path = run_dir / "config_snapshot.json"
    status = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": cfg.get("candidate"),
        "seed": cfg["seed"],
        "repetition": cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": {
            "name": None,
            "splits": None,
            "preprocessing": None,
            "subset": None,
        },
        "selection": {
            "name": None,
            "direction": "maximize",
            "split": None,
            "checkpoint": None,
        },
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(config_path),
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    write_json(status_path, status)

    def report_progress(current: int, total: int, unit: str) -> None:
        # Read by utils/sweep.py to show natural-unit progress of a running worker.
        status["progress"] = {"current": current, "total": total, "unit": unit}
        write_json(status_path, status)

    logger.info(
        "RUN_STARTED | run_id=%s | task=%s | seed=%s",
        status["run_id"],
        cfg["task"],
        cfg["seed"],
    )
    try:
        task_cfg = dict(cfg)
        task_cfg["torch_dtype"] = _torch_dtype(cfg["dtype"])
        torch.set_num_threads(int(cfg["num_threads"]))
        metrics = _task_function(cfg["task"])(
            task_cfg, cfg[cfg["task"]], run_dir, status, report_progress
        )
        write_json(run_dir / "metrics.json", metrics)
        status["metrics_sha256"] = sha256(run_dir / "metrics.json")
        logger.info("METRICS_WRITTEN | path=metrics.json")
        status.update(status="COMPLETED", completed_at=utc_now())
        write_json(status_path, status)
        logger.info("RUN_COMPLETED | run_id=%s", status["run_id"])
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise
