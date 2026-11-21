"""Runtime entry point for the photonic_trainability reproduction.

``train_and_evaluate`` dispatches on ``cfg["experiment"]`` to one experiment
module. Each module exposes ``describe_dataset(cfg)`` and ``run(cfg, run_dir)``;
``run`` logs ``DATASET_READY`` once, its evaluations (and training iterations,
when it trains), and returns a JSON-serialisable metrics record.

Artifacts written to the run directory (the public reproduction contract):

* ``run.log`` - lifecycle, progress and evaluation events;
* ``run_status.json`` - structured status following the experiment-logging
  contract (schema version 1);
* ``config_snapshot.json`` - written by the shared runtime, or here when the
  runner is called directly;
* ``metrics.json`` - the experiment's metrics record;
* any figures or tables the experiment adds (listed in ``metrics.json``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import (
    exp_estimator_scaling,
    exp_first_moment,
    exp_merlin_variance,
    exp_network_polynomial,
    exp_output_probability,
    exp_pnp_variances,
    exp_sample_complexity,
    exp_training,
    exp_validate_moments,
    exp_variance_scaling,
)
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)

logger = logging.getLogger(__name__)

EXPERIMENTS = {
    "validate_moments": exp_validate_moments,
    "first_moment": exp_first_moment,
    "variance_scaling": exp_variance_scaling,
    "output_probability": exp_output_probability,
    "pnp_variances": exp_pnp_variances,
    "training": exp_training,
    "sample_complexity": exp_sample_complexity,
    "network_polynomial": exp_network_polynomial,
    "merlin_variance": exp_merlin_variance,
    "training_hardware": exp_training,
    "estimator_scaling": exp_estimator_scaling,
}

PROJECT_DIR = Path(__file__).resolve().parents[1]

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def _selection(cfg: dict[str, Any]) -> dict[str, Any]:
    """Model-selection record; only training experiments select anything."""
    return cfg.get(
        "selection",
        {
            "name": "none (analytic or Monte Carlo evaluation, no model selection)",
            "direction": "minimize",
            "split": "n/a",
            "checkpoint": "n/a",
        },
    )


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run one configured experiment and write its artifacts to ``run_dir``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration after defaults, overlays and CLI flags. Must
        contain ``experiment`` (a key of :data:`EXPERIMENTS`) and ``seed``.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Raises
    ------
    KeyError
        If ``experiment`` or ``seed`` is missing or the experiment is unknown.
    """
    run_dir = Path(run_dir)
    level_name = str(cfg.get("logging", {}).get("level", "info")).lower()
    configure_logging(run_dir / "run.log", level=_LOG_LEVELS[level_name])

    experiment_name = cfg["experiment"]
    if experiment_name not in EXPERIMENTS:
        raise KeyError(
            f"Unknown experiment {experiment_name!r}; expected one of {sorted(EXPERIMENTS)}"
        )
    experiment = EXPERIMENTS[experiment_name]

    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    status: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": cfg.get("candidate"),
        "seed": int(cfg["seed"]),
        "repetition": cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(PROJECT_DIR),
        "dataset": experiment.describe_dataset(cfg),
        "selection": _selection(cfg),
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(config_path),
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    status_path = run_dir / "run_status.json"
    write_json(status_path, status)
    logger.info(
        "RUN_STARTED | run_id=%s | experiment=%s | seed=%s | device=%s",
        status["run_id"],
        experiment_name,
        status["seed"],
        cfg.get("device"),
    )
    try:
        metrics = experiment.run(cfg, run_dir)
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
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise
