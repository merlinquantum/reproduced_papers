"""Shared-runtime entry point: lifecycle, provenance and dispatch by ``cfg["experiment"]``.

Every run writes, relative to ``run_dir``:

- ``run.log``: lifecycle, every optimizer step and every evaluation;
- ``run_status.json``: the experiment-logging contract's status record;
- ``config_snapshot.json``: the resolved configuration (written by the shared
  runtime; written here only when the runner is called directly);
- ``metrics.json``: the experiment's structured results;
- experiment artifacts named in ``metrics.json["artifacts"]``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from runtime_lib.dtypes import dtype_label, dtype_torch

from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .experiments import (
    run_coefficient_experiment,
    run_coefficient_reach_experiment,
    run_fit_experiment,
    run_photonic_spectrum_experiment,
    run_spectrum_experiment,
    run_universality_experiment,
)

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
STATUS_SCHEMA_VERSION = 1
SUPPORTED_DTYPES = (torch.float32, torch.float64)

_DATASET_IDENTITY = {
    "fit": {
        "name": "synthetic_fourier_targets (paper Figs. 3-4)",
        "splits": [
            "full grid; train = mini-batches with replacement, eval = full grid"
        ],
        "preprocessing": "none; input scaling is part of the model encoding",
        "subset": "full grid",
    },
    "coefficients": {
        "name": "none (random circuits, paper Fig. 5)",
        "splits": [],
        "preprocessing": "none",
        "subset": "none",
    },
    "spectrum": {
        "name": "none (random circuits and analytic spectra)",
        "splits": [],
        "preprocessing": "none",
        "subset": "none",
    },
    "photonic_spectrum": {
        "name": "none (random photonic circuits, Phase 4 MerLin adaptation)",
        "splits": [],
        "preprocessing": "none",
        "subset": "none",
    },
    "coefficient_reach": {
        "name": "none (coefficient maximisation, Phase 4 diagnostic)",
        "splits": [],
        "preprocessing": "none",
        "subset": "none",
    },
    "universality": {
        "name": "synthetic truncated Fourier series and sign(sin x)",
        "splits": [],
        "preprocessing": "none",
        "subset": "none",
    },
}

# No experiment selects models or hyperparameters: every case and seed is
# reported. The contract still needs a metric and direction, so each entry names
# the headline metric that metrics.json reports and the direction in which it is
# better.
_REPORTED_METRIC = {
    "fit": {
        "name": "full-grid MSE per case and seed, compared with B1 (reported, no selection)",
        "direction": "minimize",
        "split": "full grid (no held-out split in the paper)",
        "checkpoint": "final optimizer step",
    },
    "coefficients": {
        "name": "per-frequency coefficient statistics (reported, no selection)",
        "direction": "maximize",
        "split": "n/a (random circuits)",
        "checkpoint": "n/a (no training)",
    },
    "spectrum": {
        "name": "largest Fourier coefficient outside the predicted spectrum (reported, no selection)",
        "direction": "minimize",
        "split": "n/a (random circuits)",
        "checkpoint": "n/a (no training)",
    },
    "photonic_spectrum": {
        "name": "largest Fourier coefficient outside the predicted photonic spectrum (reported, no selection)",
        "direction": "minimize",
        "split": "n/a (random circuits)",
        "checkpoint": "n/a (no training)",
    },
    "coefficient_reach": {
        "name": "largest |c_k| reached over initialisations (reported, no selection)",
        "direction": "maximize",
        "split": "n/a (no data)",
        "checkpoint": "best value over the optimisation",
    },
    "universality": {
        "name": "reconstruction error of the App. C construction (reported, no selection)",
        "direction": "minimize",
        "split": "n/a (analytic targets)",
        "checkpoint": "n/a (no training)",
    },
}


def _json_safe(value: Any) -> Any:
    """Convert a resolved config (which may hold ``DtypeSpec`` objects) to plain JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and not hasattr(value, "_fields"):
        return [_json_safe(item) for item in value]
    if hasattr(value, "_fields") or isinstance(value, torch.dtype):
        return dtype_label(value)
    return value


def _resolve_dtype(cfg: dict) -> torch.dtype:
    dtype = dtype_torch(cfg["dtype"])
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be float32 or float64, got {cfg['dtype']!r}")
    return dtype


def _dispatch(
    cfg: dict, run_dir: Path, device: torch.device, dtype: torch.dtype
) -> dict:
    experiment = cfg["experiment"]
    seed = int(cfg["seed"])
    if experiment == "fit":
        return run_fit_experiment(cfg["fit"], run_dir, device, dtype)
    if experiment == "coefficients":
        return run_coefficient_experiment(
            cfg["coefficients"], run_dir, device, dtype, seed
        )
    if experiment == "spectrum":
        return run_spectrum_experiment(cfg["spectrum"], run_dir, device, dtype, seed)
    if experiment == "photonic_spectrum":
        return run_photonic_spectrum_experiment(
            cfg["photonic_spectrum"], run_dir, device, dtype, seed
        )
    if experiment == "coefficient_reach":
        return run_coefficient_reach_experiment(
            cfg["coefficient_reach"], run_dir, device, dtype
        )
    if experiment == "universality":
        return run_universality_experiment(cfg["universality"], run_dir, seed)
    raise ValueError(
        f"Unknown experiment {experiment!r}; expected one of {sorted(_DATASET_IDENTITY)}"
    )


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run the experiment selected by ``cfg["experiment"]`` and write its evidence.

    Parameters
    ----------
    cfg : dict
        Resolved configuration. Required top-level keys: ``experiment``,
        ``seed``, ``dtype``, ``device``, ``logging.level`` and the section named
        by ``experiment`` (``fit``, ``coefficients``, ``spectrum``,
        ``universality`` or ``photonic_spectrum``).
    run_dir : Path
        Timestamped run directory created by the shared runtime.

    Raises
    ------
    KeyError
        If a required configuration key is missing.
    ValueError
        If the experiment name or dtype is not supported.
    """
    run_dir = Path(run_dir)
    log_level = getattr(logging, str(cfg["logging"]["level"]).upper())
    configure_logging(run_dir / "run.log", level=log_level)

    experiment = cfg["experiment"]
    if experiment not in _DATASET_IDENTITY:
        raise ValueError(
            f"Unknown experiment {experiment!r}; expected one of {sorted(_DATASET_IDENTITY)}"
        )
    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        write_json(config_path, _json_safe(cfg))

    run_id = run_dir.name
    status_path = run_dir / "run_status.json"
    status = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_id": run_id,
        "sweep_id": None,
        "candidate": {"experiment": experiment, "description": cfg.get("description")},
        "seed": int(cfg["seed"]),
        "repetition": None,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(PROJECT_DIR),
        "dataset": _DATASET_IDENTITY[experiment],
        "selection": _REPORTED_METRIC[experiment],
        "log_path": "run.log",
        "config_path": config_path.name,
        "config_sha256": sha256(config_path),
        "metrics_path": None,
        "metrics_sha256": None,
        "error": None,
    }
    write_json(status_path, status)
    LOGGER.info(
        "RUN_STARTED | run_id=%s | experiment=%s | seed=%s | device=%s | dtype=%s",
        run_id,
        experiment,
        cfg["seed"],
        cfg["device"],
        dtype_label(cfg["dtype"]),
    )
    try:
        device = torch.device(cfg["device"])
        dtype = _resolve_dtype(cfg)
        metrics = _dispatch(cfg, run_dir, device, dtype)
        metrics_path = run_dir / "metrics.json"
        write_json(metrics_path, json.loads(json.dumps(metrics)))
        LOGGER.info("METRICS_WRITTEN | path=%s", metrics_path.name)
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_path=metrics_path.name,
            metrics_sha256=sha256(metrics_path),
        )
        write_json(status_path, status)
        LOGGER.info("RUN_COMPLETED | run_id=%s", run_id)
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        LOGGER.exception("RUN_FAILED | run_id=%s", run_id)
        raise
