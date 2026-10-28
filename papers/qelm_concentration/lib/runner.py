"""Runtime entry point for the QELM concentration reproduction (arXiv:2603.13005).

``cfg["task"]`` selects the experiment family:

- ``landsat_readouts``: QELM readout study on Landsat (E5, E3, E4, E7, E8);
- ``landsat_classical``: classical baselines on the same splits (E10, fair baselines);
- ``narma_scaling``: NARMA-n R^2 versus order at one qubit count (E2);
- ``pareto_search``: multi-objective hyperparameter search (E1);
- ``concentration``: variability versus size and depth (E1b).

Every run writes the evidence files of the logging contract next to the task artifacts:
``run.log``, ``run_status.json``, ``config_snapshot.json`` and ``metrics.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from runtime_lib.dtypes import DtypeSpec

from .concentration_task import run_concentration
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .landsat_task import (
    run_landsat_classical,
    run_landsat_gap_diagnostics,
    run_landsat_readouts,
)
from .narma_task import run_narma_scaling
from .pareto_task import run_pareto_search

logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TASKS = {
    "landsat_readouts": run_landsat_readouts,
    "landsat_classical": run_landsat_classical,
    "landsat_gap_diagnostics": run_landsat_gap_diagnostics,
    "narma_scaling": run_narma_scaling,
    "pareto_search": run_pareto_search,
    "concentration": run_concentration,
}
SELECTION = {
    "landsat_readouts": ("test_f1_macro_per_readout", "maximize", "test"),
    "landsat_classical": ("test_f1_macro_per_baseline", "maximize", "test"),
    "landsat_gap_diagnostics": ("test_f1_macro_per_readout", "maximize", "test"),
    "narma_scaling": ("test_r2_per_order", "maximize", "test"),
    "pareto_search": ("pareto_rank_of_table_I", "minimize", "train_and_test_narma"),
    # Higher variability means weaker concentration; nothing is selected from it.
    "concentration": ("mean_variability", "maximize", "uniform_inputs"),
}
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
    dataset_name = cfg["dataset"]["name"]
    if dataset_name == "landsat":
        dataset_cfg = cfg["dataset"]["landsat"]
        return {
            "name": "statlog_landsat_uci146",
            "splits": ["train", "test"],
            "preprocessing": (
                "pooled sat.trn+sat.tst; stratified subsample; stratified train/test split; "
                f"input_scaling={dataset_cfg['input_scaling']} with train statistics; "
                f"tiled x{dataset_cfg['n_copies']}"
            ),
            "subset": {
                "n_subsample": dataset_cfg["n_subsample"],
                "test_fraction": dataset_cfg["test_fraction"],
                "split_seeds": cfg[cfg["task"]]["split_seeds"],
            },
        }
    if dataset_name == "narma":
        dataset_cfg = cfg["dataset"]["narma"]
        return {
            "name": "narma_synthetic",
            "splits": ["train", "test"],
            "preprocessing": (
                f"u ~ Unif(0, 0.5), input_encoding={dataset_cfg['input_encoding']}; "
                "rolling windows of length N/2"
            ),
            "subset": {
                "orders": dataset_cfg["orders"],
                "t_init": dataset_cfg["t_init"],
                "t_train": dataset_cfg["t_train"],
                "t_test": dataset_cfg["t_test"],
            },
        }
    if dataset_name == "uniform_inputs":
        return {
            "name": "uniform_inputs",
            "splits": ["evaluation"],
            "preprocessing": "u ~ Unif(-1, 1)^(N/2)",
            "subset": {"n_samples": cfg["concentration"]["n_samples"]},
        }
    raise ValueError(f"Unknown dataset name {dataset_name!r}")


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
    selection_name, selection_direction, selection_split = SELECTION[task]
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
            "name": selection_name,
            "direction": selection_direction,
            "split": selection_split,
            "checkpoint": "closed_form_ridge_fit",
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
        "RUN_STARTED | run_id=%s | task=%s | seed=%s | device=%s",
        run_dir.name,
        task,
        cfg["seed"],
        cfg["device"],
    )

    try:
        metrics = {"task": task, **TASKS[task](cfg, run_dir)}
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
