"""Runtime entry point for the QNN_STOCK reproduction.

``experiment.kind`` selects what one run does:

- ``qstl``: train one QBGU circuit per ticker (Sec. IV, VI.B; E3, E6).
- ``qmtl``: train the share-and-specify circuit over all tickers (Sec. V, VI.C; E4, E7).
- ``baselines``: fit the classical baselines B0-B4 (Fair Baseline Plan).
- ``loader``: train only the HEA context loader (Sec. III, VI.A; E2).
- ``photonic``: train the MerLin photonic ADAPTATION of the context-conditional
  predictor (Phase 4; not the paper's method).

Artifacts written to ``run_dir`` (paths in ``run_status.json`` are relative):
``run.log``, ``run_status.json``, ``config_snapshot.json``, ``metrics.json``,
``distributions.json`` (targets and model tables), ``losses.csv`` (per-epoch
history) and ``params.npz`` (trained angles).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import counts_table, fit_baselines, fit_pooled_baseline
from .data import TaskData, build_tasks, joint_distribution
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .metrics import (
    distribution_metrics,
    normalize_rows,
    path_step_metrics,
    sequence_metrics,
)
from .model import ContextualTask
from .photonic_merlin import fit_photonic_model
from .training import train_context_loader, train_qmtl, train_qstl

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DATA_LABEL = "substitute-data"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _jsonable(cfg: Any) -> Any:
    """Config with runtime-resolved dtype objects turned back into labels."""

    if isinstance(cfg, dict):
        return {key: _jsonable(value) for key, value in cfg.items()}
    if isinstance(cfg, list):
        return [_jsonable(value) for value in cfg]
    if hasattr(cfg, "label") and hasattr(cfg, "torch"):
        return cfg.label
    return cfg


def _evaluate(
    task_data: TaskData,
    model_joint_on_train: np.ndarray,
    conditional: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Metrics against the training target (paper protocol) and the test split."""

    levels = task_data.levels
    results = {}
    for split, windows in (
        ("train", task_data.train_windows),
        ("test", task_data.test_windows),
    ):
        target = joint_distribution(windows, levels, task_data.outcome_levels)
        results[split] = {
            **distribution_metrics(target, model_joint_on_train, conditional),
            **sequence_metrics(windows, levels, conditional),
        }
        if task_data.horizon > 1:
            results[split].update(
                path_step_metrics(windows, task_data.horizon, conditional)
            )
    return results


def _quantum_entry(
    task_data: TaskData, task: ContextualTask, circuit, n_params: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a trained circuit two ways.

    ``inference``: load each basis context and read the prediction register
    (the paper's intended use). ``output``: the conditional implied by the
    measured QBGU output state on the superposed training input.
    """

    output_joint = task.output_joint(circuit)
    inference_conditional = task.inference_conditional(circuit)
    output_conditional = normalize_rows(output_joint)
    fidelity = task.fidelity(circuit)
    metrics = {
        "n_params": n_params,
        "final_fidelity": fidelity,
        "final_swap_loss_exact": 0.5 * (1.0 - fidelity),
        "inference": _evaluate(task_data, output_joint, inference_conditional),
        "output": _evaluate(task_data, output_joint, output_conditional),
    }
    tables = {
        "output_joint": output_joint.tolist(),
        "inference_conditional": inference_conditional.tolist(),
    }
    return metrics, tables


def _status(
    cfg: dict[str, Any], run_dir: Path, tasks: list[TaskData] | None
) -> dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    kind = cfg["experiment"]["kind"]
    if kind == "baselines":
        selection = {
            "name": "validation_log_loss",
            "direction": "minimize",
            "split": "validation (last fraction of train)",
            "checkpoint": "refit_on_full_train",
        }
    else:
        # No model selection: the training objective is minimized and the
        # final epoch is reported, as in the paper.
        selection = {
            "name": "training_objective",
            "direction": "minimize",
            "split": "train",
            "checkpoint": "final_epoch",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        # A sweep coordinator assigns the run ID it tracks; standalone runs use
        # the timestamped run directory name.
        "run_id": cfg.get("run_id") or run_dir.name,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": cfg.get("candidate", {}),
        "seed": cfg.get("seed"),
        "repetition": cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(_REPO_ROOT),
        "dataset": {
            "name": f"{dataset_cfg['source_dir']} ({DATA_LABEL})",
            "splits": {
                task.ticker: {
                    "train": len(task.train_windows),
                    "test": len(task.test_windows),
                }
                for task in (tasks or [])
            },
            "preprocessing": {
                key: dataset_cfg[key]
                for key in (
                    "price_field",
                    "end_date",
                    "moving_average_window",
                    "quantization",
                    "levels",
                    "context_length",
                    "horizon",
                    "train_fraction",
                )
            },
            "subset": (
                "full history up to end_date"
                if dataset_cfg["train_size"] is None
                else f"contiguous block of {dataset_cfg['train_size']} training windows, "
                "subset seed "
                + str(
                    cfg["seed"]
                    if dataset_cfg["train_subset_seed"] == "seed"
                    else dataset_cfg["train_subset_seed"]
                )
            ),
        },
        "selection": selection,
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": None,
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    fields = sorted(
        {key for row in history for key in row}, key=lambda k: (k != "stage", k)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def _run_experiment(
    cfg: dict[str, Any], tasks: list[TaskData], history: list
) -> tuple[dict, dict, dict]:
    kind = cfg["experiment"]["kind"]
    rng = np.random.default_rng(int(cfg["seed"]))
    metrics: dict[str, Any] = {
        "experiment": kind,
        "data_label": DATA_LABEL,
        "assets": {},
    }
    tables: dict[str, Any] = {}
    params: dict[str, np.ndarray] = {}

    for task_data in tasks:
        metrics["assets"][task_data.ticker] = {
            "data": task_data.provenance(),
            "models": {},
        }
        tables[task_data.ticker] = {
            "train_joint": joint_distribution(
                task_data.train_windows, task_data.levels, task_data.outcome_levels
            ).tolist(),
            "test_joint": joint_distribution(
                task_data.test_windows, task_data.levels, task_data.outcome_levels
            ).tolist(),
        }

    logger.info(
        "TRAINING_STARTED | kind=%s | tickers=%s", kind, [t.ticker for t in tasks]
    )

    if kind == "baselines":
        baseline_cfg = cfg["baselines"]
        fitted = {
            t.ticker: fit_baselines(
                t.train_windows,
                t.levels,
                t.outcome_levels,
                baseline_cfg,
                int(cfg["seed"]),
                cfg["device"],
            )
            for t in tasks
        }
        if len(tasks) > 1:
            pooled = fit_pooled_baseline(
                [t.train_windows for t in tasks],
                tasks[0].levels,
                tasks[0].outcome_levels,
                baseline_cfg,
            )
            for task_data, result in zip(tasks, pooled):
                fitted[task_data.ticker]["B4_pooled_shrinkage"] = result
        logger.info("TRAINING_COMPLETED | kind=baselines")
        for task_data in tasks:
            train_context = joint_distribution(
                task_data.train_windows, task_data.levels, task_data.outcome_levels
            ).sum(axis=1)
            for name, result in fitted[task_data.ticker].items():
                logger.info(
                    "EVALUATION_STARTED | ticker=%s | model=%s", task_data.ticker, name
                )
                conditional = result["conditional"]
                entry = {
                    "n_params": result["n_params"],
                    "selected": result.get("selected"),
                    "validation_log_loss": result.get("validation_log_loss"),
                    # Baselines model only the next step, so their joint uses the
                    # empirical training context distribution.
                    "inference": _evaluate(
                        task_data, train_context[:, None] * conditional, conditional
                    ),
                }
                metrics["assets"][task_data.ticker]["models"][name] = entry
                tables[task_data.ticker][f"{name}_conditional"] = conditional.tolist()
                logger.info(
                    "EVALUATION_COMPLETED | ticker=%s | model=%s | test_log_loss=%.6f | "
                    "train_kl_conditional_sum=%.6f",
                    task_data.ticker,
                    name,
                    entry["inference"]["test"]["log_loss"],
                    entry["inference"]["train"]["kl_conditional_sum"],
                )
        return metrics, tables, params

    if kind == "photonic":
        photonic_cfg = cfg["photonic"]
        fitted_photonic = {}
        for task_data in tasks:
            if task_data.levels != 2 or task_data.horizon != 1:
                raise ValueError(
                    "The photonic adaptation needs binary symbols and horizon 1; got "
                    f"levels={task_data.levels}, horizon={task_data.horizon}"
                )
            fitted_photonic[task_data.ticker] = fit_photonic_model(
                counts_table(
                    task_data.train_windows, task_data.levels, task_data.outcome_levels
                ),
                photonic_cfg,
                int(cfg["seed"]),
                cfg["dtype"].torch,
                f"photonic_{task_data.ticker}",
                history,
            )
        logger.info("TRAINING_COMPLETED | kind=photonic")
        for task_data in tasks:
            logger.info(
                "EVALUATION_STARTED | ticker=%s | model=PHOTONIC", task_data.ticker
            )
            result = fitted_photonic[task_data.ticker]
            conditional = result["conditional"]
            train_context = joint_distribution(
                task_data.train_windows, task_data.levels, task_data.outcome_levels
            ).sum(axis=1)
            entry = {
                "n_params": result["n_params"],
                "final_train_loss": result["final_train_loss"],
                "keep_rate_mean": result["keep_rate_mean"],
                "keep_rate_min": result["keep_rate_min"],
                # The model predicts only the next step, so its joint uses the
                # empirical training context distribution, as for the baselines.
                "inference": _evaluate(
                    task_data, train_context[:, None] * conditional, conditional
                ),
            }
            metrics["assets"][task_data.ticker]["models"]["PHOTONIC"] = entry
            tables[task_data.ticker]["PHOTONIC_conditional"] = conditional.tolist()
            tables[task_data.ticker]["PHOTONIC_input_state"] = result["input_state"]
            logger.info(
                "EVALUATION_COMPLETED | ticker=%s | model=PHOTONIC | test_log_loss=%.6f | "
                "train_log_loss=%.6f | keep_rate_mean=%.4f",
                task_data.ticker,
                entry["inference"]["test"]["log_loss"],
                entry["inference"]["train"]["log_loss"],
                entry["keep_rate_mean"],
            )
        return metrics, tables, params

    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    loader_cfg = model_cfg["context_loader"]

    contextual_tasks = []
    for task_data in tasks:
        train_joint = joint_distribution(
            task_data.train_windows, task_data.levels, task_data.outcome_levels
        )
        context_state = None
        if loader_cfg["mode"] == "hea" or kind == "loader":
            context_state, loader_params = train_context_loader(
                train_joint.sum(axis=1),
                loader_cfg,
                rng,
                f"loader_{task_data.ticker}",
                history,
            )
            params[f"loader_{task_data.ticker}"] = loader_params
            metrics["assets"][task_data.ticker]["loader"] = history[-1] | {
                "n_params": int(loader_params.size)
            }
        elif loader_cfg["mode"] != "exact":
            raise ValueError(f"Unknown context_loader.mode '{loader_cfg['mode']}'")
        contextual_tasks.append(ContextualTask(train_joint, context_state))

    if kind == "loader":
        logger.info("TRAINING_COMPLETED | kind=loader")
        for task_data in tasks:
            logger.info(
                "EVALUATION_STARTED | ticker=%s | model=loader", task_data.ticker
            )
            logger.info(
                "EVALUATION_COMPLETED | ticker=%s | model=loader | tvd=%.6f",
                task_data.ticker,
                metrics["assets"][task_data.ticker]["loader"]["tvd"],
            )
        return metrics, tables, params

    if kind == "qstl":
        trained = []
        for task_data, task in zip(tasks, contextual_tasks):
            circuit, task_params = train_qstl(
                task, model_cfg, training_cfg, rng, f"qstl_{task_data.ticker}", history
            )
            params[f"qstl_{task_data.ticker}"] = task_params
            trained.append((circuit, int(task_params.size)))
        model_name = "QSTL"
    elif kind == "qmtl":
        names = [t.ticker for t in tasks]
        circuits, qmtl_params = train_qmtl(
            contextual_tasks, names, model_cfg, training_cfg, rng, history
        )
        params.update({f"qmtl_{key}": value for key, value in qmtl_params.items()})
        per_asset = int(
            qmtl_params["shared"].size + qmtl_params[f"specific_{names[0]}"].size
        )
        trained = [(circuit, per_asset) for circuit in circuits]
        metrics["qmtl_total_params"] = int(sum(v.size for v in qmtl_params.values()))
        model_name = "QMTL"
    else:
        raise ValueError(f"Unknown experiment.kind '{kind}'")
    logger.info("TRAINING_COMPLETED | kind=%s", kind)

    for task_data, task, (circuit, n_model_params) in zip(
        tasks, contextual_tasks, trained
    ):
        logger.info(
            "EVALUATION_STARTED | ticker=%s | model=%s", task_data.ticker, model_name
        )
        entry, model_tables = _quantum_entry(task_data, task, circuit, n_model_params)
        metrics["assets"][task_data.ticker]["models"][model_name] = entry
        tables[task_data.ticker].update(
            {f"{model_name}_{k}": v for k, v in model_tables.items()}
        )
        logger.info(
            "EVALUATION_COMPLETED | ticker=%s | model=%s | fidelity=%.6f | "
            "train_kl_joint=%.6f | train_kl_conditional_sum=%.6f | test_log_loss=%.6f",
            task_data.ticker,
            model_name,
            entry["final_fidelity"],
            entry["inference"]["train"]["kl_joint"],
            entry["inference"]["train"]["kl_conditional_sum"],
            entry["inference"]["test"]["log_loss"],
        )
    return metrics, tables, params


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """Run one configured experiment and write its evidence to ``run_dir``."""

    run_dir = Path(run_dir)
    log_level = getattr(
        logging, str(cfg.get("logging", {}).get("level", "info")).upper()
    )
    configure_logging(run_dir / "run.log", level=log_level)

    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(_jsonable(cfg), indent=2), encoding="utf-8")
    status = _status(cfg, run_dir, None)
    status["config_sha256"] = sha256(config_path)
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "RUN_STARTED | run_id=%s | kind=%s | seed=%s | data=%s",
        status["run_id"],
        cfg["experiment"]["kind"],
        cfg.get("seed"),
        DATA_LABEL,
    )
    try:
        tasks = build_tasks(cfg)
        status["dataset"] = _status(cfg, run_dir, tasks)["dataset"]
        write_json(run_dir / "run_status.json", status)
        logger.info("DATASET_READY | tickers=%s", [t.ticker for t in tasks])

        history: list[dict[str, Any]] = []
        metrics, tables, params = _run_experiment(cfg, tasks, history)

        _write_history(run_dir / "losses.csv", history)
        write_json(run_dir / "distributions.json", tables)
        if params:
            np.savez(run_dir / "params.npz", **params)
        logger.info("ARTIFACT_WRITTEN | files=losses.csv,distributions.json,params.npz")

        write_json(run_dir / "metrics.json", metrics)
        status["metrics_sha256"] = sha256(run_dir / "metrics.json")
        logger.info("METRICS_WRITTEN | path=metrics.json")
        status.update(status="COMPLETED", completed_at=utc_now())
        write_json(run_dir / "run_status.json", status)
        logger.info("RUN_COMPLETED | run_id=%s", status["run_id"])
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise
