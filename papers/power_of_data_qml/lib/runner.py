"""Runtime entry point: ``train_and_evaluate(cfg, run_dir)``.

Experiments (``cfg["experiment"]``):

* ``study``: the (seed, n, embedding) sweep behind arXiv Figs. 1, 2, S1, S3
  (journal Figs. 3, 4, Supp. Figs. 2, 4); components chosen by
  ``study.components``.
* ``training_size``: arXiv Fig. S2 (journal Supp. Fig. 3).
* ``convention_check``: EXP-AMB, inferred conventions vs the Source Data.
* ``photonic_scaling``: cost of the MerLin banks vs the Fock dimension (probe;
  no model is trained).

Artifacts written to ``run_dir`` (the reproduction's evidence contract):

* ``run.log``: lifecycle, one ``TRAIN_UNIT_COMPLETED`` per unit, evaluations.
* ``run_status.json``: run identity, code state, dataset, selection, status.
* ``config_snapshot.json``: written by the shared runtime before this function.
* ``metrics.json``: aggregated summary rows (mean, std, 95% CI, paper value).
* ``records*.jsonl``: every per-seed record, appended as soon as computed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from . import (
    convention_check,
    embeddings,
    paper_values,
    photonic_scaling,
    study,
    summary,
)
from .data import build_pca_features
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
EXPERIMENTS = ("study", "training_size", "convention_check", "photonic_scaling")
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def _precision_label(dtype_spec) -> str | None:
    return None if dtype_spec is None else dtype_spec.label


def _max_system_size(cfg: dict) -> int:
    sizes = [int(n) for n in cfg["study"]["system_sizes"]]
    if not sizes:
        raise ValueError("study.system_sizes is empty")
    return max(sizes)


def _dataset_identity(cfg: dict) -> dict:
    study_cfg = cfg["study"]
    subset = {
        "subsample_size": study_cfg["subsample_size"],
        "system_sizes": study_cfg["system_sizes"],
        "seeds": study_cfg["seeds"],
        "classical_dataset_classes": study_cfg["classical_dataset_classes"],
    }
    if cfg["experiment"] == "training_size":
        subset["training_size"] = cfg["training_size"]
    return {
        "name": "fashion_mnist",
        "splits": {
            "quantum_dataset": f"train first {study_cfg['quantum_dataset_train_size']} of the subsample, test the rest",
            "classical_dataset": f"train first {study_cfg['classical_dataset_train_size']} of the subsample, test the rest",
            "engineered": f"train first N in {study_cfg['engineered_train_sizes']}, test the rest",
        },
        "preprocessing": "all 70,000 images /255 -> PCA (full SVD) -> per-component standardization",
        "subset": subset,
    }


def _selection(cfg: dict) -> dict:
    model_cfg = cfg["models"]
    return {
        "name": (
            "per-family hyperparameter selection; classification by accuracy, "
            "regression by MAE (maximize -MAE)"
        ),
        "direction": "maximize",
        "split": (
            f"protocols {model_cfg['protocols']}: paper = best test score (test-set selection), "
            f"cv = {model_cfg['cv_folds']}-fold CV on the training split"
        ),
        "checkpoint": "not applicable (closed-form kernel and tree fits; no epochs)",
    }


def _write_status(run_dir: Path, status: dict) -> None:
    write_json(run_dir / "run_status.json", status)


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run the configured experiment and write its evidence to ``run_dir``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (defaults, config overlay and CLI flags).
    run_dir : pathlib.Path
        Timestamped run directory created by the shared runtime.

    Raises
    ------
    KeyError
        If a required config key is missing.
    ValueError
        If ``cfg["experiment"]`` is unknown or a setting is invalid.
    """
    run_dir = Path(run_dir)
    level = LOG_LEVELS[str(cfg["logging"]["level"]).lower()]
    configure_logging(run_dir / "run.log", level=level)
    experiment = cfg["experiment"]
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment}'; expected one of {EXPERIMENTS}"
        )
    config_path = run_dir / "config_snapshot.json"
    status = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": {
            "experiment": experiment,
            "config_description": cfg["description"],
        },
        "seed": cfg["study"]["seeds"],
        "repetition": None,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": _dataset_identity(cfg),
        "selection": _selection(cfg),
        "log_path": "run.log",
        "config_path": config_path.name,
        "config_sha256": sha256(config_path),
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    _write_status(run_dir, status)
    LOGGER.info(
        "RUN_STARTED | run_id=%s | experiment=%s | seeds=%s | device=%s | torch_threads=%d",
        run_dir.name,
        experiment,
        cfg["study"]["seeds"],
        cfg["device"],
        torch.get_num_threads(),
    )
    try:
        started = time.perf_counter()
        pca_features = build_pca_features(cfg["data_root"], _max_system_size(cfg))
        LOGGER.info(
            "DATASET_READY | name=fashion_mnist | images=%d | pca_components=%d | explained_variance=%.4f",
            pca_features.components.shape[0],
            pca_features.components.shape[1],
            float(pca_features.explained_variance_ratio.sum()),
        )
        complex_dtype = embeddings.complex_dtype_for(_precision_label(cfg.get("dtype")))
        conventions = embeddings.EmbeddingConventions.from_config(cfg["conventions"])
        settings = study.StudySettings.from_config(cfg)
        reference_values = paper_values.load_paper_values(cfg["data_root"])

        def on_unit_completed(
            done: int, total: int, record: dict, variant: str | None = None
        ) -> None:
            LOGGER.info(
                "TRAIN_UNIT_COMPLETED | unit=%d/%d | variant=%s | seed=%s | n=%s | embedding=%s | elapsed_s=%.1f",
                done,
                total,
                variant,
                record["seed"],
                record["n"],
                record.get("embedding"),
                record["elapsed_seconds"],
            )

        LOGGER.info(
            "TRAINING_STARTED | experiment=%s | embeddings=%s | system_sizes=%s | components=%s | "
            "protocols=%s | grid=%s | precision=%s",
            experiment,
            list(settings.embeddings),
            list(settings.system_sizes),
            list(settings.components),
            list(settings.protocols),
            settings.grid,
            complex_dtype,
        )
        if experiment == "convention_check":
            outcome = convention_check.run_convention_check(
                cfg,
                pca_features,
                complex_dtype,
                run_dir,
                reference_values,
                on_unit_completed,
            )
            metrics = {"experiment": experiment, **outcome}
        elif experiment == "photonic_scaling":
            outcome = photonic_scaling.run_photonic_scaling(
                cfg,
                pca_features,
                settings,
                conventions,
                complex_dtype,
                run_dir / "records.jsonl",
                on_unit_completed,
            )
            metrics = {"experiment": experiment, **outcome}
        else:
            records_path = run_dir / "records.jsonl"
            if experiment == "study":
                records = study.run_study(
                    pca_features,
                    settings,
                    conventions,
                    complex_dtype,
                    records_path,
                    on_unit_completed,
                )
            else:
                records = study.run_training_size_study(
                    cfg,
                    pca_features,
                    settings,
                    conventions,
                    complex_dtype,
                    records_path,
                    on_unit_completed,
                )
            rows = [row for record in records for row in summary.record_rows(record)]
            metrics = {
                "experiment": experiment,
                "records_file": records_path.name,
                "record_count": len(records),
                "summary": summary.aggregate_rows(rows, reference_values),
            }
        metrics["elapsed_seconds"] = time.perf_counter() - started
        metrics_path = run_dir / "metrics.json"
        write_json(metrics_path, metrics)
        LOGGER.info(
            "METRICS_WRITTEN | path=%s | elapsed_s=%.1f",
            metrics_path.name,
            metrics["elapsed_seconds"],
        )
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(metrics_path),
        )
        _write_status(run_dir, status)
        LOGGER.info("RUN_COMPLETED | run_id=%s", run_dir.name)
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _write_status(run_dir, status)
        LOGGER.exception("RUN_FAILED | run_id=%s", run_dir.name)
        raise
