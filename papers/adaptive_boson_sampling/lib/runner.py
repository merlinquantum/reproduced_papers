"""Runtime entry point for the Adaptive Boson Sampling kernel reproduction.

Two experiment types, selected by ``cfg["experiment"]``:

- ``platform_kernel`` (E1, E3-E5): simulate one ABS platform, write its 15x15
  (or 3x3) kernel and compare it with the paper's printed or digitised kernels.
- ``classification`` (E6-E8): SVM on the 1D or moons + K-means dataset with ABS
  kernels and classical baselines, over identical random partitions.

Artifacts in ``run_dir`` (public reproduction contract): ``run.log``,
``run_status.json``, ``config_snapshot.json`` (written by the shared runtime),
``metrics.json``, plus ``kernels.npz`` and ``kernels.png`` (platform_kernel) or
``summary.csv`` and ``dataset_<repetition>.npz`` (classification).
"""

from __future__ import annotations

import csv
import itertools
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from runtime_lib.dtypes import dtype_label

from .abs_merlin import simulated_state_table
from .circuits import get_platform
from .data import build_dataset, digitized_kernels_path, partition_indices
from .evaluation import ARMS, evaluate_over_partitions
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .kernels import (
    designed_kernel,
    load_digitized_kernel,
    quantum_kernel,
    u0_for_config,
)
from .models import build_instances
from .plotting import save_kernel_heatmaps

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
STATUS_SCHEMA_VERSION = 1
# Phase 3 configs predate the MerLin backends; without a ``simulations`` key a
# platform_kernel run uses the numpy reference simulator of ``lib/fock.py``.
REFERENCE_SIMULATION = {"label": "numpy", "backend": "numpy"}


def _require(mapping: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"config section {where!r} is missing keys {missing}")


def validate_config(cfg: dict[str, Any]) -> None:
    """Check the keys every experiment needs; fail clearly instead of defaulting.

    Raises
    ------
    ValueError
        For a missing key, an unsupported device or dtype, or an unknown
        experiment.
    """
    _require(cfg, ("experiment", "device", "dtype", "validity"), "<root>")
    if cfg["device"] != "cpu":
        # The simulation is exact NumPy linear algebra on at most 8 modes and 3
        # photons; there is no accelerator path.
        raise ValueError(
            f"device {cfg['device']!r} unsupported; this paper runs on cpu"
        )
    if dtype_label(cfg["dtype"]) != "float64":
        raise ValueError("this reproduction computes in complex128/float64 only")
    if cfg["experiment"] == "platform_kernel":
        _require(cfg["platform_kernel"], ("platform", "kernel_kind"), "platform_kernel")
    elif cfg["experiment"] == "classification":
        _require(cfg, ("dataset", "models", "evaluation"), "<root>")
        _require(
            cfg["evaluation"],
            ("n_partitions", "test_fraction", "svm_C", "arms", "tuning"),
            "evaluation",
        )
        unknown_arms = set(cfg["evaluation"]["arms"]) - set(ARMS)
        if unknown_arms:
            raise ValueError(f"unknown evaluation arms {sorted(unknown_arms)}")
        for spec in cfg["models"]:
            _require(spec, ("label", "kind"), "models[]")
    else:
        raise ValueError(f"unknown experiment {cfg['experiment']!r}")


def _dataset_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg["experiment"] == "platform_kernel":
        return {
            "name": f"abs_outcomes_platform_{cfg['platform_kernel']['platform']}",
            "splits": "none (all post-selected outcomes)",
            "preprocessing": "none",
            "subset": cfg["validity"],
        }
    dataset = cfg["dataset"]
    return {
        "name": dataset["name"],
        "splits": (
            f"{cfg['evaluation']['n_partitions']} random partitions, test fraction "
            f"{cfg['evaluation']['test_fraction']}, seeds 0.."
            f"{cfg['evaluation']['n_partitions'] - 1}"
        ),
        "preprocessing": (
            "K-means outcome assignment " + str(dataset.get("assignment"))
            if dataset["name"] == "moons_kmeans"
            else "none"
        ),
        "subset": cfg["validity"],
    }


def _selection_rule(cfg: dict[str, Any]) -> dict[str, str]:
    """Return the declared comparison metric; no run selects a checkpoint."""
    if cfg["experiment"] == "classification":
        return {
            "name": "mean_test_accuracy",
            "direction": "maximize",
            "split": "test",
            "checkpoint": "none (SVM refit per partition)",
        }
    return {
        "name": "kernel_error_vs_paper",
        "direction": "minimize",
        "split": "none (all outcomes)",
        "checkpoint": "none (no trained parameters)",
    }


def _new_status(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": cfg.get("candidate", {}),
        "seed": cfg.get("seed"),
        "repetition": cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": _dataset_identity(cfg),
        "selection": _selection_rule(cfg),
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(run_dir / "config_snapshot.json"),
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def compare_kernels(simulated: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return MAE, max error and Pearson correlation over off-diagonal entries."""
    a, b = _upper_triangle(simulated), _upper_triangle(reference)
    return {
        "mean_abs_error": float(np.mean(np.abs(a - b))),
        "max_abs_error": float(np.max(np.abs(a - b))),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
    }


def run_platform_kernel(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Simulate one platform's kernel(s) and compare with the paper (E1, E3-E5)."""
    section = cfg["platform_kernel"]
    platform = get_platform(section["platform"])
    logger.info(
        "DATASET_READY | dataset=abs_outcomes | platform=%s | outcomes=%d | d=%d",
        platform.name,
        len(platform.outcomes),
        platform.output_dimension,
    )
    if platform.needs_random_u0:
        u0_cfg = section["u0"]
        choices = [
            (
                seed,
                *u0_for_config(platform, seed, u0_cfg, digitized_kernels_path(cfg)),
            )
            for seed in u0_cfg["seeds"]
        ]
    else:
        choices = [(None, None, {"strategy": "fixed_mesh"})]

    reference = section.get("paper_reference", {})
    digitized = {
        key: load_digitized_kernel(digitized_kernels_path(cfg), key)
        for key in reference.get("digitized_keys", [])
    }
    design = designed_kernel(platform) if platform.needs_random_u0 else None

    simulations = section.get("simulations", [REFERENCE_SIMULATION])
    labels = [simulation["label"] for simulation in simulations]
    if len(set(labels)) != len(labels):
        raise ValueError(f"simulation labels must be unique, got {labels}")
    instances, kernels_to_save = [], {}
    for (seed, u0, record), simulation in itertools.product(choices, simulations):
        logger.info(
            "EVALUATION_STARTED | platform=%s | u0_seed=%s | simulation=%s",
            platform.name,
            seed,
            simulation,
        )
        states, probabilities = simulated_state_table(platform, u0, simulation)
        diagonal_states = np.array(
            [states[k, k] for k in range(len(platform.outcomes))]
        )
        kernel = quantum_kernel(diagonal_states, section["kernel_kind"])
        name = "fixed" if seed is None else f"u0_seed_{seed}"
        if len(simulations) > 1:
            name = f"{simulation['label']}__{name}"
        kernels_to_save[name] = kernel
        entry: dict[str, Any] = {
            "u0_seed": seed,
            "simulation": simulation,
            "u0_selection": record,
            "kernel": kernel.tolist(),
            "postselection_probability": np.diag(probabilities).tolist(),
            "total_postselection_probability": float(np.trace(probabilities)),
        }
        if "entries" in reference:
            errors = [
                abs(kernel[row, column] - value)
                for row, column, value in reference["entries"]
            ]
            entry["printed_entries"] = {
                "max_abs_error": float(max(errors)),
                "tolerance": reference["tolerance"],
                "within_tolerance": bool(max(errors) <= reference["tolerance"]),
            }
        entry["vs_digitized"] = {
            key: compare_kernels(kernel, value) for key, value in digitized.items()
        }
        if design is not None:
            entry["vs_designed_kernel"] = compare_kernels(kernel, design)
        instances.append(entry)
        logger.info(
            "EVALUATION_COMPLETED | platform=%s | u0_seed=%s | summary=%s",
            platform.name,
            seed,
            {
                key: entry[key]
                for key in ("printed_entries", "vs_digitized", "vs_designed_kernel")
                if key in entry
            },
        )

    if design is not None:
        kernels_to_save["designed"] = design
    kernels_to_save.update({f"paper_{key}": value for key, value in digitized.items()})
    np.savez(run_dir / "kernels.npz", **kernels_to_save)
    save_kernel_heatmaps(
        dict(list(kernels_to_save.items())[:6]),
        run_dir / "kernels.png",
        f"Platform {platform.name} kernels ({cfg['validity']})",
    )
    logger.info("ARTIFACT_WRITTEN | path=kernels.npz | path=kernels.png")
    return {
        "experiment": "platform_kernel",
        "platform": platform.name,
        "output_dimension": platform.output_dimension,
        "outcomes": [list(outcome) for outcome in platform.outcomes],
        "kernel_kind": section["kernel_kind"],
        "validity": cfg["validity"],
        "instances": instances,
    }


def _dataset_variants(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base = cfg["dataset"]
    repetitions = cfg.get("dataset_repetitions", [])
    if not repetitions:
        return [deepcopy(base)]
    return [{**deepcopy(base), **override} for override in repetitions]


def run_classification(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Evaluate every model on every dataset variant and arm (E6-E8)."""
    evaluation_cfg = cfg["evaluation"]
    variants = []
    for index, dataset_cfg in enumerate(_dataset_variants(cfg)):
        dataset = build_dataset(dataset_cfg)
        partitions = [
            (
                seed,
                *partition_indices(
                    len(dataset.labels), evaluation_cfg["test_fraction"], seed
                ),
            )
            for seed in range(evaluation_cfg["n_partitions"])
        ]
        np.savez(
            run_dir / f"dataset_{index}.npz",
            outcome_index=dataset.outcome_index,
            labels=dataset.labels,
            features=dataset.features,
            **{
                key: value
                for key, value in dataset.extras.items()
                if isinstance(value, np.ndarray)
            },
        )
        variants.append((index, dataset_cfg, dataset, partitions))
        logger.info(
            "DATASET_VARIANT_READY | index=%d | name=%s | n=%d | positives=%d | "
            "weighted_purity=%s",
            index,
            dataset.name,
            len(dataset.labels),
            int(np.sum(dataset.labels == 1)),
            dataset.extras.get("weighted_purity", "n/a"),
        )
    logger.info("DATASET_READY | variants=%d | artifacts=dataset_*.npz", len(variants))

    results: dict[str, Any] = {}
    summary_rows = []
    total_models = len(cfg["models"])
    logger.info(
        "TRAINING_STARTED | models=%d | arms=%s | partitions=%d | variants=%d",
        total_models,
        evaluation_cfg["arms"],
        evaluation_cfg["n_partitions"],
        len(variants),
    )
    for model_number, spec in enumerate(cfg["models"], start=1):
        label = spec["label"]
        results[label] = {"spec": spec, "arms": {}}
        # Instances depend on the dataset variant but not on the arm.
        instances_by_variant = [
            build_instances(spec, dataset, cfg) for _, _, dataset, _ in variants
        ]
        for arm in evaluation_cfg["arms"]:
            logger.info("EVALUATION_STARTED | model=%s | arm=%s", label, arm)
            per_instance = []
            for (variant_index, dataset_cfg, _, partitions), instances in zip(
                variants, instances_by_variant
            ):
                for instance in instances:
                    outcome = evaluate_over_partitions(
                        instance.fit_one, partitions, arm
                    )
                    per_instance.append(
                        {
                            "dataset_variant": variant_index,
                            "dataset_config": dataset_cfg,
                            "instance": instance.instance,
                            "details": instance.details,
                            **outcome,
                        }
                    )
                    logger.info(
                        "TRAIN_MODEL_COMPLETED | model=%d/%d | label=%s | arm=%s | "
                        "variant=%d | instance=%s | mean_test_acc=%.4f | "
                        "mean_train_acc=%.4f",
                        model_number,
                        total_models,
                        label,
                        arm,
                        variant_index,
                        instance.instance,
                        outcome["mean_test_accuracy"],
                        outcome["mean_train_accuracy"],
                    )
            means = np.array([entry["mean_test_accuracy"] for entry in per_instance])
            aggregate = {
                "n_instances": len(per_instance),
                "mean_test_accuracy_over_instances": float(means.mean()),
                "std_test_accuracy_over_instances": float(means.std()),
                "min_instance_mean_test_accuracy": float(means.min()),
                "max_instance_mean_test_accuracy": float(means.max()),
                # Partition-level spread, pooled over instances and variants.
                "pooled_std_test_accuracy": float(
                    np.std(
                        [
                            value
                            for entry in per_instance
                            for value in entry["test_accuracy_per_partition"]
                        ]
                    )
                ),
            }
            results[label]["arms"][arm] = {
                "aggregate": aggregate,
                "instances": per_instance,
            }
            summary_rows.append({"model": label, "arm": arm, **aggregate})
            logger.info(
                "EVALUATION_COMPLETED | model=%s | arm=%s | instances=%d | "
                "mean_test_acc=%.4f | std_over_instances=%.4f",
                label,
                arm,
                aggregate["n_instances"],
                aggregate["mean_test_accuracy_over_instances"],
                aggregate["std_test_accuracy_over_instances"],
            )

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info("ARTIFACT_WRITTEN | path=summary.csv")
    return {
        "experiment": "classification",
        "dataset": cfg["dataset"]["name"],
        "validity": cfg["validity"],
        "n_variants": len(variants),
        "results": results,
    }


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """Run one configured experiment and write its artifacts to ``run_dir``.

    Parameters
    ----------
    cfg : dict
        Resolved experiment configuration.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime; it already
        holds ``config_snapshot.json``.

    Raises
    ------
    ValueError
        If the configuration is invalid. Any failure is recorded in
        ``run_status.json`` and re-raised.
    """
    run_dir = Path(run_dir)
    configure_logging(
        run_dir / "run.log",
        level=getattr(logging, str(cfg["logging"]["level"]).upper()),
    )
    if not (run_dir / "config_snapshot.json").exists():
        # Direct calls (tests, notebooks) bypass the shared runtime snapshot.
        write_json(
            run_dir / "config_snapshot.json",
            {key: value for key, value in cfg.items() if key != "dtype"}
            | {"dtype": dtype_label(cfg.get("dtype"))},
        )
    status_path = run_dir / "run_status.json"
    status: dict[str, Any] | None = None
    try:
        validate_config(cfg)
        status = _new_status(cfg, run_dir)
        write_json(status_path, status)
        logger.info(
            "RUN_STARTED | run_id=%s | experiment=%s | validity=%s",
            run_dir.name,
            cfg["experiment"],
            cfg["validity"],
        )
        if cfg["experiment"] == "platform_kernel":
            metrics = run_platform_kernel(cfg, run_dir)
        else:
            metrics = run_classification(cfg, run_dir)
        write_json(run_dir / "metrics.json", metrics)
        logger.info("METRICS_WRITTEN | path=metrics.json")
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(run_dir / "metrics.json"),
        )
        write_json(status_path, status)
        logger.info("RUN_COMPLETED | run_id=%s", run_dir.name)
    except Exception as exc:
        if status is None:
            status = {
                "schema_version": STATUS_SCHEMA_VERSION,
                "run_id": run_dir.name,
                "status": "FAILED",
                "started_at": utc_now(),
                "log_path": "run.log",
                "config_path": "config_snapshot.json",
                "metrics_path": "metrics.json",
            }
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("RUN_FAILED | run_id=%s", run_dir.name)
        raise
