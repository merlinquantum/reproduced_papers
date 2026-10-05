"""Shared-runtime entry point for one Q-CHIPP task/model/seed run.

Artifacts written to ``run_dir`` (the shared runtime adds ``config_snapshot.json``
and ``run.log``):

- ``run_status.json``: lifecycle, dataset provenance and category codes. The
  category codes are required to encode new peptides at inference time.
- ``metrics.json``: train/test metrics, parameter counts, training record and,
  for MerLin models, optical resource metadata and finite-shot metrics.
- ``predictions.csv``: per-peptide probabilities for both splits.
- ``model.pt`` (torch models) or ``model.pkl`` (scikit-learn controls).
- ``losses.csv``: per-update training objective (torch models only).
"""

from __future__ import annotations

import importlib.metadata
import logging
import pickle
import time
from pathlib import Path

import torch

from .artifacts import (
    git_state,
    sha256,
    utc_now,
    write_json,
    write_losses,
    write_predictions,
)
from .baselines import (
    BASELINE_MODELS,
    baseline_features,
    fit_baseline,
    predict_baseline,
)
from .data import load_task
from .metrics import binary_metrics
from .models import PHOTONIC_MODELS, TRAINABLE_MODELS
from .training import predict_probabilities, train_model

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]
CELLS_PER_PEPTIDE = 3


def _initial_status(cfg: dict, run_dir: Path) -> dict:
    return {
        "schema_version": 2,
        "run_id": cfg.get("run_id") or run_dir.name,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": {"model": cfg["model"]["name"], "task": cfg["dataset"]["task"]},
        "seed": int(cfg["seed"]),
        "repetition": cfg.get("repetition"),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": {
            **git_state(PROJECT_DIR),
            "source_hashes": {
                str(p.relative_to(PROJECT_DIR)): sha256(p)
                for p in sorted((PROJECT_DIR / "lib").glob("*.py"))
            },
        },
        "dataset": {
            "name": "Q-CHIPP DataS1",
            "splits": ["train", "test"],
            "subset": "official released task splits",
        },
        "selection": "last fixed-budget checkpoint; no test-set selection",
        "metrics_sha256": None,
        "error": None,
    }


def _photonic_metadata(model, cfg: dict) -> dict:
    name = cfg["model"]["name"]
    return {
        "status": "PARTIAL_MERLIN_TRANSLATION",
        "computation_space": "FOCK",
        "detector_model": "ideal photon-number resolving",
        "n_photons": model.n_photons,
        "n_modes": len(model.input_state),
        "input_state": list(model.input_state),
        "phase_reference_mode": 6 if name == "merlin_reference" else None,
        "encoding": (
            "residue-local QRAC angles, phase scale 1; measured features "
            "re-encoded at pi"
        ),
        "measurement_strategy": "full Fock probabilities",
        "postselection": "none",
        # merlin.__version__ is stale in the 0.4.2 wheel, so read the
        # installed distribution version instead.
        "backend": (
            f"MerLin {importlib.metadata.version('merlinquantum')} "
            "CPU analytic simulator"
        ),
        "training_shots": 0,
        "cells_per_peptide": CELLS_PER_PEPTIDE,
        "seed": int(cfg["seed"]),
    }


def _finite_shot_metrics(model, cfg: dict, data: dict) -> dict:
    """Evaluate the test split with sampled Fock outcomes for each shot budget."""
    results = {}
    for shots in cfg["evaluation"]["shots"]:
        if shots == 0:
            continue
        sampling_seed = int(cfg["seed"]) + int(shots)
        torch.manual_seed(sampling_seed)
        LOGGER.info("EVALUATION_STARTED | split=test | shots_per_cell=%d", shots)
        start = time.perf_counter()
        probabilities = predict_probabilities(
            model, cfg, data["test"]["bits"], shots=int(shots)
        ).numpy()
        metrics = binary_metrics(data["test"]["targets"].numpy(), probabilities)
        metrics.update(
            shots_per_cell=int(shots),
            shots_per_peptide=CELLS_PER_PEPTIDE * int(shots),
            wall_seconds=time.perf_counter() - start,
            sampling_seed=sampling_seed,
        )
        results[str(shots)] = metrics
        LOGGER.info(
            "EVALUATION_COMPLETED | split=test | shots_per_cell=%d | f1=%.6f",
            shots,
            metrics["f1"],
        )
    return results


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Train one model on one released task split and write its artifacts.

    Parameters
    ----------
    cfg : dict
        Resolved shared-runtime config. Needs ``model.name``,
        ``dataset.task``, ``dataset.positions``, ``seed``, ``dtype``,
        ``device``, ``training``, ``evaluation`` and ``baselines``.
    run_dir : Path
        Run directory created by the shared runtime.

    Raises
    ------
    ValueError
        If the model name is unknown or a non-CPU device is requested.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["model"]["name"]
    if name not in TRAINABLE_MODELS + BASELINE_MODELS:
        raise ValueError(f"Model not implemented: {name}")
    if cfg["device"] != "cpu":
        raise ValueError("This reproduction is verified on CPU only")
    # The reported results were produced single-threaded; thread count changes
    # floating-point reduction order.
    torch.set_num_threads(1)

    status = _initial_status(cfg, run_dir)
    status_path = run_dir / "run_status.json"
    write_json(status_path, status)
    LOGGER.info("RUN_STARTED | run_id=%s | seed=%d", status["run_id"], cfg["seed"])
    start = time.perf_counter()
    try:
        data = load_task(cfg)
        status["dataset"].update(
            source_sha256=data["source_sha256"],
            excluded_missing_assay=data["excluded_missing_assay"],
            category_codes=data["category_codes"],
            preprocessing=data["encoding"]
            + ("; 80-column one-hot control" if name.endswith("_onehot") else ""),
        )
        write_json(status_path, status)
        LOGGER.info(
            "DATASET_READY | train=%d | test=%d | sha256=%s",
            len(data["train"]["targets"]),
            len(data["test"]["targets"]),
            data["source_sha256"],
        )

        LOGGER.info("TRAINING_STARTED | model=%s", name)
        training_start = time.perf_counter()
        if name in BASELINE_MODELS:
            model, training = fit_baseline(
                cfg, data["train"]["bits"], data["train"]["targets"]
            )
        else:
            model, training = train_model(
                cfg, data["train"]["bits"], data["train"]["targets"]
            )
        train_seconds = time.perf_counter() - training_start

        metrics = {
            "model": name,
            "task": cfg["dataset"]["task"],
            "seed": int(cfg["seed"]),
            "input_features": int(
                baseline_features(name, data["train"]["bits"]).shape[1]
            ),
            "scope": "partial release-split comparison",
            "train_count": len(data["train"]["targets"]),
            "test_count": len(data["test"]["targets"]),
            "train_seconds": train_seconds,
            "source_sha256": data["source_sha256"],
            "parameters": training["parameters"],
            "training": training,
        }
        predictions = {}
        for split in ("train", "test"):
            LOGGER.info("EVALUATION_STARTED | split=%s", split)
            if name in BASELINE_MODELS:
                probabilities = predict_baseline(model, name, data[split]["bits"])
            else:
                probabilities = predict_probabilities(
                    model, cfg, data[split]["bits"]
                ).numpy()
            predictions[split] = probabilities
            metrics[split] = binary_metrics(
                data[split]["targets"].numpy(), probabilities
            )
            LOGGER.info(
                "EVALUATION_COMPLETED | split=%s | f1=%.6f | auroc=%.6f",
                split,
                metrics[split]["f1"],
                metrics[split]["auroc"],
            )
        write_predictions(run_dir / "predictions.csv", predictions, data)

        if name in BASELINE_MODELS:
            # Pickled scikit-learn models: reload only trusted local runs.
            with (run_dir / "model.pkl").open("wb") as handle:
                pickle.dump(model, handle)
        else:
            torch.save(model.state_dict(), run_dir / "model.pt")
            write_losses(run_dir / "losses.csv", training["losses"])

        if name in PHOTONIC_MODELS:
            metrics["photonic"] = _photonic_metadata(model, cfg)
            metrics["finite_shots"] = _finite_shot_metrics(model, cfg, data)

        metrics["wall_seconds"] = time.perf_counter() - start
        write_json(run_dir / "metrics.json", metrics)
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(run_dir / "metrics.json"),
        )
        write_json(status_path, status)
        LOGGER.info("RUN_COMPLETED | run_id=%s", status["run_id"])
    except Exception as error:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        write_json(status_path, status)
        LOGGER.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise
