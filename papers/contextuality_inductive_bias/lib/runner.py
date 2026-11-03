"""Shared-runtime entry point for the contextuality / inductive-bias reproduction.

One call of :func:`train_and_evaluate` performs one unit of work and writes,
in ``run_dir``::

    run.log               lifecycle, per-step progress, evaluations
    run_status.json       structured status (experiment-logging contract)
    config_snapshot.json  resolved configuration (written by the runtime)
    metrics.json          final metrics of the run

plus, for ``experiment.task = "train"`` (one model, one seed, one protocol)::

    history.csv           per-step train objective/NLL and test KL (Fig. 8 curves)
    model_state.pt        final parameters
    predictions.npz       final E[Y_k | x] on train and test, test strategies

and nothing else for ``"theory_lp"`` (experiment E5). Seed, protocol and model
grids are run as separate calls (see ``utils/run_sweep.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import merlin
import numpy as np
import torch

from runtime_lib.dtypes import dtype_torch

from .classical_baselines import (
    BiasedMLP,
    BiasEncodedFourier,
    QuadraticModel,
    UnbiasedMLP,
)
from .data import describe_dataset, load_split
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .photonic_merlin import PhotonicZeroSumModel
from .quantum_models import BiasedQuantumModel, GenericQuantumModel
from .surrogate import FourierSurrogate
from .theory import lemma_lhs_for_eta, max_lemma_lhs, max_noncontextual_eta
from .training import train

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[3]
TASKS = ("train", "theory_lp")
MODEL_KINDS = (
    "biased_quantum",
    "generic_quantum",
    "surrogate",
    "photonic_merlin",
    "bias_fourier",
    "mlp_biased",
    "mlp_unbiased",
    "quadratic",
)


def build_model(
    model_cfg: dict,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate the model class named by ``model.kind``.

    Raises
    ------
    ValueError
        If ``model.kind`` is unknown.
    """
    kind = model_cfg["kind"]
    if kind == "biased_quantum":
        model = BiasedQuantumModel(
            int(model_cfg["layers"]),
            int(model_cfg["blocks"]),
            generator,
            dtype,
            zz_repetitions=int(model_cfg["zz_repetitions"]),
        )
    elif kind == "generic_quantum":
        model = GenericQuantumModel(
            int(model_cfg["layers"]),
            int(model_cfg["blocks"]),
            generator,
            dtype,
            cnot_pattern=model_cfg["cnot_pattern"],
        )
    elif kind == "surrogate":
        model = FourierSurrogate(
            int(model_cfg["spectrum"]),
            generator,
            dtype,
            chunk_size=int(model_cfg["chunk_size"]),
        )
    elif kind == "photonic_merlin":
        n_photons = int(model_cfg["n_photons"])
        input_states = model_cfg["input_states"]
        if str(n_photons) not in input_states:
            raise ValueError(
                f"model.input_states has no entry for n_photons={n_photons}"
            )
        # MerLin draws the mesh phases from the global torch RNG, which
        # train_and_evaluate seeds with cfg["seed"] before calling this.
        model = PhotonicZeroSumModel(
            model_cfg["readout"],
            n_photons,
            input_states[str(n_photons)],
            int(model_cfg["layers"]),
            dtype,
            computation_space=model_cfg["computation_space"],
        )
    elif kind == "bias_fourier":
        model = BiasEncodedFourier(
            int(model_cfg["spectrum"]),
            generator,
            dtype,
            chunk_size=int(model_cfg["chunk_size"]),
        )
    elif kind == "mlp_biased":
        model = BiasedMLP(int(model_cfg["hidden"]), dtype)
    elif kind == "mlp_unbiased":
        model = UnbiasedMLP(int(model_cfg["hidden"]), dtype)
    elif kind == "quadratic":
        model = QuadraticModel(dtype)
    else:
        raise ValueError(f"model.kind must be one of {MODEL_KINDS}, got {kind!r}")
    return model.to(device)


def _model_candidate(model_cfg: dict) -> dict:
    keys = {
        "biased_quantum": ("layers", "blocks", "zz_repetitions"),
        "generic_quantum": ("layers", "blocks", "cnot_pattern"),
        "surrogate": ("spectrum",),
        "photonic_merlin": ("readout", "n_photons", "layers", "computation_space"),
        "bias_fourier": ("spectrum",),
        "mlp_biased": ("hidden",),
        "mlp_unbiased": ("hidden",),
        "quadratic": (),
    }[model_cfg["kind"]]
    return {"kind": model_cfg["kind"], **{key: model_cfg[key] for key in keys}}


def _candidate(cfg: dict) -> dict:
    """Configuration values that identify this run inside a sweep."""
    task = cfg["experiment"]["task"]
    if task == "theory_lp":
        return {"task": task}
    return {
        "task": task,
        "model": _model_candidate(cfg["model"]),
        "protocol": cfg["data"]["protocol"],
        "train_source": cfg["data"]["train"]["source"],
        "train_seed": cfg["data"]["train"]["seed"]
        if cfg["data"]["train"]["source"] == "generated"
        else None,
        "optimizer": cfg["training"]["optimizer"],
        "lr": cfg["training"]["lr"],
        "l2": cfg["training"]["l2"],
        "steps": cfg["training"]["steps"],
    }


def _describe_selection(cfg: dict) -> dict:
    if cfg["experiment"]["task"] == "theory_lp":
        return {
            "name": "LP optima (Lemma 4.1 bound, eta threshold)",
            "direction": "maximize",
            "split": "n/a",
            "checkpoint": "solver optimum",
        }
    return {
        "name": "test_kl (Eqs. (102)-(103))",
        "direction": "minimize",
        "split": "test",
        "checkpoint": "final step, as in Fig. 8 (no early stopping)",
    }


def _ensure_config_snapshot(cfg: dict, run_dir: Path) -> Path:
    """Return the runtime's snapshot, writing one when called outside the runtime."""
    path = run_dir / "config_snapshot.json"
    if not path.exists():
        path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    return path


def _real_dtype(cfg: dict) -> torch.dtype:
    dtype = dtype_torch(cfg["dtype"])
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be float32 or float64, got {cfg['dtype']!r}")
    return dtype


def run_training(cfg: dict, run_dir: Path) -> dict:
    """Train one model on one protocol and seed (experiments E1-E3)."""
    dtype = _real_dtype(cfg)
    device = torch.device(cfg["device"])
    train_split = load_split(cfg, "train")
    test_split = load_split(cfg, "test")
    validation_split = (
        load_split(cfg, "validation") if cfg["data"]["validation"]["enabled"] else None
    )
    logger.info(
        "DATASET_READY | train=%s n=%d scale=%.6f | test n=%d seed=%s scale=%.6f | validation=%s",
        train_split.provenance["source"],
        train_split.size,
        train_split.input_scale,
        test_split.size,
        test_split.provenance["seed"],
        test_split.input_scale,
        validation_split.size if validation_split else "none",
    )
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    model = build_model(cfg["model"], generator, dtype, device)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "MODEL_READY | kind=%s | parameters=%d | device=%s | dtype=%s",
        cfg["model"]["kind"],
        n_parameters,
        device,
        dtype,
    )

    result = train(
        model,
        train_split,
        test_split,
        validation_split,
        cfg["training"],
        cfg["evaluation"],
        run_dir / "history.csv",
    )

    torch.save(model.state_dict(), run_dir / "model_state.pt")
    logger.info("ARTIFACT_WRITTEN | path=model_state.pt")
    with torch.no_grad():
        np.savez(
            run_dir / "predictions.npz",
            train_expectations=model(train_split.features).cpu().numpy(),
            test_expectations=model(test_split.features).cpu().numpy(),
            test_strategies=test_split.strategies,
            test_win_probabilities=test_split.win_probabilities.cpu().numpy(),
        )
    logger.info("ARTIFACT_WRITTEN | path=predictions.npz")
    photonic = None
    if isinstance(model, PhotonicZeroSumModel):
        success = model.postselection_success(test_split.features)
        photonic = {
            "computation_space": "UNBUNCHED (collision-free postselection, renormalised)"
            if model.computation_space == "unbunched"
            else "FOCK (photon-number-resolving, no postselection)",
            "detector_model": "threshold"
            if model.computation_space == "unbunched"
            else "photon-number-resolving",
            "n_photons": model.n_photons,
            "n_modes": 6,
            "input_state": model.input_state,
            "encoding": f"angle (phase) encoding of z(x) on modes 0-5, scale 1, re-uploaded L={model.layers} times",
            "measurement_strategy": f"MeasurementStrategy.probs({model.computation_space.upper()}) + linear readout",
            "readout": model.readout,
            "postselection": "collision-free outcomes (exactly one photon per clicked mode)"
            if model.computation_space == "unbunched"
            else "none",
            "postselection_success_test_mean": float(success.mean()),
            "postselection_success_test_min": float(success.min()),
            "simulator": f"MerLin {merlin.__version__} exact CPU simulation (no sampling, shots = 0)",
            "gradients": "exact autograd through the simulator",
        }
        logger.info(
            "EVALUATION_COMPLETED | split=test | postselection_success_mean=%.6f | postselection_success_min=%.6f",
            photonic["postselection_success_test_mean"],
            photonic["postselection_success_test_min"],
        )
    return {
        "photonic": photonic,
        "task": "train",
        "model": _model_candidate(cfg["model"]),
        "n_parameters": n_parameters,
        "protocol": cfg["data"]["protocol"],
        "data": {
            "train": train_split.provenance,
            "test": test_split.provenance,
            "validation": validation_split.provenance if validation_split else None,
        },
        "seed": cfg["seed"],
        **result,
        "artifacts": {
            "history": "history.csv",
            "model_state": "model_state.pt",
            "predictions": "predictions.npz",
        },
    }


def run_theory_lp(cfg: dict, run_dir: Path) -> dict:
    """Solve the E5 linear programs (Lemma 4.1 bound and eta threshold)."""
    logger.info("DATASET_READY | none (analytic scenario of Sec. 4.3)")
    logger.info(
        "EVALUATION_STARTED | split=n/a | programs=lemma_lhs,lemma_lhs_no_prep_equivalence,eta_threshold"
    )
    lemma = max_lemma_lhs(with_preparation_equivalence=True)
    unconstrained = max_lemma_lhs(with_preparation_equivalence=False)
    eta = max_noncontextual_eta()
    eta_grid = [round(value, 4) for value in np.linspace(0.0, 1.0, 11)]
    metrics = {
        "task": "theory_lp",
        "lemma_lhs_max": lemma["optimum"],
        "lemma_bound_paper": 2.5,
        "lemma_lhs_max_without_preparation_equivalence": unconstrained["optimum"],
        "eta_max_noncontextual": eta["eta_max"],
        "eta_threshold_paper": 2.0 / 3.0,
        "lhs_on_u_eta": {str(value): lemma_lhs_for_eta(value) for value in eta_grid},
        "solver_status": {"lemma": lemma["status"], "eta": eta["status"]},
        "lemma_optimal_weights": lemma["weights"],
    }
    logger.info(
        "EVALUATION_COMPLETED | lemma_lhs_max=%.12f | without_prep_equivalence=%.12f | eta_max=%.12f",
        metrics["lemma_lhs_max"],
        metrics["lemma_lhs_max_without_preparation_equivalence"],
        metrics["eta_max_noncontextual"],
    )
    return metrics


TASK_FUNCTIONS = {"train": run_training, "theory_lp": run_theory_lp}


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run one configured experiment.

    Parameters
    ----------
    cfg : dict
        Resolved configuration. ``experiment.task`` is ``"train"`` or
        ``"theory_lp"``; ``seed`` seeds the parameter initialisation.
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
            f"Unsupported experiment.task={task!r}; implemented tasks: {TASKS}"
        )
    level = getattr(logging, str(cfg["logging"]["level"]).upper())
    configure_logging(run_dir / "run.log", level=level)
    config_path = _ensure_config_snapshot(cfg, run_dir)
    status_path = run_dir / "run_status.json"
    run_id = cfg["experiment"]["run_id"] or run_dir.name
    status = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "sweep_id": cfg["experiment"]["sweep_id"],
        "candidate": _candidate(cfg),
        "seed": cfg["seed"],
        "repetition": cfg["experiment"]["repetition"],
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(_REPO_ROOT),
        "dataset": describe_dataset(cfg)
        if task == "train"
        else {
            "name": "none",
            "splits": "n/a",
            "preprocessing": "none",
            "subset": "n/a",
        },
        "selection": _describe_selection(cfg),
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
        cfg["experiment"]["id"],
        cfg["seed"],
        torch.__version__,
    )
    try:
        torch.manual_seed(int(cfg["seed"]))
        metrics = TASK_FUNCTIONS[task](cfg, run_dir)
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
