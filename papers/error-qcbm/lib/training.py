"""Auditable finite-shot QCBM campaigns using native MerLin sampling and SPSA."""

import csv
import gc
import importlib.metadata
import logging
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from runtime_lib.data_paths import paper_data_dir
from runtime_lib.dtypes import dtype_torch

from .artifacts import planned_objective_calls, write_json
from .backend import PhotonicBackend, initialize_parameters
from .data import (
    collision_free_states,
    financial_target,
    gaussian_target,
    save_financial_target,
)
from .recycling import recycling_mitigation, recycling_provenance
from .spsa import SPSA, distribution_loss


def conditional_distribution(distribution: dict, states: tuple) -> np.ndarray:
    """Normalize full probabilities or counts on a fixed collision-free support.

    Parameters
    ----------
    distribution : dict
        Tuple-keyed probabilities or counts, including other sectors.
    states : tuple
        Fixed, complete collision-free state order.

    Returns
    -------
    numpy.ndarray
        Conditional distribution in target bin order.
    """
    values = np.array([distribution.get(state, 0) for state in states], dtype=float)
    if values.sum() <= 0:
        raise ValueError(
            "Empty collision-free sector; no extra shots or fallback allowed"
        )
    return values / values.sum()


class FiniteShotObjective:
    """Evaluate a loss from native MerLin counts with a sampling ledger.

    Parameters
    ----------
    backend : PhotonicBackend
        Native MerLin finite-shot backend.
    states : tuple
        Fixed collision-free bin order.
    target : numpy.ndarray
        Normalized target probabilities.
    estimator : str
        postselection or recycling.
    shots : int
        Initial attempts for every objective call.
    sampling_seed : int
        First Torch seed; each call advances it by one.
    metric : str
        tvd or kl. Default value is tvd for financial benchmark compatibility.
    """

    def __init__(
        self, backend, states, target, estimator, shots, sampling_seed, metric="tvd"
    ):
        self.backend = backend
        self.states = states
        self.target = target
        self.estimator = estimator
        self.shots = shots
        self.first_seed = sampling_seed
        self.metric = metric
        self.calls = 0
        self.sampling_seconds = 0.0
        self.last_counts = {}
        self.last_diagnostics = {}
        if estimator not in ("postselection", "recycling"):
            raise ValueError("Estimator must be postselection or recycling")

    def __call__(self, parameters):
        self.backend.set_parameters(parameters)
        sampling_seed = self.first_seed + self.calls
        counts = self.backend.sample(self.shots, sampling_seed)
        self.calls += 1
        self.sampling_seconds += self.backend.timings["sample_seconds"]
        self.last_counts = counts
        sectors = {
            str(k): sum(count for state, count in counts.items() if sum(state) == k)
            for k in range(sum(self.backend.input_state) + 1)
        }
        diagnostics = {"sampling_seed": sampling_seed, "click_counts": sectors}
        self.last_diagnostics = diagnostics
        if sum(counts.values()) != self.shots:
            raise RuntimeError(
                "Native sampler did not preserve the initial shot budget"
            )
        if self.estimator == "recycling":
            result = recycling_mitigation(
                counts, len(self.backend.input_state), sum(self.backend.input_state)
            )
            if result["states"] != self.states:
                raise RuntimeError("Recycling bin order differs from target order")
            probabilities = result["probabilities"]
            diagnostics.update(result["diagnostics"])
        else:
            probabilities = conditional_distribution(counts, self.states)
        loss = distribution_loss(probabilities, self.target, self.metric)
        if not np.isfinite(loss):
            raise RuntimeError(
                f"Nonfinite {self.metric} objective; no smoothing applied"
            )
        self.last_probabilities = probabilities
        self.last_diagnostics = diagnostics
        return loss


def _train_instance(cfg, run_dir, seed, case, initial, states, target, campaign):
    settings = cfg["training"]
    directory = run_dir / f"seed_{seed:02d}" / case["name"]
    directory.mkdir(parents=True, exist_ok=False)
    profile = {
        "name": case["name"],
        "transmittance": case["transmittance"],
        "indistinguishability": case["indistinguishability"],
        "g2": 0.0,
        "detector": "threshold",
    }
    backend = PhotonicBackend(
        cfg["model"]["input_state"],
        profile,
        dtype=dtype_torch(cfg["dtype"]),
        device=cfg["device"],
    )
    evaluation_backend = PhotonicBackend(
        cfg["model"]["input_state"],
        {**profile, "name": "lossless_independent_evaluation", "transmittance": 1.0},
        dtype=dtype_torch(cfg["dtype"]),
        device=cfg["device"],
    )
    sampling_seed = settings["sampling_seed_offset"] + seed * 1_000_000
    objective = FiniteShotObjective(
        backend,
        states,
        target,
        case["estimator"],
        settings["shots"],
        sampling_seed,
        metric=settings["loss"],
    )
    report = {
        "status": "running",
        "seed": seed,
        "condition": case,
        "completed_steps": 0,
        "objective_calls": 0,
        "initial_shots_per_call": settings["shots"],
        "sampling_seed_first": sampling_seed,
        "perturbation_seed": settings["perturbation_seed_offset"] + seed,
        "sampling_engine": "merlin_native_QuantumLayer_shots",
        "loss_name": settings["loss"],
        "history": [],
        "qpu_jobs_submitted": 0,
        "planned_objective_calls": planned_objective_calls(
            settings["steps"], settings["record_every"]
        ),
    }
    if case["estimator"] == "recycling":
        report["recycling_implementation"] = recycling_provenance()
    np.savez(directory / "initial_parameters.npz", parameters=initial)
    write_json(directory / "parameter_names.json", {"names": backend.parameter_names})
    write_json(directory / "run_status.json", report)
    started = perf_counter()
    optimizer = None
    try:
        initial_loss = objective(initial)
        initial_counts = dict(objective.last_counts)
        initial_probabilities = objective.last_probabilities.copy()
        initial_diagnostics = dict(objective.last_diagnostics)
        optimizer = SPSA(
            initial,
            objective,
            settings["spsa"],
            settings["steps"],
            report["perturbation_seed"],
        )
        report["calibrated_gain"] = optimizer.gain
        report["calibration_magnitude"] = optimizer.calibration_magnitude
        history_path = directory / "losses.csv"
        with history_path.open("w", encoding="utf-8", newline="") as stream:
            fields = [
                "step",
                "finite_shot_loss",
                "exact_lossless_loss",
                "finite_shot_tvd",
                "exact_lossless_tvd",
                "objective_calls",
                "update_norm",
                "parameter_change_norm",
                "elapsed_seconds",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()

            def record(step, loss, update_norm, probabilities=None):
                evaluation_backend.set_parameters(optimizer.parameters)
                exact = conditional_distribution(evaluation_backend.exact(), states)
                if probabilities is None:
                    probabilities = objective.last_probabilities
                row = {
                    "step": step,
                    "finite_shot_loss": loss,
                    "exact_lossless_loss": distribution_loss(
                        exact, target, settings["loss"]
                    ),
                    "finite_shot_tvd": distribution_loss(probabilities, target, "tvd"),
                    "exact_lossless_tvd": distribution_loss(exact, target, "tvd"),
                    "objective_calls": objective.calls,
                    "update_norm": update_norm,
                    "parameter_change_norm": float(
                        np.linalg.norm(optimizer.parameters - initial)
                    ),
                    "elapsed_seconds": perf_counter() - started,
                }
                report["history"].append(row)
                report.update(
                    completed_steps=step,
                    objective_calls=objective.calls,
                    sampling_seconds=objective.sampling_seconds,
                    last_sampling_diagnostics=initial_diagnostics
                    if step == 0
                    else objective.last_diagnostics,
                )
                writer.writerow(row)
                stream.flush()
                np.savez(
                    directory / "checkpoint.npz",
                    **optimizer.checkpoint(),
                    objective_calls=objective.calls,
                    next_sampling_seed=objective.first_seed + objective.calls,
                    initial_parameters=initial,
                )
                write_json(directory / "run_status.json", report)
                campaign["active"] = {"seed": seed, "condition": case["name"], **row}
                write_json(run_dir / "campaign.json", campaign)
                logging.getLogger(__name__).info(
                    "seed=%s %s step=%s/%s sampled %s=%.5f exact=%.5f phase change=%.5g",
                    seed,
                    case["name"],
                    step,
                    settings["steps"],
                    settings["loss"],
                    loss,
                    row["exact_lossless_loss"],
                    row["parameter_change_norm"],
                )
                return exact

            record(0, initial_loss, 0.0, initial_probabilities)
            for step in range(1, settings["steps"] + 1):
                update = optimizer.step()
                if step % settings["record_every"] == 0 or step == settings["steps"]:
                    loss = objective(optimizer.parameters)
                    exact = record(step, loss, update["update_norm"])
            if np.linalg.norm(optimizer.parameters - initial) <= 1e-12:
                raise RuntimeError(
                    "Smoke/update gate failed: no phase parameter changed"
                )
            np.savez(
                directory / "final.npz",
                parameters=optimizer.parameters,
                exact_probabilities=exact,
                sampled_probabilities=objective.last_probabilities,
                target=target,
                states=np.asarray(states),
                initial_count_states=np.asarray(list(initial_counts)),
                initial_counts=np.asarray(list(initial_counts.values())),
                final_count_states=np.asarray(list(objective.last_counts)),
                final_counts=np.asarray(list(objective.last_counts.values())),
            )
        report.update(
            status="complete",
            elapsed_seconds=perf_counter() - started,
            initial_exact_tvd=report["history"][0]["exact_lossless_tvd"],
            final_exact_tvd=report["history"][-1]["exact_lossless_tvd"],
            initial_exact_loss=report["history"][0]["exact_lossless_loss"],
            final_exact_loss=report["history"][-1]["exact_lossless_loss"],
            total_initial_shots=objective.calls * settings["shots"],
            all_parameters_finite=bool(np.isfinite(optimizer.parameters).all()),
        )
        write_json(directory / "run_status.json", report)
        return report
    except Exception as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            objective_calls=objective.calls,
            elapsed_seconds=perf_counter() - started,
        )
        if optimizer is not None:
            np.savez(
                directory / "failed_checkpoint.npz",
                **optimizer.checkpoint(),
                objective_calls=objective.calls,
            )
        write_json(directory / "run_status.json", report)
        raise


def run_training(cfg: dict, run_dir: Path) -> None:
    """Run a Gaussian or financial smoke or paired multi-seed campaign.

    Parameters
    ----------
    cfg : dict
        Runtime configuration with explicit dataset, training and model fields.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Raises
    ------
    ValueError
        If required settings are missing or invalid.
    RuntimeError
        If any seed/condition fails; all failures remain in campaign.json.
    """
    required = ("dataset", "training", "model", "dtype", "device", "data_root")
    if any(key not in cfg for key in required):
        raise ValueError(
            "Training requires explicit dataset, training, model, dtype, device and data_root"
        )
    settings = cfg["training"]
    seeds = settings["seeds"]
    cases = settings["conditions"]
    if cfg["dataset"]["name"] not in ("fx", "gaussian"):
        raise ValueError("Dataset must be fx or gaussian")
    if settings["loss"] not in ("tvd", "kl"):
        raise ValueError("Training loss must be tvd or kl")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or not cases
        or settings["steps"] <= 0
        or settings["record_every"] <= 0
        or settings["shots"] <= 0
    ):
        raise ValueError(
            "Training needs distinct seeds and positive steps, record_every and shots"
        )
    if len({case["name"] for case in cases}) != len(cases):
        raise ValueError("Training condition names must be distinct")
    torch.set_num_threads(settings["torch_threads"])
    input_state = tuple(cfg["model"]["input_state"])
    states = collision_free_states(len(input_state), sum(input_state))
    data_dir = paper_data_dir("error-qcbm", data_root=cfg["data_root"])
    if cfg["dataset"]["name"] == "fx":
        target, arrays, metadata = financial_target(
            data_dir / "fx" / cfg["dataset"]["source_file"], cfg["dataset"], states
        )
        save_financial_target(data_dir / "fx", arrays, metadata)
    else:
        target = gaussian_target(
            len(states), cfg["dataset"]["means"], cfg["dataset"]["standard_deviations"]
        )
        arrays = {
            "probabilities": target,
            "states": np.asarray(states),
            "coordinates": np.arange(len(states)),
        }
        metadata = {
            "dataset": "author discrete Gaussian mixture",
            "bins": len(states),
            "means": cfg["dataset"]["means"],
            "standard_deviations": cfg["dataset"]["standard_deviations"],
            "target_definition": "sum of two Gaussian densities evaluated on integer coordinates, then normalized",
            "state_order": "ascending binary lexicographic author output order",
            "author_commit": "d593d209128e403644b9aee595c474ce551391ae",
            "paper_deviations": [
                "MerLin simulation; demo input chosen by user; reproducible seeds are reproduction choices."
            ],
        }
    np.savez(run_dir / "target.npz", **arrays)
    write_json(run_dir / "dataset.json", metadata)
    parameters = initialize_parameters(len(input_state) * (len(input_state) - 1), seeds)
    np.savez(run_dir / "initializations.npz", seeds=seeds, parameters=parameters)
    campaign = {
        "status": "running",
        "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Gaussian_Fig2ab"
        if cfg["dataset"]["name"] == "gaussian"
        else "financial_Fig2cd_ECB_adaptation",
        "loss_name": settings["loss"],
        "seeds": seeds,
        "conditions": cases,
        "steps_per_instance": settings["steps"],
        "initial_shots_per_evaluation": settings["shots"],
        "record_every": settings["record_every"],
        "sampler": "unmodified native MerLin QuantumLayer(shots=N)",
        "dataset": metadata,
        "device": cfg["device"],
        "dtype": str(dtype_torch(cfg["dtype"])),
        "torch_threads": settings["torch_threads"],
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "merlinquantum",
                "perceval-quandela",
                "torch",
                "numpy",
                "scipy",
            )
        },
        "planned_instances": len(seeds) * len(cases),
        "completed_instances": 0,
        "failed_instances": 0,
        "active": None,
        "instances": [],
        "qpu_jobs_submitted": 0,
        "comparison": "paired initial phases, perturbation seeds and shot budgets; independently trained conditions",
        "deviations": metadata["paper_deviations"],
    }
    write_json(run_dir / "campaign.json", campaign)
    started = perf_counter()
    for seed, initial in zip(seeds, parameters, strict=True):
        for case in cases:
            campaign["active"] = {"seed": seed, "condition": case["name"], "step": 0}
            write_json(run_dir / "campaign.json", campaign)
            try:
                report = _train_instance(
                    cfg, run_dir, seed, case, initial, states, target, campaign
                )
                row = {key: value for key, value in report.items() if key != "history"}
                campaign["completed_instances"] += 1
            except Exception as error:
                logging.getLogger(__name__).exception(
                    "Failed seed=%s condition=%s", seed, case["name"]
                )
                row = {
                    "seed": seed,
                    "condition": case,
                    "status": "failed",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                campaign["failed_instances"] += 1
            campaign["instances"].append(row)
            campaign["elapsed_seconds"] = perf_counter() - started
            write_json(run_dir / "campaign.json", campaign)
            gc.collect()
    campaign.update(
        status="failed" if campaign["failed_instances"] else "complete",
        active=None,
        elapsed_seconds=perf_counter() - started,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(run_dir / "campaign.json", campaign)
    if campaign["failed_instances"]:
        raise RuntimeError(
            f"Campaign has {campaign['failed_instances']} failed instances; see campaign.json"
        )
