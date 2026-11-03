"""Atomic update-boundary checkpoints for long native MerLin campaigns."""

import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from merlin.measurement.process import SamplingProcess

from runtime_lib.dtypes import dtype_torch

from .artifacts import (
    atomic_replace,
    planned_objective_calls,
    summarize_sampling_ledger,
)
from .backend import PhotonicBackend, initialize_parameters
from .data import collision_free_states, gaussian_target
from .noise import load_configured_profiles
from .recycling import recycling_provenance
from .spsa import SPSA, distribution_loss
from .training import FiniteShotObjective, conditional_distribution, write_json


def verify_accumulation_evidence(path):
    """Require complete environment-matched stock/vectorized 50M evidence.

    Parameters
    ----------
    path : pathlib.Path
        Legacy initial/trained campaign audit or fresh v2 diagnostic-probe audit.

    Returns
    -------
    dict
        Immutable evidence fingerprint attached to each execution transition.

    Raises
    ------
    ValueError
        If the environment, source, conditions or count comparisons disagree.
    """
    raw = path.read_bytes()
    evidence = json.loads(raw)
    if evidence.get("schema") == "merlin_count_equivalence_v2":
        from .sampler_validation import validate_full_evidence

        validate_full_evidence(evidence)
        if torch.get_num_threads() != 1:
            raise ValueError(
                "Validated vectorized production sampling requires Torch thread 1"
            )
        return {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "adapter_sha256": evidence["environment"]["adapter_sha256"],
            "schema": evidence["schema"],
        }
    adapter_hash = hashlib.sha256(
        Path(__file__).with_name("sampling.py").read_bytes()
    ).hexdigest()
    sampler_hash = hashlib.sha256(
        Path(inspect.getsourcefile(SamplingProcess)).read_bytes()
    ).hexdigest()
    names = {
        "belenos_postselection",
        "belenos_recycling",
        "lossless_partial",
        "lossless_ideal",
    }
    if evidence["status"] != "passed" or set(evidence["cases"]) != names:
        raise ValueError("Vectorized sampler requires all four passed condition checks")
    for name, row in evidence["cases"].items():
        if not (
            row["status"] == "passed"
            and row["shots"] == 50000000
            and row["native_chunk_shots"] == 1000000
            and row["adapter_sha256"] == adapter_hash
            and row["original_sampler_sha256"] == sampler_hash
            and row["torch_version"] == torch.__version__
            and {r["label"] for r in row["comparisons"]}
            == {"initial_archived_reference", "trained_fresh_reference"}
        ):
            raise ValueError(f"Sampler verification settings/source mismatch: {name}")
        for result in row["comparisons"]:
            if not (
                result["original_shots"] == result["vectorized_shots"] == 50000000
                and result["counts_identical"]
                and result["max_count_difference"] == 0
                and result["probabilities_identical"]
                and result["loss_identical"]
            ):
                raise ValueError(f"Sampler exact comparison failed: {name}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "adapter_sha256": adapter_hash,
    }


def _atomic_npz(path, values):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_replace(temporary, path)


def _save_checkpoint(
    directory, optimizer, objective, report, initial, initial_counts, signature
):
    counts = objective.last_counts
    report.update(
        objective_calls=objective.calls,
        committed_objective_calls=objective.calls,
        committed_initial_shots=objective.calls * objective.shots,
        total_initial_shots=objective.calls * objective.shots,
        shot_accounting=summarize_sampling_ledger(directory / "sampling_ledger.jsonl"),
    )
    _atomic_npz(
        directory / "checkpoint.npz",
        {
            **optimizer.checkpoint(),
            "objective_calls": objective.calls,
            "sampling_seconds": objective.sampling_seconds,
            "next_sampling_seed": objective.first_seed + objective.calls,
            "initial_parameters": initial,
            "initial_count_states": np.asarray(list(initial_counts)),
            "initial_counts": np.asarray(list(initial_counts.values())),
            "last_count_states": np.asarray(list(counts)),
            "last_counts": np.asarray(list(counts.values())),
            "last_probabilities": objective.last_probabilities,
            "report_json": json.dumps(report, allow_nan=False),
            "signature": signature,
        },
    )
    _publish(directory, report)


def _publish(directory, report):
    write_json(directory / "run_status.json", report)
    temporary = directory / "losses.csv.tmp"
    if report["history"]:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report["history"][0]))
            writer.writeheader()
            writer.writerows(report["history"])
        atomic_replace(temporary, directory / "losses.csv")


class AuditedObjective(FiniteShotObjective):
    """Persist every attempted objective call, including replayed interrupted work.

    Parameters
    ----------
    ledger_path : pathlib.Path
        Append-only sampling diagnostics ledger.
    *args, **kwargs
        Arguments passed to FiniteShotObjective.
    """

    def __init__(self, ledger_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger_path = ledger_path

    def __call__(self, parameters):
        call = self.calls
        started = perf_counter()
        row = {
            "call_index": call,
            "sampling_seed": self.first_seed + call,
            "initial_shots": self.shots,
            "status": "started",
        }
        self._append(row)
        try:
            loss = super().__call__(parameters)
        except Exception as error:
            self._append(
                {
                    **row,
                    "status": "failed",
                    "sampling_completed": self.calls > call
                    and sum(self.last_counts.values()) == self.shots,
                    "error": str(error),
                    "seconds": perf_counter() - started,
                    "diagnostics": self.last_diagnostics if self.calls > call else None,
                }
            )
            raise
        self._append(
            {
                **row,
                "status": "complete",
                "sampling_completed": True,
                "loss": loss,
                "seconds": perf_counter() - started,
                "diagnostics": self.last_diagnostics,
                "sample_engine": self.backend.timings["sample_engine"],
            }
        )
        return loss

    def _append(self, row):
        with self.ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()


def train_instance(cfg, directory, seed, case, *, max_updates=None):
    """Train or resume one Gaussian instance at atomic update boundaries.

    Parameters
    ----------
    cfg : dict
        Resolved configuration including a segment_seconds training limit.
    directory : pathlib.Path
        Stable per-seed, per-condition directory, independent of runtime run_dir.
    seed : int
        Paired initialization seed.
    case : dict
        Noise and estimator settings.
    max_updates : int or None
        Test-only segment update limit. Default value is None.

    Returns
    -------
    dict
        Complete, paused or failed state; failed numerical runs raise an error.
    """
    settings = cfg["training"]
    signature_data = {
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "dtype": str(cfg["dtype"]),
        "device": cfg["device"],
        "seed": seed,
        "case": case,
        "training": {
            k: settings[k]
            for k in (
                "steps",
                "record_every",
                "shots",
                "sampling_seed_offset",
                "perturbation_seed_offset",
                "spsa",
                "loss",
                "torch_threads",
            )
        },
        "noise_provenance": cfg.get("noise_provenance"),
        "native_chunk_shots": settings.get("native_chunk_shots"),
    }
    if case["estimator"] == "recycling":
        # A different numerical estimator must not silently continue an old run.
        signature_data["recycling_implementation"] = recycling_provenance()
    # Accumulation is execution provenance, not a scientific-setting change:
    # python -> vectorized is allowed only with exact 50M comparison evidence.
    signature = hashlib.sha256(
        json.dumps(signature_data, sort_keys=True).encode()
    ).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "run_status.json"
    checkpoint_path = directory / "checkpoint.npz"
    previous = None
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError(
                "Resume configuration differs from the saved scientific settings "
                "or recycling implementation; use a separate run directory"
            )
        if previous["status"] == "failed":
            raise RuntimeError(
                "Failed instance requires investigation; automatic numerical retry disabled"
            )
    # Validate persisted metadata before constructing a backend or writing status.
    # A corrupt checkpoint is an input-integrity error, not a new numerical run.
    modes = len(cfg["model"]["input_state"])
    initial = initialize_parameters(modes * (modes - 1), [seed])[0]
    checkpoint_values = None
    restored_optimizer = None
    if checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            checkpoint_values = {
                key: checkpoint[key].copy() for key in checkpoint.files
            }
        if str(checkpoint_values["signature"]) != signature:
            raise ValueError("Checkpoint configuration signature mismatch")
        saved_report = json.loads(str(checkpoint_values["report_json"]))
        if saved_report["signature"] != signature:
            raise ValueError("Checkpoint report signature mismatch")
        if settings["sampling_seed_offset"] + seed * 1_000_000 + int(
            checkpoint_values["objective_calls"]
        ) != int(checkpoint_values["next_sampling_seed"]):
            raise ValueError("Checkpoint sampling ledger mismatch")
        if not np.array_equal(initial, checkpoint_values["initial_parameters"]):
            raise ValueError("Checkpoint initialization mismatch")
        step = int(checkpoint_values["step"])
        if (
            not 0 <= step <= settings["steps"]
            or saved_report["completed_steps"] != step
        ):
            raise ValueError("Checkpoint update count mismatch")
        if (
            checkpoint_values["parameters"].shape != initial.shape
            or not np.isfinite(checkpoint_values["parameters"]).all()
        ):
            raise ValueError("Checkpoint parameters are invalid")
        restored_optimizer = SPSA.restore(checkpoint_values, None, settings["spsa"])
    elif previous is not None and (
        previous["status"] in ("complete", "paused") or previous["completed_steps"] > 0
    ):
        raise ValueError("Saved run status has no committed checkpoint")
    if previous is not None and previous["status"] == "complete":
        final_path = directory / "final.npz"
        if not final_path.exists():
            raise ValueError("Completed status has no final checkpoint")
        if saved_report["status"] != "complete" or step != settings["steps"]:
            raise ValueError("Completed status disagrees with the committed checkpoint")
        with np.load(final_path, allow_pickle=False) as final:
            if not np.array_equal(final["parameters"], checkpoint_values["parameters"]):
                raise ValueError(
                    "Final parameters disagree with the committed checkpoint"
                )
        return previous
    if summarize_sampling_ledger(directory / "sampling_ledger.jsonl")[
        "incomplete_trailing_line"
    ]:
        raise ValueError(
            "Incomplete sampling ledger tail; preserve and repair it before resuming"
        )
    profile = {**case, "g2": 0.0, "detector": "threshold"}
    backend = PhotonicBackend(
        cfg["model"]["input_state"],
        profile,
        dtype=dtype_torch(cfg["dtype"]),
        device=cfg["device"],
        native_chunk_shots=settings.get("native_chunk_shots"),
        native_count_accumulation=settings.get("native_count_accumulation", "python"),
    )
    evaluation = PhotonicBackend(
        cfg["model"]["input_state"],
        {**profile, "transmittance": 1.0},
        dtype=dtype_torch(cfg["dtype"]),
        device=cfg["device"],
    )
    states = collision_free_states(len(backend.input_state), sum(backend.input_state))
    target = gaussian_target(
        len(states), cfg["dataset"]["means"], cfg["dataset"]["standard_deviations"]
    )
    objective = AuditedObjective(
        directory / "sampling_ledger.jsonl",
        backend,
        states,
        target,
        case["estimator"],
        settings["shots"],
        settings["sampling_seed_offset"] + seed * 1_000_000,
        metric=settings["loss"],
    )
    report = {
        "status": "running",
        "seed": seed,
        "condition": case,
        "signature": signature,
        "history": [],
        "completed_steps": 0,
        "initial_shots_per_call": settings["shots"],
        "loss_name": settings["loss"],
        "sampling_engine": "merlin_native_QuantumLayer_shots",
        "native_count_accumulation": settings.get(
            "native_count_accumulation", "python"
        ),
        "native_chunk_shots": settings.get("native_chunk_shots"),
        "qpu_jobs_submitted": 0,
        "noise_provenance": cfg.get("noise_provenance"),
        "elapsed_seconds": 0.0,
        "evaluation_scope": "lossless at SAME indistinguishability as training condition",
        "planned_objective_calls": planned_objective_calls(
            settings["steps"], settings["record_every"]
        ),
    }
    if case["estimator"] == "recycling":
        report["recycling_implementation"] = recycling_provenance()
    started = perf_counter()
    previous_elapsed = 0.0

    def sampler_provenance():
        mode = settings.get("native_count_accumulation", "python")
        previous = report.get("native_count_accumulation", "python")
        history = report.setdefault("sampler_execution_history", [])
        if not history:
            history.append({"mode": previous, "at_step": 0, "objective_calls": 0})
            if previous == "vectorized":
                history[-1]["verification"] = cfg["_vectorized_validation"]
        if mode != previous:
            proof = cfg.get("_vectorized_validation")
            if mode != "vectorized" or proof is None:
                raise ValueError(
                    "Sampler execution changes require verified 50M equivalence evidence"
                )
            history.append(
                {
                    "mode": mode,
                    "at_step": optimizer.step_number,
                    "objective_calls": objective.calls,
                    "verification": proof,
                }
            )
        report["native_count_accumulation"] = mode
        report["sampling_engine"] = (
            "merlin_native_multinomial_vectorized_counts"
            if mode == "vectorized"
            else "merlin_native_QuantumLayer_shots"
        )

    def record(step, loss, update_norm, probabilities=None):
        evaluation.set_parameters(optimizer.parameters)
        exact = conditional_distribution(evaluation.exact(), states)
        probabilities = (
            objective.last_probabilities if probabilities is None else probabilities
        )
        report["history"].append(
            {
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
            }
        )
        return exact

    try:
        if checkpoint_values is not None:
            checkpoint = checkpoint_values
            optimizer = restored_optimizer
            optimizer.objective = objective
            report = json.loads(str(checkpoint["report_json"]))
            previous_elapsed = report["elapsed_seconds"]
            objective.calls = int(checkpoint["objective_calls"])
            objective.sampling_seconds = float(checkpoint["sampling_seconds"])
            objective.last_probabilities = checkpoint["last_probabilities"].copy()
            objective.last_counts = {
                tuple(s): int(c)
                for s, c in zip(
                    checkpoint["last_count_states"],
                    checkpoint["last_counts"],
                    strict=True,
                )
            }
            initial_counts = {
                tuple(s): int(c)
                for s, c in zip(
                    checkpoint["initial_count_states"],
                    checkpoint["initial_counts"],
                    strict=True,
                )
            }
            report["status"] = "running"
            report["planned_objective_calls"] = planned_objective_calls(
                settings["steps"], settings["record_every"]
            )
            sampler_provenance()
            _publish(directory, report)
        else:
            _publish(directory, report)
            initial_loss = objective(initial)
            initial_counts = objective.last_counts.copy()
            initial_probabilities = objective.last_probabilities.copy()
            optimizer = SPSA(
                initial,
                objective,
                settings["spsa"],
                settings["steps"],
                settings["perturbation_seed_offset"] + seed,
            )
            report["calibrated_gain"] = optimizer.gain
            report["calibration_magnitude"] = optimizer.calibration_magnitude
            record(0, initial_loss, 0.0, initial_probabilities)
            sampler_provenance()
            _save_checkpoint(
                directory,
                optimizer,
                objective,
                report,
                initial,
                initial_counts,
                signature,
            )
        segment_updates = 0
        while optimizer.step_number < settings["steps"]:
            if perf_counter() - started >= settings["segment_seconds"] or (
                max_updates is not None and segment_updates >= max_updates
            ):
                report.update(
                    status="paused",
                    elapsed_seconds=previous_elapsed + perf_counter() - started,
                )
                _save_checkpoint(
                    directory,
                    optimizer,
                    objective,
                    report,
                    initial,
                    initial_counts,
                    signature,
                )
                return report
            update = optimizer.step()
            if (
                optimizer.step_number % settings["record_every"] == 0
                or optimizer.step_number == settings["steps"]
            ):
                loss = objective(optimizer.parameters)
                record(optimizer.step_number, loss, update["update_norm"])
            report.update(
                completed_steps=optimizer.step_number,
                objective_calls=objective.calls,
                sampling_seconds=objective.sampling_seconds,
                elapsed_seconds=previous_elapsed + perf_counter() - started,
            )
            _save_checkpoint(
                directory,
                optimizer,
                objective,
                report,
                initial,
                initial_counts,
                signature,
            )
            segment_updates += 1
        if np.linalg.norm(optimizer.parameters - initial) <= 1e-12:
            raise RuntimeError("Smoke/update gate failed: no parameter changed")
        evaluation.set_parameters(optimizer.parameters)
        exact = conditional_distribution(evaluation.exact(), states)
        _atomic_npz(
            directory / "final.npz",
            {
                "parameters": optimizer.parameters,
                "initial_parameters": initial,
                "target": target,
                "states": np.asarray(states),
                "exact_probabilities": exact,
                "sampled_probabilities": objective.last_probabilities,
                "initial_count_states": np.asarray(list(initial_counts)),
                "initial_counts": np.asarray(list(initial_counts.values())),
                "final_count_states": np.asarray(list(objective.last_counts)),
                "final_counts": np.asarray(list(objective.last_counts.values())),
            },
        )
        report.update(
            status="complete",
            total_initial_shots=objective.calls * settings["shots"],
            all_parameters_finite=bool(np.isfinite(optimizer.parameters).all()),
        )
        _save_checkpoint(
            directory, optimizer, objective, report, initial, initial_counts, signature
        )
        return report
    except Exception as error:
        report.update(
            status="interrupted" if isinstance(error, OSError) else "failed",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=previous_elapsed + perf_counter() - started,
            objective_calls=objective.calls,
            shot_accounting=summarize_sampling_ledger(
                directory / "sampling_ledger.jsonl"
            ),
        )
        with (directory / "execution_events.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    {
                        key: report[key]
                        for key in (
                            "status",
                            "error_type",
                            "error",
                            "completed_steps",
                            "objective_calls",
                            "elapsed_seconds",
                        )
                    }
                )
                + "\n"
            )
            stream.flush()
        _publish(directory, report)
        raise


def run_resumable_training(cfg, run_dir):
    """Run explicitly configured resumable Gaussian training via the shared runtime.

    Parameters
    ----------
    cfg : dict
        Resolved runtime configuration with a stable training.state_dir.
    run_dir : pathlib.Path
        Timestamped invocation directory for the shared runtime.
    """
    if cfg["dataset"]["name"] != "gaussian" or cfg["training"]["loss"] != "kl":
        raise ValueError("Resumable Belenos runner requires Gaussian KL")
    if str(cfg["device"]) != "cpu" or dtype_torch(cfg["dtype"]) != torch.float64:
        raise ValueError("Belenos native 50M training requires CPU float64")
    settings = cfg["training"]
    if any(
        settings[key] <= 0
        for key in ("steps", "record_every", "shots", "segment_seconds")
    ):
        raise ValueError(
            "Steps, shots, recording interval and segment duration must be positive"
        )
    torch.set_num_threads(settings["torch_threads"])
    if settings.get("native_count_accumulation", "python") == "vectorized":
        proof_path = Path(settings["vectorized_validation_file"])
        if not proof_path.is_absolute():
            proof_path = Path(__file__).resolve().parents[1] / proof_path
        proof = verify_accumulation_evidence(proof_path)
        cfg = {**cfg, "_vectorized_validation": proof}
    profiles, provenance = load_configured_profiles(cfg)
    expected = {
        "belenos_postselection": (
            profiles["belenos_partial"]["transmittance"],
            profiles["belenos_partial"]["indistinguishability"],
            "postselection",
        ),
        "belenos_recycling": (
            profiles["belenos_partial"]["transmittance"],
            profiles["belenos_partial"]["indistinguishability"],
            "recycling",
        ),
        "belenos_loss_recycling": (
            profiles["belenos_loss"]["transmittance"],
            1.0,
            "recycling",
        ),
        "belenos_overlap084_recycling": (
            profiles["belenos_partial"]["transmittance"],
            0.84,
            "recycling",
        ),
        "lossless_partial": (
            1.0,
            profiles["belenos_partial"]["indistinguishability"],
            "postselection",
        ),
        "lossless_overlap084": (1.0, 0.84, "postselection"),
        "lossless_ideal": (1.0, 1.0, "postselection"),
    }
    for case in settings["conditions"]:
        transmission, indistinguishability, estimator = expected[case["name"]]
        if not (
            np.isclose(case["transmittance"], transmission, rtol=0, atol=1e-15)
            and np.isclose(
                case["indistinguishability"], indistinguishability, rtol=0, atol=1e-15
            )
            and case["estimator"] == estimator
        ):
            raise ValueError(
                "Belenos conditions disagree with the frozen characterization"
            )
    if any(
        case["name"] in ("belenos_overlap084_recycling", "lossless_overlap084")
        for case in settings["conditions"]
    ):
        provenance = {
            **provenance,
            "indistinguishability_override": {
                "value": 0.84,
                "source": "QCBM experimental supplement: 0.84 +/- 0.03",
                "interpretation": "Chosen two-photon overlap model input; measurement convention unconfirmed",
                "replaces_hub_hom": profiles["belenos_partial"]["indistinguishability"],
            },
            "scope": (
                "Paper-motivated overlap override with lossless transmission; not a Belenos HOM characterization"
                if all(
                    case["name"] == "lossless_overlap084"
                    for case in settings["conditions"]
                )
                else "Belenos total transmission with paper-motivated overlap override; not a Belenos HOM characterization"
            ),
        }
    cfg = {**cfg, "noise_provenance": provenance}
    state_root = Path(settings["state_dir"])
    if not state_root.is_absolute():
        state_root = Path(__file__).resolve().parents[1] / state_root
    reports = []
    for seed in settings["seeds"]:
        for case in settings["conditions"]:
            reports.append(
                train_instance(
                    cfg, state_root / f"seed_{seed:02d}" / case["name"], seed, case
                )
            )
    write_json(
        run_dir / "campaign.json",
        {
            "status": "complete"
            if all(r["status"] == "complete" for r in reports)
            else "paused",
            "state_dir": str(state_root),
            "instances": reports,
            "qpu_jobs_submitted": 0,
            "noise_provenance": provenance,
        },
    )
