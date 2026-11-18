"""Fixed-circuit theory checks and native-sampling diagnostics."""

import gc
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from runtime_lib.data_paths import paper_data_dir
from runtime_lib.dtypes import dtype_torch

from .backend import PhotonicBackend, initialize_parameters
from .data import collision_free_states, gaussian_target
from .noise import load_configured_profiles
from .recycling import recycling_mitigation
from .theory import theoretical_distribution, total_variation


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _conditional(distribution, states):
    values = np.array([distribution.get(state, 0.0) for state in states])
    if values.sum() <= 0:
        raise ValueError("No collision-free probability mass")
    return values / values.sum()


def _sampling_bound(support, shots, comparisons):
    # E[TVD] <= .5*sqrt(K/N); changing one sample changes TVD by at most 1/N.
    return min(
        1.0,
        0.5 * math.sqrt(support / shots)
        + math.sqrt(math.log(comparisons / 1e-6) / (2 * shots)),
    )


def run_phase_one(cfg: dict, run_dir: Path) -> None:
    """Validate the paper circuit and frozen Belenos noise ablation.

    Parameters
    ----------
    cfg : dict
        Runtime-resolved model, validation, data-root and precision settings.
    run_dir : pathlib.Path
        Timestamped output directory. Contains artifacts even on failure.

    Raises
    ------
    RuntimeError
        If a backend/theory/sampling check fails, or any required sector is empty.
    """
    logger = logging.getLogger(__name__)
    settings = cfg["validation"]
    input_state = tuple(cfg["model"]["input_state"])
    seeds = settings["seeds"]
    if not seeds or settings["native_shots"] <= 0:
        raise ValueError("Seeds and positive native_shots are required")
    if settings["profiles"][0] != "ideal":
        raise ValueError(
            "Validate ideal first to establish a paired physical reference"
        )
    if not set(seeds).issubset(settings["theory_seeds"]):
        raise ValueError("Every validation seed must have an independent theory check")
    torch.set_num_threads(settings["torch_threads"])
    data_dir = paper_data_dir("error-qcbm", data_root=cfg["data_root"])
    profiles, provenance = load_configured_profiles(cfg, settings.get("snapshot_file"))
    states = collision_free_states(len(input_state), sum(input_state))
    target = gaussian_target(len(states))
    dataset_dir = data_dir / "gaussian"
    dataset_dir.mkdir(exist_ok=True)
    np.savez(
        dataset_dir / "target.npz",
        coordinates=np.arange(len(states)),
        probabilities=target,
        states=np.array(states),
    )
    metadata = {
        "source": "synthetic author gaussian_mixture_pdf; no external dataset download",
        "bins": len(states),
        "means": [2 * len(states) // 7, 5 * len(states) // 7],
        "standard_deviations": [len(states) // 8] * 2,
        "training_representation": "known normalized probability vector, not finite training samples",
        "state_order": "ascending author mixed-radix integer, equivalent to binary lexicographic",
    }
    _write_json(dataset_dir / "metadata.json", metadata)
    parameter_count = len(input_state) * (len(input_state) - 1)
    initializations = initialize_parameters(parameter_count, seeds)
    np.savez(run_dir / "initializations.npz", seeds=seeds, parameters=initializations)
    report = {
        "status": "running",
        "scope": "phase_one_fixed_circuit_validation_no_training",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_state": input_state,
        "seeds": seeds,
        "dataset": metadata,
        "trainable_parameters": parameter_count,
        "parameter_identifiability": "nominal trainable count, not independent identifiable probability degrees of freedom",
        "noise_provenance": provenance,
        "profiles": profiles,
        "device": cfg["device"],
        "dtype": str(dtype_torch(cfg["dtype"])),
        "torch_threads": settings["torch_threads"],
        "records": [],
        "failures": [],
        "qpu_jobs_submitted": 0,
    }
    tolerance = settings["probability_atol"]
    reference_by_seed = {}
    figure_payload = {}
    for profile_name in settings["profiles"]:
        profile = profiles[profile_name]
        started = perf_counter()
        backend = PhotonicBackend(
            input_state, profile, dtype=dtype_torch(cfg["dtype"]), device=cfg["device"]
        )
        construction_seconds = perf_counter() - started
        if len(backend.parameter_names) != parameter_count:
            raise RuntimeError("Trainable parameter count differs from the paper")
        _write_json(run_dir / "parameter_names.json", backend.parameter_names)
        shots = settings["shots"][profile_name]
        logger.info("Validating %s: %s seeds, N=%s", profile_name, len(seeds), shots)
        profile_records = []
        for seed, vector in zip(seeds, initializations, strict=True):
            backend.set_parameters(vector)
            exact = backend.exact()
            exact_seconds = backend.timings["exact_seconds"]
            unitary = np.array(backend.circuit.compute_unitary(), dtype=complex)
            unitarity_error = float(
                np.max(np.abs(unitary.conj().T @ unitary - np.eye(len(input_state))))
            )
            if unitarity_error > tolerance:
                raise RuntimeError("Initialized circuit is not unitary")
            theory_error = None
            theory_conditional_error = None
            theory_seconds = None
            if seed in settings["theory_seeds"]:
                started = perf_counter()
                theory = theoretical_distribution(
                    unitary,
                    input_state,
                    profile["transmittance"],
                    profile["indistinguishability"],
                )
                theory_seconds = perf_counter() - started
                theory_error = max(
                    abs(exact.get(key, 0) - theory.get(key, 0))
                    for key in exact.keys() | theory.keys()
                )
                if theory_error > tolerance:
                    raise RuntimeError(
                        f"Independent permutation reference mismatch for {profile_name}: {theory_error}"
                    )
                # The four-click sector is extremely rare at Belenos loss.
                # An absolute error check alone would miss an incorrect sector.
                theory_conditional_error = float(
                    np.max(
                        np.abs(
                            _conditional(exact, states) - _conditional(theory, states)
                        )
                    )
                )
                if theory_conditional_error > tolerance:
                    raise RuntimeError(
                        f"Conditional four-click reference mismatch: {theory_conditional_error}"
                    )
            counts = backend.sample(shots, settings["sampling_seed_offset"] + int(seed))
            empirical = {key: count / shots for key, count in counts.items()}
            distance = total_variation(exact, empirical)
            bound = _sampling_bound(
                sum(value > 0 for value in exact.values()),
                shots,
                len(seeds) * len(settings["profiles"]),
            )
            if distance > bound:
                raise RuntimeError(
                    f"Sampling TVD exceeds concentration bound: {distance}>{bound}"
                )
            expected_conditional = _conditional(exact, states)
            record = {
                "profile": profile_name,
                "seed": seed,
                "initial_shots": shots,
                "observed_shots": sum(counts.values()),
                "exact_seconds": exact_seconds,
                "construction_seconds": construction_seconds
                if seed == seeds[0]
                else None,
                "sample_with_forward_seconds": backend.timings["sample_seconds"],
                "sample_engine": backend.timings["sample_engine"],
                "sampling_tvd": distance,
                "sampling_tvd_bound": bound,
                "theory_max_probability_error": theory_error,
                "theory_conditioned_max_probability_error": theory_conditional_error,
                "theory_seconds": theory_seconds,
                "unitarity_error": unitarity_error,
                "click_sector_counts": {
                    str(k): sum(
                        value for state, value in counts.items() if sum(state) == k
                    )
                    for k in range(sum(input_state) + 1)
                },
                "expected_full_survivors_before_collision": shots
                * profile["transmittance"] ** sum(input_state),
            }
            if seed == seeds[0]:
                native = backend.sample_native(
                    settings["native_shots"], settings["sampling_seed_offset"] + seed
                )
                native_empirical = {
                    key: value / settings["native_shots"]
                    for key, value in native.items()
                }
                record["native_shots"] = settings["native_shots"]
                record["native_sample_with_forward_seconds"] = backend.timings[
                    "native_sample_seconds"
                ]
                record["native_sampling_tvd"] = total_variation(exact, native_empirical)
                if record["native_sampling_tvd"] > _sampling_bound(
                    len(exact), settings["native_shots"], len(settings["profiles"])
                ):
                    raise RuntimeError(
                        "Native MerLin sampling failed its concentration check"
                    )
            if profile_name == "ideal":
                reference_by_seed[seed] = expected_conditional
            else:
                if profile["indistinguishability"] == 1.0:
                    loss_invariance_error = float(
                        np.max(np.abs(expected_conditional - reference_by_seed[seed]))
                    )
                    record["conditional_uniform_loss_invariance_error"] = (
                        loss_invariance_error
                    )
                    if loss_invariance_error > tolerance:
                        raise RuntimeError(
                            "Uniform loss changed the conditional no-loss distribution"
                        )
                started = perf_counter()
                try:
                    recycled = recycling_mitigation(
                        counts, len(input_state), sum(input_state)
                    )
                except (ValueError, RuntimeError) as error:
                    record["recycling_error"] = str(error)
                    report["failures"].append(
                        {"profile": profile_name, "seed": seed, "reason": str(error)}
                    )
                else:
                    record["recycling_seconds"] = perf_counter() - started
                    record["postselection_tvd"] = float(
                        np.abs(recycled["postselected"] - expected_conditional).sum()
                        / 2
                    )
                    record["recycling_tvd"] = float(
                        np.abs(recycled["probabilities"] - expected_conditional).sum()
                        / 2
                    )
                    record["recycling_tvd_to_indistinguishable"] = float(
                        np.abs(
                            recycled["probabilities"] - reference_by_seed[seed]
                        ).sum()
                        / 2
                    )
                    record["recycling_diagnostics"] = recycled["diagnostics"]
                    if seed == seeds[0]:
                        figure_payload[profile_name] = (
                            expected_conditional,
                            recycled["postselected"],
                            recycled["probabilities"],
                        )
                    np.savez(
                        run_dir / f"recycling_{profile_name}_seed{seed}.npz",
                        reference=expected_conditional,
                        postselected=recycled["postselected"],
                        mitigated=recycled["probabilities"],
                        sector_counts=np.asarray(recycled["sector_counts"]),
                    )
            _write_json(
                run_dir / f"counts_{profile_name}_seed{seed}.json",
                [
                    {"state": state, "count": count}
                    for state, count in sorted(counts.items())
                ],
            )
            profile_records.append(record)
            report["records"].append(record)
            _write_json(run_dir / "phase1_report.json", report)
            logger.info(
                "%s seed=%s checked, clicks4=%s, forward=%.3fs",
                profile_name,
                seed,
                record["click_sector_counts"][str(sum(input_state))],
                exact_seconds,
            )
        report.setdefault("summary", {})[profile_name] = _summarize(profile_records)
        del backend
        gc.collect()
    report["status"] = "passed" if not report["failures"] else "failed"
    _write_json(run_dir / "phase1_report.json", report)
    _plot_checks(run_dir, target, figure_payload, report)
    if report["failures"]:
        raise RuntimeError(
            "Phase-one recycling has failed seeds; see phase1_report.json"
        )


def _summarize(records):
    summary = {"runs": len(records)}
    for name in [
        "exact_seconds",
        "sample_with_forward_seconds",
        "sampling_tvd",
        "postselection_tvd",
        "recycling_tvd",
        "recycling_seconds",
    ]:
        values = [record[name] for record in records if name in record]
        if values:
            summary[name + "_mean"] = float(np.mean(values))
            summary[name + "_median"] = float(np.median(values))
    if all("postselection_tvd" in record for record in records):
        summary["recycling_better_runs"] = sum(
            record["recycling_tvd"] < record["postselection_tvd"] for record in records
        )
    return summary


def _plot_checks(run_dir, target, payload, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes.flat[0].plot(target)
    axes.flat[0].set(
        title="Synthetic training target (no training yet)",
        xlabel="Fixed bin",
        ylabel="Probability",
    )
    for axis, (name, values) in zip(list(axes.flat)[1:], payload.items(), strict=False):
        for label, vector in zip(
            ["Reference", "Postselected", "Recycled"], values, strict=True
        ):
            axis.plot(vector, label=label, alpha=0.8, linewidth=0.8)
        axis.set(
            title=f"{name}, initialization seed 0",
            xlabel="Fixed bin",
            ylabel="Probability",
        )
        axis.legend(fontsize=8)
    figure.suptitle(
        f"Phase-one fixed-circuit checks: {len(report['seeds'])} seeds, no parameter optimization"
    )
    figure.savefig(run_dir / "phase1_checks.png", dpi=160)
    plt.close(figure)


def check_native_sampling(cfg: dict, run_dir) -> None:
    """Benchmark genuine MerLin ``layer(shots=N)`` on noisy paper circuits.

    Parameters
    ----------
    cfg : dict
        Shared-runtime configuration with an explicit benchmark schedule.
    run_dir : pathlib.Path
        Destination for timing records and actual native integer counts.

    Raises
    ------
    RuntimeError
        If native counts violate the budget or a sampling concentration bound.
    """
    settings = cfg["native_sampling"]
    torch.set_num_threads(settings["torch_threads"])
    profiles, provenance = load_configured_profiles(cfg, settings.get("snapshot_file"))
    input_state = tuple(cfg["model"]["input_state"])
    parameter_count = len(input_state) * (len(input_state) - 1)
    parameters = initialize_parameters(
        parameter_count, [settings["initialization_seed"]]
    )[0]
    comparisons = len(settings["profiles"]) * sum(
        row["repeats"] for row in settings["schedule"]
    )
    logger = logging.getLogger(__name__)
    report = {
        "status": "running",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "unmodified_MerLin_QuantumLayer_shots_multinomial",
        "sampling_implementation": "PhotonicBackend.sample_native -> torch.no_grad -> layer(shots=N)",
        "numpy_sampling_used": False,
        "library_patched": False,
        "input_state": input_state,
        "initialization_seed": settings["initialization_seed"],
        "device": cfg["device"],
        "dtype": str(dtype_torch(cfg["dtype"])),
        "torch_threads": settings["torch_threads"],
        "noise_provenance": provenance,
        "profiles": {name: profiles[name] for name in settings["profiles"]},
        "records": [],
        "summary": {},
        "qpu_jobs_submitted": 0,
    }
    report_path = run_dir / "native_sampling.json"
    for name in settings["profiles"]:
        backend = PhotonicBackend(
            input_state,
            profiles[name],
            dtype=dtype_torch(cfg["dtype"]),
            device=cfg["device"],
        )
        backend.set_parameters(parameters)
        exact = backend.exact()
        warmup = backend.sample_native(1000, settings["sampling_seed"])
        if sum(warmup.values()) != 1000:
            raise RuntimeError("Native warm-up count mismatch")
        profile_records = []
        for row in settings["schedule"]:
            shots = row["shots"]
            for repetition in range(row["repeats"]):
                seed = settings["sampling_seed"] + repetition
                logger.info(
                    "Native MerLin %s, N=%s, repetition=%s: starting",
                    name,
                    shots,
                    repetition,
                )
                counts = backend.sample_native(shots, seed)
                empirical = {key: value / shots for key, value in counts.items()}
                distance = total_variation(exact, empirical)
                bound = min(
                    1.0,
                    0.5 * math.sqrt(len(exact) / shots)
                    + math.sqrt(math.log(comparisons / 1e-6) / (2 * shots)),
                )
                if sum(counts.values()) != shots or distance > bound:
                    raise RuntimeError("Native MerLin sampling verification failed")
                record = {
                    "profile": name,
                    "initial_shots": shots,
                    "sampling_seed": seed,
                    "repetition": repetition,
                    "forward_and_native_sampling_seconds": backend.timings[
                        "native_sample_seconds"
                    ],
                    "observed_shots": sum(counts.values()),
                    "sampling_tvd": distance,
                    "sampling_tvd_bound": bound,
                    "vacuum_count": counts.get((0,) * len(input_state), 0),
                    "click_counts": {
                        str(k): sum(
                            value for key, value in counts.items() if sum(key) == k
                        )
                        for k in range(sum(input_state) + 1)
                    },
                }
                profile_records.append(record)
                report["records"].append(record)
                (
                    run_dir / f"native_counts_{name}_{shots}_{repetition}.json"
                ).write_text(
                    json.dumps(
                        [
                            {"state": key, "count": value}
                            for key, value in sorted(counts.items())
                        ]
                    ),
                    encoding="utf-8",
                )
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                logger.info(
                    "Native MerLin %s N=%s: %.3fs, TVD=%.6g, full-click=%s",
                    name,
                    shots,
                    record["forward_and_native_sampling_seconds"],
                    distance,
                    record["click_counts"][str(sum(input_state))],
                )
        medians = [
            {
                "shots": row["shots"],
                "median_seconds": float(
                    np.median(
                        [
                            record["forward_and_native_sampling_seconds"]
                            for record in profile_records
                            if record["initial_shots"] == row["shots"]
                        ]
                    )
                ),
            }
            for row in settings["schedule"]
        ]
        fit = np.polyfit(
            [row["shots"] for row in medians],
            [row["median_seconds"] for row in medians],
            deg=1,
        )
        report["summary"][name] = {
            "measured": medians,
            "fit_seconds_per_shot": float(fit[0]),
            "fit_intercept_seconds": float(fit[1]),
            "estimated_50million_seconds_not_measured": float(
                np.polyval(fit, 50_000_000)
            ),
            "estimate_scope": "linear fit from actual measured sizes, not a measured 50M call or training run",
        }
    report["status"] = "passed"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
