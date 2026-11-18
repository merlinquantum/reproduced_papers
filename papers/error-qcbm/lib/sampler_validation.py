"""Fresh, portable stock-versus-vectorized MerLin count equivalence checks."""

import ctypes
import hashlib
import importlib.metadata
import inspect
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from merlin.measurement.process import SamplingProcess

from .backend import DEMO_INPUT_STATE, PhotonicBackend, initialize_parameters
from .data import collision_free_states, gaussian_target
from .recycling import recycling_mitigation
from .spsa import SPSA, distribution_loss
from .training import FiniteShotObjective, conditional_distribution, write_json

SCHEMA = "merlin_count_equivalence_v2"
CASES = {
    "noisy": ("belenos_postselection", "belenos_recycling"),
    "partial": ("lossless_partial",),
    "ideal": ("lossless_ideal",),
}


def fingerprint():
    """Return exact source and runtime identifiers used by the audit gate.

    Returns
    -------
    dict
        Package/platform versions, source hashes and validated execution scope.
    """
    project = Path(__file__).resolve().parent
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "device": "cpu",
        "dtype": "float64",
        "threads": 1,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("merlinquantum", "perceval-quandela", "torch", "numpy", "scipy")
        },
        "adapter_sha256": hashlib.sha256(
            (project / "sampling.py").read_bytes()
        ).hexdigest(),
        "backend_sha256": hashlib.sha256(
            (project / "backend.py").read_bytes()
        ).hexdigest(),
        "merlin_sampler_sha256": hashlib.sha256(
            Path(inspect.getsourcefile(SamplingProcess)).read_bytes()
        ).hexdigest(),
    }


def peak_memory_bytes():
    """Return process peak resident memory using operating-system APIs.

    Returns
    -------
    int
        Peak working-set bytes on Windows or peak resident bytes on Unix.

    Raises
    ------
    OSError
        If the operating-system memory query fails.
    """
    if sys.platform == "win32":
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working_set",
                    "working_set",
                    "quota_peak_paged",
                    "quota_paged",
                    "quota_peak_nonpaged",
                    "quota_nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(MemoryCounters),
            wintypes.DWORD,
        ]
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.peak_working_set)
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
        1 if sys.platform == "darwin" else 1024
    )


def validate_full_evidence(evidence):
    """Reject partial, incompatible or unequal evidence before production use.

    Parameters
    ----------
    evidence : dict
        Complete four-condition, two-phase audit report.

    Returns
    -------
    dict
        The accepted report, unchanged.

    Raises
    ------
    ValueError
        If scope, fingerprints, budgets, phases or equality checks disagree.
    """
    names = set().union(*map(set, CASES.values()))
    if evidence["schema"] != SCHEMA or evidence["status"] != "passed":
        raise ValueError("Sampler evidence is not a passed v2 audit")
    if evidence["environment"] != fingerprint():
        raise ValueError("Sampler evidence runtime/source fingerprint mismatch")
    if set(evidence["cases"]) != names or evidence["input_state"] != list(
        DEMO_INPUT_STATE
    ):
        raise ValueError("Sampler evidence requires all paper-scale conditions")
    for name, row in evidence["cases"].items():
        expected_transmission = 0.028 if name.startswith("belenos_") else 1.0
        expected_overlap = 1.0 if name == "lossless_ideal" else 0.729
        if (
            row["profile"]["transmittance"] != expected_transmission
            or row["profile"]["indistinguishability"] != expected_overlap
        ):
            raise ValueError(f"Sampler evidence physical profile mismatch: {name}")
        if row["shots"] != 50_000_000 or row["chunk_shots"] != 1_000_000:
            raise ValueError(f"Sampler evidence requires 50M in 1M chunks: {name}")
        if len(row["comparisons"]) != 2 or {c["label"] for c in row["comparisons"]} != {
            "initial_fresh_reference",
            "trained_probe_fresh_reference",
        }:
            raise ValueError("Sampler evidence needs both fresh phase points")
        for comparison in row["comparisons"]:
            if not (
                comparison["counts_identical"]
                and comparison["rng_traces_identical"]
                and comparison["caller_rng_restored"]
                and comparison["probabilities_identical"]
                and comparison["loss_identical"]
                and comparison["max_count_difference"] == 0
                and comparison["stock_chunks"] == comparison["vectorized_chunks"] == 50
                and comparison["stock_shots"]
                == comparison["vectorized_shots"]
                == 50_000_000
            ):
                raise ValueError(f"Sampler exact comparison failed: {name}")
    phase_hashes = [
        {c["label"]: c["phase_sha256"] for c in row["comparisons"]}
        for row in evidence["cases"].values()
    ]
    if any(value != phase_hashes[0] for value in phase_hashes):
        raise ValueError("Sampler evidence phases differ between profile groups")
    return evidence


def combine_evidence(reports):
    """Merge disjoint physical-profile audits, retaining shared noisy draws.

    Parameters
    ----------
    reports : sequence of dict
        Completed reports with identical runtime/source fingerprints and phases.

    Returns
    -------
    dict
        Complete production-qualified report.

    Raises
    ------
    ValueError
        If reports overlap, omit conditions or fail full-budget validation.
    """
    if not reports:
        raise ValueError("No sampler audit reports")
    first = reports[0]
    result = {k: first[k] for k in ("schema", "environment", "input_state")}
    result.update(status="passed", cases={})
    for report in reports:
        if report["status"] != "passed" or any(
            report[k] != result[k] for k in ("schema", "environment", "input_state")
        ):
            raise ValueError("Incompatible or incomplete sampler audit reports")
        if set(result["cases"]) & set(report["cases"]):
            raise ValueError("Duplicate sampler audit cases")
        result["cases"].update(report["cases"])
    validate_full_evidence(result)
    return result


def run_sampler_validation(cfg, run_dir):
    """Compare fresh native draws at initial and bounded trained-probe phases.

    Parameters
    ----------
    cfg : dict
        Shared-runtime configuration with sampler_validation settings.
    run_dir : pathlib.Path
        Timestamped artifact directory. No private campaign inputs are read.
    """
    settings = cfg["sampler_validation"]
    if str(cfg["device"]) != "cpu":
        raise ValueError("Sampler audit requires CPU")
    from runtime_lib.dtypes import dtype_torch

    if dtype_torch(cfg["dtype"]) != torch.float64:
        raise ValueError("Sampler audit requires float64")
    shots, chunk = settings["shots"], settings["chunk_shots"]
    if (
        not isinstance(shots, int)
        or shots <= 0
        or not isinstance(chunk, int)
        or chunk <= 0
    ):
        raise ValueError("Positive integer audit shots and chunks are required")
    if len(set(settings["profiles"])) != len(settings["profiles"]) or not set(
        settings["profiles"]
    ) <= set(CASES):
        raise ValueError("Select distinct noisy/partial/ideal profiles")
    torch.set_num_threads(1)
    original = SamplingProcess.pcvl_sampler
    environment = fingerprint()
    inputs = tuple(cfg["model"]["input_state"])
    states = collision_free_states(len(inputs), sum(inputs))
    bins = len(states)
    target = gaussian_target(
        bins, (2 * bins // 7, 5 * bins // 7), (max(1, bins // 8),) * 2
    )
    initial = initialize_parameters(
        len(inputs) * (len(inputs) - 1), [settings["initialization_seed"]]
    )[0]
    probe = PhotonicBackend(
        inputs, {"name": "probe", "transmittance": 1.0, "indistinguishability": 1.0}
    )
    objective = FiniteShotObjective(
        probe,
        states,
        target,
        "postselection",
        settings["probe_shots"],
        settings["sampling_seed"] + 100000,
        metric="kl",
    )
    optimizer = SPSA(
        initial, objective, settings["spsa"], 1000, settings["perturbation_seed"]
    )
    for _ in range(settings["probe_updates"]):
        optimizer.step()
    trained = optimizer.parameters.copy()
    if np.array_equal(initial, trained):
        raise RuntimeError("Trained diagnostic probe did not change phases")
    points = [
        ("initial_fresh_reference", initial),
        ("trained_probe_fresh_reference", trained),
    ]
    np.savez(run_dir / "diagnostic_phases.npz", initial=initial, trained_probe=trained)
    report = {
        "schema": SCHEMA,
        "status": "running",
        "environment": environment,
        "input_state": list(inputs),
        "cases": {},
        "physical_profiles": {},
        "probe": {
            "updates": settings["probe_updates"],
            "shots_per_call": settings["probe_shots"],
            "objective_calls": objective.calls,
            "scope": "Small ideal finite-shot SPSA diagnostic; not a production-trained model",
        },
    }
    write_json(run_dir / "sampler_equivalence.json", report)
    try:
        for group in settings["profiles"]:
            transmission = settings["noisy_transmittance"] if group == "noisy" else 1.0
            overlap = 1.0 if group == "ideal" else 0.729
            profile = {
                "name": group,
                "transmittance": transmission,
                "indistinguishability": overlap,
            }
            backend = PhotonicBackend(inputs, profile, native_chunk_shots=chunk)
            per_case = {
                name: {
                    "shots": shots,
                    "chunk_shots": chunk,
                    "profile": profile,
                    "comparisons": [],
                }
                for name in CASES[group]
            }
            for point_index, (label, phases) in enumerate(points):
                backend.set_parameters(phases)
                measured = {}
                for mode in ("python", "vectorized"):
                    backend.native_count_accumulation = mode
                    trace = []
                    original_forward = backend._probabilities

                    def observed(
                        shots=0,
                        forward=original_forward,
                        rng_trace=trace,
                        group_name=group,
                        phase_label=label,
                        accumulation=mode,
                    ):
                        output = forward(shots)
                        if shots:
                            rng_trace.append(
                                hashlib.sha256(
                                    torch.random.get_rng_state().numpy().tobytes()
                                ).hexdigest()
                            )
                            # Append-only progress avoids Windows readers/OneDrive
                            # denying os.replace on a frequently refreshed file.
                            progress = {
                                "profile": group_name,
                                "point": phase_label,
                                "accumulation": accumulation,
                                "completed_chunks": len(rng_trace),
                                "chunk_shots": chunk,
                                "planned_shots": settings["shots"],
                            }
                            with (run_dir / "sampling_progress.jsonl").open(
                                "a", encoding="utf-8"
                            ) as stream:
                                stream.write(json.dumps(progress) + "\n")
                        return output

                    backend._probabilities = observed
                    caller_state = torch.random.get_rng_state().clone()
                    started = perf_counter()
                    try:
                        counts = backend.sample(
                            shots, settings["sampling_seed"] + point_index
                        )
                    finally:
                        backend._probabilities = original_forward
                    measured[mode] = {
                        "counts": counts,
                        "rng": trace,
                        "seconds": perf_counter() - started,
                        "caller_restored": bool(
                            torch.equal(caller_state, torch.random.get_rng_state())
                        ),
                        "chunks": backend.timings["native_chunks"],
                    }
                    # Preserve a completed stock call even if a later audit
                    # operation fails; these are evidence, not training state.
                    np.savez(
                        run_dir / f"{group}_{point_index}_{mode}_completed.npz",
                        states=np.asarray(list(counts)),
                        counts=np.asarray(list(counts.values())),
                        rng_trace=np.asarray(trace),
                        phase_sha256=hashlib.sha256(phases.tobytes()).hexdigest(),
                        sampling_seed=settings["sampling_seed"] + point_index,
                        seconds=measured[mode]["seconds"],
                    )
                a, b = measured["python"], measured["vectorized"]
                keys = sorted(a["counts"].keys() | b["counts"].keys())
                max_difference = max(
                    abs(a["counts"].get(k, 0) - b["counts"].get(k, 0)) for k in keys
                )
                base = {
                    "label": label,
                    "phase_sha256": hashlib.sha256(phases.tobytes()).hexdigest(),
                    "counts_identical": a["counts"] == b["counts"],
                    "max_count_difference": max_difference,
                    "rng_traces_identical": a["rng"] == b["rng"],
                    "caller_rng_restored": a["caller_restored"]
                    and b["caller_restored"],
                    "stock_chunks": a["chunks"],
                    "vectorized_chunks": b["chunks"],
                    "stock_shots": sum(a["counts"].values()),
                    "vectorized_shots": sum(b["counts"].values()),
                    "stock_seconds": a["seconds"],
                    "vectorized_seconds": b["seconds"],
                    "click_sectors": {
                        str(i): sum(
                            c for state, c in b["counts"].items() if sum(state) == i
                        )
                        for i in range(sum(inputs) + 1)
                    },
                }
                np.savez(
                    run_dir / f"{group}_{point_index}_counts.npz",
                    states=np.asarray(keys),
                    stock=np.asarray([a["counts"].get(k, 0) for k in keys]),
                    vectorized=np.asarray([b["counts"].get(k, 0) for k in keys]),
                )
                for name in CASES[group]:
                    estimator = (
                        recycling_mitigation if name == "belenos_recycling" else None
                    )
                    probabilities = [
                        estimator(row["counts"], len(inputs), sum(inputs))[
                            "probabilities"
                        ]
                        if estimator
                        else conditional_distribution(row["counts"], states)
                        for row in (a, b)
                    ]
                    losses = [distribution_loss(p, target, "kl") for p in probabilities]
                    comparison = {
                        **base,
                        "probabilities_identical": bool(np.array_equal(*probabilities)),
                        "loss_identical": losses[0] == losses[1],
                        "stock_kl": losses[0],
                        "vectorized_kl": losses[1],
                    }
                    per_case[name]["comparisons"].append(comparison)
                    if not all(
                        comparison[k]
                        for k in (
                            "counts_identical",
                            "rng_traces_identical",
                            "caller_rng_restored",
                            "probabilities_identical",
                            "loss_identical",
                        )
                    ):
                        raise RuntimeError(f"Sampler mismatch: {name}/{label}")
                report["cases"].update(per_case)
                report["physical_profiles"][group] = {
                    "peak_process_memory_bytes": peak_memory_bytes()
                }
                write_json(run_dir / "sampler_equivalence.json", report)
        if SamplingProcess.pcvl_sampler is not original or fingerprint() != environment:
            raise RuntimeError("Installed sampler identity/source changed during audit")
        report["status"] = "passed"
        report["production_budget"] = (
            shots == 50000000 and chunk == 1000000 and inputs == DEMO_INPUT_STATE
        )
        write_json(run_dir / "sampler_equivalence.json", report)
        if settings.get("evidence_file"):
            validated = combine_evidence([report])
            output = Path(settings["evidence_file"])
            if not output.is_absolute():
                output = Path(__file__).resolve().parents[1] / output
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, validated)
    except Exception as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        write_json(run_dir / "sampler_equivalence.json", report)
        raise
