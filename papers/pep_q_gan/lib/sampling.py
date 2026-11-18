"""Generate raw photon events, recording actual simulator and device provenance."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from math import comb
from pathlib import Path

import numpy as np
import perceval as pc
import torch
from perceval.utils import random_seed

from .artifacts import replace_file, sha256, write_json
from .circuits import circuit_id, fock_moments, input_state
from .devices import checked_device
from .distinguishable import sample_distinguishable

SAMPLER_SCHEMA = "photonic_event_pool_v1"


def _peak_memory_mib() -> float:
    if os.name == "nt":

        class MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
            ] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        memory = MemoryCounters()
        memory.cb = ctypes.sizeof(memory)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(MemoryCounters),
            ctypes.c_uint32,
        )
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return memory.PeakWorkingSetSize / 2**20
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (2**20 if platform.system() == "Darwin" else 1024)


def _merlin_distribution(circuit, state, device):
    import merlin as ml

    # Exact SLOS first; shots would return a histogram, not raw latent events.
    layer = ml.QuantumLayer(
        input_size=0,
        circuit=circuit,
        input_state=state,
        trainable_parameters=[],
        device=str(device),
        dtype=torch.float64,
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )
    with torch.no_grad():
        probabilities = layer().detach().reshape(-1)
    keys = np.asarray(layer.output_keys, dtype=np.uint16)
    backend = type(layer.computation_process.simulation_graph).__name__
    if "slos" not in backend.lower():
        raise ValueError(f"Unexpected MerLin simulation graph: {backend}")
    values = probabilities.cpu().numpy()
    if (
        not np.isfinite(values).all()
        or values.min() < -1e-10
        or abs(values.sum() - 1) > 1e-6
    ):
        raise ValueError("MerLin returned an invalid probability distribution")
    if len(keys) != comb(len(state) + sum(state) - 1, sum(state)):
        raise ValueError(
            "Expected the complete Fock output basis, including collisions"
        )
    probabilities = probabilities.clamp_min(0)
    probabilities /= probabilities.sum()
    return keys, probabilities, backend


def generate_pool(
    circuit,
    unitary: np.ndarray,
    spec: dict,
    photons: int,
    backend: str,
    settings: dict,
    folder: Path,
    condition: str = "indist",
) -> dict:
    """Generate or verify a completed pool for one fixed circuit and input.

    Parameters
    ----------
    circuit : perceval.Circuit
        Passive circuit, shared by all photon counts and backends.
    unitary : np.ndarray
        Output-row/input-column transfer matrix.
    spec : dict
        Explicit circuit definition.
    photons : int
        Conserved photon number.
    backend : str
        ``merlin`` or ``perceval``; never selected by an implicit fallback.
    settings : dict
        Samples, chunk size, device, and sampling seed.
    folder : Path
        Pool directory. A partial pool requires explicit recovery.
    condition : str
        ``dist`` uses unique photon tags through SimulatedComputer. ``indist``
        uses the full untagged Fock input. Default is ``indist``.

    Returns
    -------
    dict
        Validated pool metadata and observed statistics.
    """
    import exqalibur as xq

    if type(settings["threads"]) is not int or settings["threads"] < 1:
        raise ValueError("threads must be a positive integer")
    xq.Config.compute_max_thread_count = settings["threads"]
    xq.Config.use_single_precision = False
    if backend not in ("merlin", "perceval"):
        raise ValueError("Sampling backend must be merlin or perceval")
    if condition not in {"dist", "indist"}:
        raise ValueError("Photon condition must be dist or indist")
    if condition == "dist" and backend != "perceval":
        raise ValueError(
            "Dist event sampling uses the official Perceval tagged-state path"
        )
    device = checked_device(settings["device"], backend)
    samples, chunk = settings["samples"], settings["chunk_size"]
    if type(samples) is not int or samples < 1 or type(chunk) is not int or chunk < 1:
        raise ValueError("Samples and chunk_size must be positive integers")
    state = input_state(photons, spec["modes"])
    if (
        backend == "merlin"
        and comb(len(state) + photons - 1, photons) > settings["max_exact_states"]
    ):
        raise ValueError(
            "Exact Fock space exceeds configured limit; explicitly select a direct sampler"
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "merlinquantum",
            "perceval-quandela",
            "exqalibur",
            "torch",
            "numpy",
        )
    }
    identity = {
        "sampler_schema": SAMPLER_SCHEMA,
        "condition": condition,
        "circuit_id": circuit_id(spec),
        "circuit_schema": spec["schema"],
        "circuit_seed": spec["circuit_seed"],
        "unitary_sha256": hashlib.sha256(
            np.ascontiguousarray(unitary).tobytes()
        ).hexdigest(),
        "photons": photons,
        "modes": len(state),
        "input_state": state,
        "samples": samples,
        "chunk_size": chunk,
        "backend": backend,
        "sampling_seed": settings["seed"],
        "device": str(device),
        "versions": versions,
        "threads": settings["threads"],
        "replay_contract": "Fresh sampler/process and fixed budget/chunking; one-thread exact replay validated",
        "event_seed_rule": (
            "uint32(SeedSequence([sampling_seed, chunk_index]).generate_state(1)[0])"
            if condition == "dist"
            else "Perceval seeded stream across chunks"
        ),
        "photon_tags": list(range(photons)) if condition == "dist" else None,
    }
    complete = folder / "complete.json"
    if complete.exists():
        saved = json.loads(complete.read_text(encoding="utf-8"))
        if (
            saved["identity"] != identity
            or saved["samples_sha256"] != sha256(folder / "samples.npy")
            or saved["moments_sha256"] != sha256(folder / "moments.npz")
        ):
            raise ValueError("Existing pool configuration or sample hash differs")
        return saved
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / "samples.partial.npy"
    if partial.exists() or (folder / "samples.npy").exists():
        raise FileExistsError(
            "Incomplete pool exists: preserve it and use a fresh output directory"
        )
    write_json(folder / "config.json", identity)
    mean, covariance = fock_moments(unitary, state, condition)
    counts_file = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.uint16, shape=(samples, len(state))
    )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        if condition == "dist":
            evidence = "Perceval SimulatedComputer / CliffordClifford2017 / unique photon tags / CPU"
        elif backend == "merlin":
            keys, probabilities, evidence = _merlin_distribution(circuit, state, device)
            if device.type == "cpu":
                probabilities = probabilities.numpy()
                rng = np.random.Generator(np.random.PCG64(settings["seed"]))
            else:
                rng = torch.Generator(device=device).manual_seed(settings["seed"])
        else:
            sampler = pc.Clifford2017Backend()
            sampler.set_circuit(circuit)
            sampler.set_input_state(pc.BasicState(state))
            random_seed(settings["seed"])
            evidence = "Perceval Clifford2017Backend / exqalibur CPU"
        sync()
        setup_seconds = time.perf_counter() - started
        sums, products = np.zeros(len(state)), np.zeros((len(state), len(state)))
        collision_events = 0
        collision_pairs = 0
        sampling_seconds = conversion_seconds = saving_seconds = 0.0
        for offset in range(0, samples, chunk):
            size = min(chunk, samples - offset)
            stamp = time.perf_counter()
            if condition == "dist":
                chunk_seed = int(
                    np.random.SeedSequence(
                        [settings["seed"], offset // chunk]
                    ).generate_state(1)[0]
                )
                counts, timing = sample_distinguishable(
                    circuit, state, size, chunk_seed
                )
                sampling_seconds += timing["sampling_and_setup_seconds"]
                conversion_seconds += timing["conversion_seconds"]
                stamp = None
            elif backend == "merlin":
                if device.type == "cpu":
                    indices = rng.choice(len(keys), size=size, p=probabilities)
                else:
                    indices = torch.multinomial(
                        probabilities, size, replacement=True, generator=rng
                    )
                    sync()
                sampling_seconds += time.perf_counter() - stamp
                stamp = time.perf_counter()
                if device.type == "cuda":
                    indices = indices.cpu().numpy()
                counts = keys[indices]
            else:
                states = sampler.samples(size)
                sampling_seconds += time.perf_counter() - stamp
                stamp = time.perf_counter()
                counts = np.asarray([list(s) for s in states], dtype=np.uint16)
            if stamp is not None:
                conversion_seconds += time.perf_counter() - stamp
            if counts.shape != (size, len(state)) or not np.all(
                counts.sum(axis=1) == photons
            ):
                raise ValueError("Output shape or conserved photon number is invalid")
            values = counts.astype(np.float64)
            sums += values.sum(axis=0)
            products += values.T @ values
            collision_events += int(np.any(counts > 1, axis=1).sum())
            collision_pairs += int(
                (counts.astype(np.int64) * (counts.astype(np.int64) - 1) // 2).sum()
            )
            stamp = time.perf_counter()
            counts_file[offset : offset + size] = counts
            counts_file.flush()
            saving_seconds += time.perf_counter() - stamp
            write_json(
                folder / "progress.json",
                {
                    "completed_samples": offset + size,
                    "target_samples": samples,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
        del counts_file
        replace_file(partial, folder / "samples.npy")
        observed_mean = sums / samples
        observed_covariance = products / samples - np.outer(
            observed_mean, observed_mean
        )
        np.savez(
            folder / "moments.npz",
            expected_mean=mean,
            expected_covariance=covariance,
            observed_mean=observed_mean,
            observed_covariance=observed_covariance,
        )
        sem = np.sqrt(np.maximum(np.diag(covariance), 0) / samples)
        completed = {
            "identity": identity,
            "status": "complete",
            "backend_evidence": evidence,
            "circuit_provenance": "MerLin fixed entangling-layer recipe; recipe identity and circuit seed recorded",
            "conserved_photons": True,
            "collisions_retained": True,
            "mean_within_10_sem": bool(
                np.all(abs(observed_mean - mean) <= 10 * sem + 1e-12)
            ),
            "max_mean_error": float(abs(observed_mean - mean).max()),
            "max_covariance_error": float(abs(observed_covariance - covariance).max()),
            "mean_total_photons": float(observed_mean.sum()),
            "collision_probability": collision_events / samples,
            "collision_pair_fraction": (
                collision_pairs / (samples * comb(photons, 2)) if photons > 1 else 0.0
            ),
            "setup_seconds": setup_seconds,
            "sampling_seconds": sampling_seconds,
            "conversion_seconds": conversion_seconds,
            "saving_seconds": saving_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_RSS_MiB": _peak_memory_mib(),
            "memory_scope": "Process lifetime high-water mark; includes imports and earlier cases in this process",
            "peak_cuda_allocated_MiB": (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else None
            ),
            "samples_sha256": sha256(folder / "samples.npy"),
            "moments_sha256": sha256(folder / "moments.npz"),
            "python_version": platform.python_version(),
        }
        write_json(complete, completed)
        return completed
    except BaseException as error:
        write_json(
            folder / "failure.json",
            {"error": str(error), "partial_pool_not_reusable": True},
        )
        raise
