"""Photonic scaling probe: cost of the MerLin banks against the Fock dimension.

Experiment ``photonic_scaling`` (LOG.md, Phase 4 GPU extension plan). Each
configured point is one ``(embedding, n, max_photons)`` bank built exactly as in
the study (same random streams for ``photonic_scaling.subsample_seed``): the
800-point subsample, no-click features, labels and, unless disabled, the
fidelity Gram matrix. No model is trained. Per point the record holds wall-clock
for simulation and Gram, the peak device memory, the host memory of the stored
states and, when ``compare_cpu`` is set, the largest difference to a CPU
complex128 reference of the same bank.

A point that runs out of memory is recorded with ``status: out_of_memory`` and
every later point with an equal or larger Fock dimension is skipped, so one
failure does not end the probe. On Linux a point whose stored states would take
more than ``HOST_MEMORY_FRACTION`` of the available host memory is skipped
before it starts, because the kernel's OOM killer would end the whole run.
"""

from __future__ import annotations

import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import photonic_merlin, study
from .data import PCAFeatures, subsample_indices

LOGGER = logging.getLogger(__name__)

HOST_MEMORY_FRACTION = 0.8


def _peak_host_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return peak / 2**10 if sys.platform.startswith("linux") else peak / 2**20


def _available_host_mb() -> float | None:
    """``MemAvailable`` from ``/proc/meminfo`` (Linux only; None elsewhere)."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**10
    return None


def _is_out_of_memory(error: BaseException) -> bool:
    if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _build_bank(
    point, features, seed, settings, conventions, *, device, complex_dtype, timings
):
    embedding, n = point["embedding"], int(point["n"])
    return photonic_merlin.compute_photonic_bank(
        embedding,
        features,
        input_scale=conventions.input_scale,
        unitary_rng=study.stream_rng(
            seed, study.STREAM_PHOTONIC_UNITARIES, embedding, n
        ),
        label_rng=study.stream_rng(seed, study.STREAM_QNN, embedding, n),
        need_quantum_kernel=bool(point.get("need_quantum_kernel", True)),
        max_photons=point.get("max_photons"),
        device=device,
        complex_dtype=complex_dtype,
        batch_memory_mb=settings.batch_memory_mb,
        gram_memory_mb=settings.gram_memory_mb,
        timings=timings,
    )


def _max_difference(reference, candidate) -> dict:
    differences = {
        "no_click": np.abs(reference.pauli_features - candidate.pauli_features).max(),
        "labels": np.abs(reference.qnn_labels - candidate.qnn_labels).max(),
    }
    if reference.quantum_kernel is not None:
        differences["fidelity_kernel"] = np.abs(
            reference.quantum_kernel - candidate.quantum_kernel
        ).max()
    return {key: float(value) for key, value in differences.items()}


def run_photonic_scaling(
    cfg: dict,
    pca_features: PCAFeatures,
    settings: study.StudySettings,
    conventions,
    complex_dtype,
    records_path: Path,
    on_unit_completed,
) -> dict:
    """Measure every configured point and return the metrics payload.

    Parameters
    ----------
    cfg : dict
        Resolved configuration; reads ``photonic_scaling.points`` and
        ``photonic_scaling.subsample_seed``.
    pca_features : PCAFeatures
        Standardized PCA features of all images.
    settings : StudySettings
        Study settings (subsample size, device, batch memory budget).
    conventions : EmbeddingConventions
        Circuit conventions (input scale).
    complex_dtype : torch.dtype
        Simulation dtype on ``settings.device``.
    records_path : pathlib.Path
        JSONL file receiving each point's record as soon as it is measured.
    on_unit_completed : callable
        Called with ``(completed, total, record)`` after each point.

    Returns
    -------
    dict
        ``points``: one record per configured point, in configuration order.

    Raises
    ------
    ValueError
        If no point is configured or a point names a non-photonic embedding.
    """
    scaling_cfg = cfg["photonic_scaling"]
    points = list(scaling_cfg["points"])
    if not points:
        raise ValueError("photonic_scaling.points is empty")
    for point in points:
        if point["embedding"] not in photonic_merlin.PHOTONIC_EMBEDDINGS:
            raise ValueError(f"Not a photonic embedding: {point['embedding']}")
    seed = int(scaling_cfg["subsample_seed"])
    indices = subsample_indices(
        pca_features.labels,
        settings.subsample_size,
        study.stream_rng(seed, study.STREAM_SUBSAMPLE_ALL),
    )
    on_cuda = settings.device.startswith("cuda")
    smallest_failed_dimension = None
    records = []
    for index, point in enumerate(points, start=1):
        embedding, n = point["embedding"], int(point["n"])
        n_modes = photonic_merlin.mode_count(embedding, n)
        n_photons = photonic_merlin.photon_count(n_modes, point.get("max_photons"))
        fock_dim = photonic_merlin.fock_dimension(n_modes, n_photons)
        need_kernel = bool(point.get("need_quantum_kernel", True))
        record = {
            "seed": seed,
            "n": n,
            "embedding": embedding,
            "n_modes": n_modes,
            "n_photons": n_photons,
            "max_photons": point.get("max_photons"),
            "fock_dimension": fock_dim,
            "need_quantum_kernel": need_kernel,
            "stored_states_mb": (
                settings.subsample_size
                * fock_dim
                * (16 if complex_dtype == torch.complex128 else 8)
                / 2**20
                if need_kernel
                else 0.0
            ),
            "device": settings.device,
            "complex_dtype": str(complex_dtype),
        }
        started = time.perf_counter()
        available_host_mb = _available_host_mb()
        record["available_host_mb"] = available_host_mb
        if (
            smallest_failed_dimension is not None
            and fock_dim >= smallest_failed_dimension
        ):
            record["status"] = "skipped_after_out_of_memory"
        elif (
            available_host_mb is not None
            and record["stored_states_mb"] > HOST_MEMORY_FRACTION * available_host_mb
        ):
            record["status"] = "skipped_host_memory"
        else:
            features = pca_features.features(n, indices)
            if on_cuda:
                torch.cuda.empty_cache()
                study.reset_peak_device_memory(settings.device)
            timings = {}
            try:
                bank = _build_bank(
                    point,
                    features,
                    seed,
                    settings,
                    conventions,
                    device=settings.device,
                    complex_dtype=complex_dtype,
                    timings=timings,
                )
                record["status"] = "measured"
                record.update(timings)
                if point.get("compare_cpu"):
                    reference = _build_bank(
                        point,
                        features,
                        seed,
                        settings,
                        conventions,
                        device="cpu",
                        complex_dtype=torch.complex128,
                        timings=None,
                    )
                    record["max_difference_to_cpu_complex128"] = _max_difference(
                        reference, bank
                    )
                del bank
            except Exception as error:  # noqa: BLE001 (only memory errors are absorbed)
                if not _is_out_of_memory(error):
                    raise
                record["status"] = "out_of_memory"
                record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
                smallest_failed_dimension = min(
                    fock_dim, smallest_failed_dimension or fock_dim
                )
            if on_cuda:
                record["peak_device_mb"] = (
                    torch.cuda.max_memory_allocated(settings.device) / 2**20
                )
        record["peak_host_mb"] = _peak_host_mb()
        record["elapsed_seconds"] = time.perf_counter() - started
        LOGGER.info(
            "PROBE_POINT | embedding=%s | n=%d | modes=%d | photons=%d | fock_dim=%d | "
            "status=%s | simulation_s=%s | gram_s=%s | peak_device_mb=%s",
            embedding,
            n,
            n_modes,
            n_photons,
            fock_dim,
            record["status"],
            record.get("simulation_seconds"),
            record.get("gram_seconds"),
            record.get("peak_device_mb"),
        )
        records.append(record)
        study.append_jsonl(records_path, record)
        on_unit_completed(index, len(points), record)
    return {"records_file": records_path.name, "points": records}


__all__ = ["run_photonic_scaling"]
