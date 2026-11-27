"""MerLin photonic adaptation of the embeddings, kernels and quantum labels.

This is an *adaptation*, not part of the original reproduction (LOG.md, Phase 4
Adaptation Plan and Amendment). Two linear-optical embeddings with k = floor(m/2)
indistinguishable photons in m modes (optionally capped at ``max_photons``),
input state ``[1, 0, 1, 0, ...]``:

* ``PA`` (photonic analogue of E2's IQP structure), m = n modes:
  ``U(x) = W . diag(exp(i phi_j)) . V`` with fixed Haar-random V and W and
  ``phi_j = 2 s x_j`` (same angle convention as the gate RX(2x)).
* ``PB`` (photonic analogue of E3's Heisenberg chain), m = n + 1 modes:
  T Trotter steps of beam splitters BS(j, j+1), j = 0..n-1, each with Perceval
  angle ``theta = 4 (t/T) s x_j`` with t = (n+1)/3. Perceval's ``BS(theta)``
  equals ``exp(i theta/2 (a^dag b + b^dag a))``, the bosonic-hopping counterpart
  of the XX+YY exchange term (up to complex conjugation).

Kernels and labels (threshold detectors, exact simulation, no postselection):

* Fidelity kernel ``|<s|U(x')^dag U(x)|s>|^2``, the Gram matrix of FOCK-space
  amplitude vectors. Physically: P(click pattern = input) after U(x')^dag U(x).
* Projected-kernel features: per-mode no-click probabilities P(n_k = 0). With a
  fixed photon number a mode's reduced state is diagonal in photon number, so
  this is the threshold-detector counterpart of the single-qubit reduced state.
* Quantum-dataset label: P(n_0 = 0) after a fixed Haar-random interferometer R.

The FOCK computation space is used for simulation because the no-click
marginals are measured without postselection; UNBUNCHED would condition on
collision-free outcomes (verified equal to postselected FOCK, LOG.md).

Memory: one state has C(m+k-1, k) amplitudes. States are simulated in batches
sized from a memory budget; when the fidelity kernel is needed they are stored
on the host and the Gram matrix is built from device-sized blocks of states, so
device memory holds at most two blocks (LOG.md, Phase 4 GPU extension).
"""

from __future__ import annotations

import logging
import math
import time

import merlin as ml
import numpy as np
import perceval as pcvl
import torch

from .embeddings import QuantumBank, _fidelity_gram

LOGGER = logging.getLogger(__name__)

PHOTONIC_EMBEDDINGS = ("PA", "PB")
PB_TROTTER_STEPS = 20


def mode_count(embedding: str, n_features: int) -> int:
    """Number of optical modes used by ``embedding`` for ``n_features`` inputs."""
    if embedding == "PA":
        return n_features
    if embedding == "PB":
        return n_features + 1
    raise ValueError(f"Unknown photonic embedding: {embedding}")


def photon_count(n_modes: int, max_photons: int | None = None) -> int:
    """k = floor(m / 2) photons, capped at ``max_photons`` when given.

    Parameters
    ----------
    n_modes : int
        Number of optical modes m.
    max_photons : int or None
        Upper bound on k. With a cap the Fock dimension grows polynomially in m
        instead of exponentially (a different scaling regime, labelled as such
        in LOG.md). Default value is None (no cap).

    Raises
    ------
    ValueError
        If fewer than two photons would result: a single photon is a classical
        linear-optical baseline and is not a valid photonic adaptation.
    """
    photons = n_modes // 2
    if max_photons is not None:
        photons = min(photons, int(max_photons))
    if photons < 2:
        raise ValueError(
            f"{n_modes} modes give {photons} photon(s); at least 2 are required"
        )
    return photons


def input_state(n_modes: int, max_photons: int | None = None) -> list[int]:
    """Alternating occupation ``[1, 0, 1, 0, ...]`` with ``photon_count`` photons."""
    photons = photon_count(n_modes, max_photons)
    return [1 if mode % 2 == 0 and mode < 2 * photons else 0 for mode in range(n_modes)]


def fock_dimension(n_modes: int, n_photons: int) -> int:
    """Number of Fock states of ``n_photons`` bosons in ``n_modes`` modes."""
    return math.comb(n_modes + n_photons - 1, n_photons)


def haar_unitary(n_modes: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary from the QR decomposition of a complex Gaussian matrix."""
    gaussian = (
        rng.standard_normal((n_modes, n_modes))
        + 1j * rng.standard_normal((n_modes, n_modes))
    ) / np.sqrt(2)
    q, r = np.linalg.qr(gaussian)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def embedding_circuit(
    embedding: str, n_features: int, rng: np.random.Generator
) -> tuple[pcvl.Circuit, int]:
    """Perceval circuit with input parameters ``px*`` and its parameter count."""
    n_modes = mode_count(embedding, n_features)
    circuit = pcvl.Circuit(n_modes)
    if embedding == "PA":
        circuit.add(0, pcvl.Unitary(pcvl.Matrix(haar_unitary(n_modes, rng))))
        for mode in range(n_modes):
            circuit.add(mode, pcvl.PS(pcvl.P(f"px{mode:03d}")))
        circuit.add(0, pcvl.Unitary(pcvl.Matrix(haar_unitary(n_modes, rng))))
        return circuit, n_modes
    for step in range(PB_TROTTER_STEPS):
        for bond in range(n_features):
            circuit.add(
                bond, pcvl.BS(theta=pcvl.P(f"px{step * n_features + bond:04d}"))
            )
    return circuit, PB_TROTTER_STEPS * n_features


def circuit_inputs(
    embedding: str, features: torch.Tensor, input_scale: float
) -> torch.Tensor:
    """Map scaled features to the circuit's input parameters (one row per point)."""
    scaled = input_scale * features
    if embedding == "PA":
        return 2.0 * scaled
    n_features = features.shape[1]
    total_time = (n_features + 1) / 3.0
    angles = 4.0 * (total_time / PB_TROTTER_STEPS) * scaled
    return angles.repeat(1, PB_TROTTER_STEPS)


def _real_dtype(complex_dtype) -> torch.dtype:
    if complex_dtype == torch.complex128:
        return torch.float64
    if complex_dtype == torch.complex64:
        return torch.float32
    raise ValueError(f"Unsupported complex dtype: {complex_dtype}")


def _layer(
    circuit: pcvl.Circuit,
    n_modes: int,
    strategy,
    *,
    max_photons: int | None = None,
    device: str = "cpu",
    real_dtype: torch.dtype = torch.float64,
) -> ml.QuantumLayer:
    return ml.QuantumLayer(
        circuit=circuit,
        input_state=input_state(n_modes, max_photons),
        n_photons=photon_count(n_modes, max_photons),
        input_parameters=["px"],
        trainable_parameters=[],
        computation_space=ml.ComputationSpace.FOCK,
        measurement_strategy=strategy,
        device=torch.device(device),
        dtype=real_dtype,
    )


def _no_click_mask(output_keys, real_dtype=torch.float64) -> torch.Tensor:
    """mask[b, k] = 1 when Fock outcome b has no photon in mode k."""
    keys = torch.as_tensor([list(key) for key in output_keys], dtype=torch.int8)
    return (keys == 0).to(real_dtype)


def points_per_block(
    fock_dim: int, complex_dtype, memory_mb: float | None, cap: int
) -> int:
    """Points whose amplitude vectors fit in ``memory_mb`` (at least 1, at most ``cap``).

    Without a budget the historical fixed batch of ``cap`` points is used, which
    keeps earlier CPU runs bit-identical.
    """
    if memory_mb is None:
        return cap
    bytes_per_point = fock_dim * (16 if complex_dtype == torch.complex128 else 8)
    return max(1, min(cap, int(memory_mb * 2**20 // bytes_per_point)))


def fidelity_gram_from_host(
    states: torch.Tensor, device: str, block_points: int
) -> np.ndarray:
    """``|<psi_i|psi_j>|^2`` for host-stored states, computed in device blocks.

    Only the upper block triangle is computed (the kernel is symmetric), so the
    device holds at most two blocks of ``block_points`` states.
    """
    n_points = states.shape[0]
    gram = np.empty((n_points, n_points), dtype=np.float64)
    for row_start in range(0, n_points, block_points):
        row_stop = min(row_start + block_points, n_points)
        rows = states[row_start:row_stop].to(device)
        rows_conjugated = rows.conj()
        for col_start in range(row_start, n_points, block_points):
            col_stop = min(col_start + block_points, n_points)
            cols = (
                rows
                if col_start == row_start
                else states[col_start:col_stop].to(device)
            )
            values = (rows_conjugated @ cols.T).abs().square().double().cpu().numpy()
            gram[row_start:row_stop, col_start:col_stop] = values
            gram[col_start:col_stop, row_start:row_stop] = values.T
            del cols
        del rows, rows_conjugated
    # Pure states: the diagonal is exactly one; remove rounding drift.
    np.fill_diagonal(gram, 1.0)
    return 0.5 * (gram + gram.T)


def compute_photonic_bank(
    embedding: str,
    features: np.ndarray,
    *,
    input_scale: float,
    unitary_rng: np.random.Generator,
    label_rng: np.random.Generator | None,
    need_quantum_kernel: bool,
    batch_size: int | None = None,
    max_photons: int | None = None,
    device: str = "cpu",
    complex_dtype=torch.complex128,
    batch_memory_mb: float | None = None,
    gram_memory_mb: float | None = None,
    timings: dict | None = None,
) -> QuantumBank:
    """No-click features, fidelity kernel and labels for one subsample.

    Parameters
    ----------
    embedding : str
        ``"PA"`` or ``"PB"``.
    features : numpy.ndarray
        Standardized PCA features, shape ``(N, n)``.
    input_scale : float
        Effective input scale ``s`` (same convention as the gate study).
    unitary_rng : numpy.random.Generator
        Generator for the fixed Haar interferometers of PA (unused by PB).
    label_rng : numpy.random.Generator or None
        Generator for the labelling interferometer R; None skips the labels.
    need_quantum_kernel : bool
        Whether to store all states and build the fidelity Gram matrix.
    batch_size : int or None
        Points per simulated batch. If omitted it is derived from
        ``batch_memory_mb`` (at most 200). Default value is None.
    max_photons : int or None
        Cap on the photon number (see ``photon_count``). Default value is None.
    device : str
        Torch device for the simulation and the Gram blocks. Default value is
        ``"cpu"``.
    complex_dtype : torch.dtype
        ``torch.complex128`` or ``torch.complex64``. Default value is
        ``torch.complex128``.
    batch_memory_mb : float or None
        Budget for the output amplitudes of one simulated batch. MerLin's
        intermediate tensors need about 22x this on the device (measured on an
        L4: about 175 B per complex64 amplitude, LOG.md). None keeps the fixed
        200-point batch. Default value is None.
    gram_memory_mb : float or None
        Budget for each of the two device blocks of the fidelity Gram matrix
        (non-CPU devices only). None puts all states on the device at once.
        Default value is None.
    timings : dict or None
        If given, filled with ``simulation_seconds``, ``gram_seconds``,
        ``batch_points``, ``gram_block_points``, ``n_photons`` and
        ``fock_dimension``. Default value is None.

    Returns
    -------
    QuantumBank
        ``pauli_features`` holds the no-click probabilities with shape
        ``(N, m, 1)``; ``n_qubits`` holds the number of modes m.
    """
    n_points, n_features = features.shape
    n_modes = mode_count(embedding, n_features)
    n_photons = photon_count(n_modes, max_photons)
    real_dtype = _real_dtype(complex_dtype)
    layer_options = {
        "max_photons": max_photons,
        "device": device,
        "real_dtype": real_dtype,
    }
    started = time.perf_counter()
    circuit, _ = embedding_circuit(embedding, n_features, unitary_rng)
    state_layer = _layer(
        circuit, n_modes, ml.MeasurementStrategy.AMPLITUDES, **layer_options
    )
    fock_dim = state_layer.output_size
    no_click = _no_click_mask(state_layer.output_keys, real_dtype).to(device)
    label_layer, label_mask = None, None
    if label_rng is not None:
        labelled = circuit.copy()
        labelled.add(0, pcvl.Unitary(pcvl.Matrix(haar_unitary(n_modes, label_rng))))
        label_layer = _layer(
            labelled, n_modes, ml.MeasurementStrategy.PROBABILITIES, **layer_options
        )
        label_mask = _no_click_mask(label_layer.output_keys, real_dtype)[:, 0].to(
            device
        )
    if batch_size is None:
        batch_size = points_per_block(fock_dim, complex_dtype, batch_memory_mb, 200)

    inputs = circuit_inputs(
        embedding, torch.as_tensor(features, dtype=torch.float64), input_scale
    ).to(device=device, dtype=real_dtype)
    # States stay on the host: at 18 modes / 9 photons 800 states are 20 GB.
    stored = (
        torch.empty((n_points, fock_dim), dtype=complex_dtype)
        if need_quantum_kernel
        else None
    )
    feature_chunks, label_chunks = [], []
    with torch.no_grad():
        for start in range(0, n_points, batch_size):
            stop = min(start + batch_size, n_points)
            amplitudes = state_layer(inputs[start:stop])
            probabilities = amplitudes.abs().square()
            feature_chunks.append((probabilities @ no_click).double().cpu().numpy())
            if stored is not None:
                stored[start:stop] = amplitudes.cpu()
            del amplitudes, probabilities
            if label_layer is not None:
                label_chunks.append(
                    (label_layer(inputs[start:stop]) @ label_mask)
                    .double()
                    .cpu()
                    .numpy()
                )
        simulated = time.perf_counter()
        gram_block = None
        quantum_kernel = None
        if stored is not None:
            if device == "cpu":
                # Same code path as the original CPU runs (bit-identical results).
                quantum_kernel = _fidelity_gram(stored)
            else:
                gram_block = points_per_block(
                    fock_dim, complex_dtype, gram_memory_mb, n_points
                )
                quantum_kernel = fidelity_gram_from_host(stored, device, gram_block)
    finished = time.perf_counter()
    LOGGER.debug(
        "Simulated %s on %d modes / %d photons (Fock dimension %d) for %d points",
        embedding,
        n_modes,
        n_photons,
        fock_dim,
        n_points,
    )
    if timings is not None:
        timings.update(
            simulation_seconds=simulated - started,
            gram_seconds=finished - simulated,
            batch_points=batch_size,
            gram_block_points=gram_block,
            n_photons=n_photons,
            fock_dimension=fock_dim,
        )
    return QuantumBank(
        pauli_features=np.concatenate(feature_chunks)[:, :, None],
        quantum_kernel=quantum_kernel,
        qnn_labels=np.concatenate(label_chunks) if label_chunks else None,
        n_qubits=n_modes,
    )


__all__ = [
    "PB_TROTTER_STEPS",
    "PHOTONIC_EMBEDDINGS",
    "circuit_inputs",
    "compute_photonic_bank",
    "embedding_circuit",
    "fidelity_gram_from_host",
    "fock_dimension",
    "haar_unitary",
    "input_state",
    "mode_count",
    "photon_count",
    "points_per_block",
]
