"""Vectorised gate-model statevector simulator for QKS circuits (1, 2, 4, 8 qubits).

The Quil ansätze from the appendix (Figs. 6-8) of arXiv:1806.08321 use ``RX``
rotations followed by a fixed CNOT/CZ network.  We implement the simulator
directly in NumPy, batched over `n_samples`, so the inner loop runs without
per-sample Python overhead.

Convention: qubit ``0`` is the most-significant bit of the integer index of
computational-basis states.  For ``|b0 b1 ... b_{n-1}>`` the integer is
``sum_i b_i * 2**(n-1-i)``.

A "circuit ansatz" is a function ``(theta_batch, n_layers, rng) -> bits``,
returning a ``(n_samples, n_qubits)`` ``int8`` array of single-shot measurements.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _per_qubit_rx_batch(angles: np.ndarray) -> np.ndarray:
    """Return ``(n_qubits, n_samples, 2, 2)`` RX unitaries for each angle.

    ``angles`` has shape ``(n_samples, n_qubits)``.
    """
    cos = np.cos(angles / 2.0)
    sin = np.sin(angles / 2.0)
    # Build (n_samples, n_qubits, 2, 2)
    u = np.empty(angles.shape + (2, 2), dtype=np.complex128)
    u[..., 0, 0] = cos
    u[..., 0, 1] = -1j * sin
    u[..., 1, 0] = -1j * sin
    u[..., 1, 1] = cos
    # Move qubit axis to front: (n_qubits, n_samples, 2, 2)
    return np.moveaxis(u, 1, 0)


def _apply_rx_layer_vec(
    psi: np.ndarray, angles: np.ndarray, n_qubits: int
) -> np.ndarray:
    """Vectorised: build the per-sample tensor product, then matmul."""
    n_samples = psi.shape[0]
    u_each = _per_qubit_rx_batch(angles)  # (n_qubits, n_samples, 2, 2)
    # Build the full per-sample unitary via repeated batched Kronecker.
    U = u_each[0]  # (n_samples, 2, 2)
    for q in range(1, n_qubits):
        # batched Kronecker: shape (n_samples, 2**q, 2**q) ⊗ (n_samples, 2, 2)
        # -> (n_samples, 2**(q+1), 2**(q+1))
        d1 = U.shape[-1]
        d2 = u_each[q].shape[-1]
        U = (U[:, :, None, :, None] * u_each[q][:, None, :, None, :]).reshape(
            n_samples, d1 * d2, d1 * d2
        )
    return np.einsum("nij,nj->ni", U, psi)


def _cnot_matrix(n_qubits: int, control: int, target: int) -> np.ndarray:
    dim = 2**n_qubits
    U = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[control] == 1:
            bits[target] ^= 1
        j = 0
        for q in range(n_qubits):
            j |= bits[q] << (n_qubits - 1 - q)
        U[j, i] = 1.0
    return U


def _cz_matrix(n_qubits: int, q1: int, q2: int) -> np.ndarray:
    dim = 2**n_qubits
    U = np.eye(dim, dtype=np.complex128)
    for i in range(dim):
        b1 = (i >> (n_qubits - 1 - q1)) & 1
        b2 = (i >> (n_qubits - 1 - q2)) & 1
        if b1 == 1 and b2 == 1:
            U[i, i] = -1.0
    return U


def _measure_bitstring(
    probs_batch: np.ndarray, n_qubits: int, rng: np.random.Generator
) -> np.ndarray:
    n = probs_batch.shape[0]
    probs_batch = np.clip(probs_batch.real, 0.0, None)
    probs_batch = probs_batch / probs_batch.sum(axis=1, keepdims=True)
    cum = np.cumsum(probs_batch, axis=1)
    u = rng.uniform(size=(n, 1))
    indices = (u > cum).sum(axis=1)
    bits = np.zeros((n, n_qubits), dtype=np.int8)
    for q in range(n_qubits):
        bits[:, q] = (indices >> (n_qubits - 1 - q)) & 1
    return bits


def _build_entangler(name: str, n_qubits: int) -> np.ndarray:
    """Build the entangling-layer unitary for a given ansatz."""
    if name == "cnot1":
        if n_qubits != 1:
            raise ValueError("cnot1 requires n_qubits == 1")
        return np.array([[1.0]], dtype=np.complex128)
    if name == "cnot2":
        if n_qubits != 2:
            raise ValueError("cnot2 requires n_qubits == 2")
        return _cnot_matrix(2, 0, 1)
    if name == "cz2":
        if n_qubits != 2:
            raise ValueError("cz2 requires n_qubits == 2")
        return _cz_matrix(2, 0, 1)
    if name == "cnot4":
        # From Fig. 6: CNOT 0 2; CNOT 1 3; CNOT 0 1; CNOT 2 3
        if n_qubits != 4:
            raise ValueError("cnot4 requires n_qubits == 4")
        U = _cnot_matrix(4, 0, 2)
        U = _cnot_matrix(4, 1, 3) @ U
        U = _cnot_matrix(4, 0, 1) @ U
        U = _cnot_matrix(4, 2, 3) @ U
        return U
    if name == "cnot8":
        # From Fig. 7 of the appendix.
        if n_qubits != 8:
            raise ValueError("cnot8 requires n_qubits == 8")
        cnots = [
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
        ]
        U = np.eye(2**8, dtype=np.complex128)
        for c, t in cnots:
            U = _cnot_matrix(8, c, t) @ U
        return U
    raise ValueError(f"Unknown ansatz: {name}")


def _hadamard_layer(n_qubits: int) -> np.ndarray:
    """H on every qubit."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    U = np.array([[1.0]], dtype=np.complex128)
    for _ in range(n_qubits):
        U = np.kron(U, h)
    return U


def entangler_precedes_rotations(name: str) -> bool:
    """Whether the entangling layer comes *before* the RX layer.

    Fig. 2(b) of arXiv:1806.08321 is drawn as ``H, H | CZ | RX(theta_0),
    RX(theta_1) | measure`` -- the CZ acts on ``|++>`` and the data-dependent
    rotations come afterwards.  This ordering is what makes the ansatz
    non-discriminating: ``CZ|++>`` is maximally entangled, so each qubit's
    reduced state is maximally mixed and no subsequent single-qubit rotation
    can reintroduce any dependence on the input.  Fig. 2(a) and Fig. 2(c) use
    the opposite order (rotations first, then the CNOT network).
    """
    return name == "cz2"


def _prep_layer(name: str, n_qubits: int) -> np.ndarray | None:
    """State-preparation layer applied once to ``|0...0>`` before everything."""
    if name == "cz2":
        return _hadamard_layer(n_qubits)
    return None


def make_ansatz_probs(
    name: str, n_qubits: int
) -> Callable[[np.ndarray, int], np.ndarray]:
    """Return ``f(theta_batch, n_layers) -> (n_samples, 2**n_qubits)`` probabilities.

    This is the noiseless, shot-free output distribution of the ansatz.  It is
    the quantity the paper's implicit-kernel derivation is written in terms of,
    so exposing it lets us check the simulator against the closed form in
    ``tests/test_kernel.py`` without going through single-shot sampling.
    """
    entangler = _build_entangler(name, n_qubits)
    prep = _prep_layer(name, n_qubits)
    ent_first = entangler_precedes_rotations(name)
    dim = 2**n_qubits

    def probs_of(theta_batch: np.ndarray, n_layers: int) -> np.ndarray:
        if n_layers == 1 and theta_batch.ndim == 2:
            theta_layers = theta_batch[:, None, :]
        else:
            theta_layers = theta_batch
        n_samples = theta_layers.shape[0]
        psi = np.zeros((n_samples, dim), dtype=np.complex128)
        psi[:, 0] = 1.0
        if prep is not None:
            psi = psi @ prep.T
        for layer in range(n_layers):
            angles = theta_layers[:, layer, :]
            if ent_first:
                if n_qubits >= 2:
                    psi = psi @ entangler.T
                psi = _apply_rx_layer_vec(psi, angles, n_qubits)
            else:
                psi = _apply_rx_layer_vec(psi, angles, n_qubits)
                if n_qubits >= 2:
                    psi = psi @ entangler.T
        return (psi * np.conj(psi)).real

    return probs_of


def qubit_marginals(probs: np.ndarray, n_qubits: int) -> np.ndarray:
    """``P(bit_q = 1)`` for each qubit, from a batch of outcome distributions."""
    idx = np.arange(probs.shape[1])
    out = np.empty((probs.shape[0], n_qubits), dtype=np.float64)
    for q in range(n_qubits):
        mask = (idx >> (n_qubits - 1 - q)) & 1
        out[:, q] = probs[:, mask == 1].sum(axis=1)
    return out


def make_ansatz(
    name: str, n_qubits: int
) -> Callable[[np.ndarray, int, np.random.Generator], np.ndarray]:
    """Return a function ``f(theta_batch, n_layers, rng) -> (n_samples, n_qubits) bits``.

    ``theta_batch`` has shape ``(n_samples, n_qubits)`` if ``n_layers == 1``,
    otherwise ``(n_samples, n_layers, n_qubits)``.  Initial state is
    ``|0...0>`` and measurement is in the computational basis.  Gate order
    follows Fig. 2 of the paper: the CNOT ansätze apply ``RX(theta)`` then the
    entangling network, while ``cz2`` prepares ``|++>``, applies the CZ, and
    only then applies ``RX(theta)`` (see ``entangler_precedes_rotations``).
    """
    probs_of = make_ansatz_probs(name, n_qubits)

    def run(
        theta_batch: np.ndarray, n_layers: int, rng: np.random.Generator
    ) -> np.ndarray:
        return _measure_bitstring(probs_of(theta_batch, n_layers), n_qubits, rng)

    return run


def number_of_gate_params(name: str, n_qubits: int) -> int:
    return n_qubits


__all__ = ["make_ansatz", "number_of_gate_params"]
