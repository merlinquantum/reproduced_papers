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


def _apply_rx_layer(psi: np.ndarray, angles: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply tensor_q RX(theta_q) to a batch of states.

    ``psi`` has shape ``(n_samples, 2**n_qubits)``; ``angles`` is
    ``(n_samples, n_qubits)``.  Returns the updated psi.
    """
    n_samples = psi.shape[0]
    u_each = _per_qubit_rx_batch(angles)  # (n_qubits, n_samples, 2, 2)
    # Reshape psi to a tensor with one axis per qubit, plus the batch axis.
    psi = psi.reshape((n_samples,) + (2,) * n_qubits)
    for q in range(n_qubits):
        # Apply 2x2 on axis (q+1), batched over n_samples (axis 0).
        # einsum over psi: indices ('n', q-axis-then-rest)
        # New psi axis q: U @ psi along that axis.
        psi = np.einsum('nij,n...j...->n...i...', u_each[q], psi, optimize=False)\
            if False else None  # placeholder, see specialised code below
        raise NotImplementedError  # never executed; replaced below
    return psi  # unreachable


def _apply_rx_layer_vec(psi: np.ndarray, angles: np.ndarray, n_qubits: int) -> np.ndarray:
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
        U = (
            U[:, :, None, :, None] * u_each[q][:, None, :, None, :]
        ).reshape(n_samples, d1 * d2, d1 * d2)
    return np.einsum('nij,nj->ni', U, psi)


def _cnot_matrix(n_qubits: int, control: int, target: int) -> np.ndarray:
    dim = 2 ** n_qubits
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
    dim = 2 ** n_qubits
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
            (0, 4), (1, 5), (2, 6), (3, 7),
            (0, 2), (1, 3), (4, 6), (5, 7),
            (0, 1), (2, 3), (4, 5), (6, 7),
        ]
        U = np.eye(2 ** 8, dtype=np.complex128)
        for c, t in cnots:
            U = _cnot_matrix(8, c, t) @ U
        return U
    raise ValueError(f"Unknown ansatz: {name}")


def make_ansatz(name: str, n_qubits: int) -> Callable[[np.ndarray, int, np.random.Generator], np.ndarray]:
    """Return a function ``f(theta_batch, n_layers, rng) -> (n_samples, n_qubits) bits``.

    ``theta_batch`` has shape ``(n_samples, n_qubits)`` if ``n_layers == 1``,
    otherwise ``(n_samples, n_layers, n_qubits)``.  Each layer is a new
    ``RX(theta)`` immediately followed by the same fixed entangling unitary.
    Initial state is ``|0...0>``; measurement is in the computational basis.
    """
    entangler = _build_entangler(name, n_qubits)
    dim = 2 ** n_qubits

    def run(theta_batch: np.ndarray, n_layers: int, rng: np.random.Generator) -> np.ndarray:
        if n_layers == 1 and theta_batch.ndim == 2:
            theta_layers = theta_batch[:, None, :]
        else:
            theta_layers = theta_batch
        n_samples = theta_layers.shape[0]
        psi = np.zeros((n_samples, dim), dtype=np.complex128)
        psi[:, 0] = 1.0
        for layer in range(n_layers):
            angles = theta_layers[:, layer, :]
            psi = _apply_rx_layer_vec(psi, angles, n_qubits)
            if n_qubits >= 2:
                psi = psi @ entangler.T
        probs = (psi * np.conj(psi)).real
        return _measure_bitstring(probs, n_qubits, rng)

    return run


def number_of_gate_params(name: str, n_qubits: int) -> int:
    return n_qubits


__all__ = ["make_ansatz", "number_of_gate_params"]
