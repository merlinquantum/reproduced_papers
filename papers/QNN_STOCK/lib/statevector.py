"""Minimal exact statevector simulator (NumPy, complex128).

Qubit 0 is the most significant bit of the basis index (big-endian), so
``|q0 q1 ... q_{n-1}>`` has index ``q0 * 2**(n-1) + ... + q_{n-1}``.
The paper's circuits have at most 13 qubits, so dense states are cheap.
"""

from __future__ import annotations

import numpy as np


def basis_state(index: int, n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[index] = 1.0
    return state


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=np.complex128
    )


def apply_1q(
    state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    psi = state.reshape((2,) * n_qubits)
    psi = np.tensordot(gate, psi, axes=([1], [qubit]))
    return np.moveaxis(psi, 0, qubit).reshape(-1)


def apply_cnot(
    state: np.ndarray, control: int, target: int, n_qubits: int
) -> np.ndarray:
    psi = state.reshape((2,) * n_qubits).copy()
    selector = [slice(None)] * n_qubits
    selector[control] = 1
    # Indexing removes the control axis, so the target axis shifts down by one
    # when it comes after the control.
    target_axis = target - 1 if target > control else target
    psi[tuple(selector)] = np.flip(psi[tuple(selector)], axis=target_axis)
    return psi.reshape(-1)


def apply_cswap(
    state: np.ndarray, control: int, qubit_a: int, qubit_b: int, n_qubits: int
) -> np.ndarray:
    """Fredkin gate; used to check the analytic SWAP test against the circuit."""

    psi = state.reshape((2,) * n_qubits).copy()
    selector = [slice(None)] * n_qubits
    selector[control] = 1
    axis_a = qubit_a - 1 if qubit_a > control else qubit_a
    axis_b = qubit_b - 1 if qubit_b > control else qubit_b
    psi[tuple(selector)] = np.swapaxes(psi[tuple(selector)], axis_a, axis_b)
    return psi.reshape(-1)


def probabilities(state: np.ndarray) -> np.ndarray:
    return np.abs(state) ** 2
