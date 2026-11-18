"""Minimal MPS bond dimension of an exact statevector (paper App. "Hyperparameter search").

The paper takes the smallest bond dimension ``chi`` whose MPS reconstruction reaches
fidelity ``F >= 1 - 5e-4`` but does not name the MPS algorithm or the qubit ordering on
the ring. Here: left-to-right TT-SVD with truncation to ``chi`` at every cut, open-chain
ordering ``0, ..., N-1``, and ``F = |<psi|phi>|^2 / <phi|phi>`` computed exactly.
"""

from __future__ import annotations

import numpy as np


def truncated_mps_fidelity(state: np.ndarray, n_qubits: int, chi: int) -> float:
    """Fidelity of the TT-SVD MPS of ``state`` truncated to bond dimension ``chi``.

    Parameters
    ----------
    state : numpy.ndarray
        Normalised statevector of length ``2**n_qubits`` (qubit 0 most significant).
    n_qubits : int
        ``N``.
    chi : int
        Maximum bond dimension.

    Returns
    -------
    float
        ``|<psi|phi>|^2 / <phi|phi>``.
    """
    remainder = state.reshape(1, -1)
    cores = []
    for _ in range(n_qubits - 1):
        left_dim = remainder.shape[0]
        matrix = remainder.reshape(left_dim * 2, -1)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        keep = min(chi, s.size)
        cores.append(u[:, :keep].reshape(left_dim, 2, keep))
        remainder = s[:keep, None] * vh[:keep]
    cores.append(remainder.reshape(remainder.shape[0], 2, 1))

    reconstructed = cores[0].reshape(2, -1)
    for core in cores[1:]:
        reconstructed = (reconstructed @ core.reshape(core.shape[0], -1)).reshape(
            -1, core.shape[2]
        )
    reconstructed = reconstructed.reshape(-1)
    overlap = np.vdot(state, reconstructed)
    return float(abs(overlap) ** 2 / np.vdot(reconstructed, reconstructed).real)


def minimal_bond_dimension(
    state: np.ndarray, n_qubits: int, fidelity_threshold: float
) -> int:
    """Smallest ``chi`` with ``truncated_mps_fidelity >= fidelity_threshold``.

    Binary search over ``[1, 2**(N // 2)]`` (the exact maximum, fidelity 1). This
    assumes the TT-SVD fidelity is non-decreasing in ``chi``; ``tests/test_mps_data.py``
    checks the search against a linear scan.
    """
    low, high = 1, 2 ** (n_qubits // 2)
    while low < high:
        middle = (low + high) // 2
        if truncated_mps_fidelity(state, n_qubits, middle) >= fidelity_threshold:
            high = middle
        else:
            low = middle + 1
    return low
