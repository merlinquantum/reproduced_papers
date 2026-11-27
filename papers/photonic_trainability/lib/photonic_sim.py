"""Linear-optical simulation: Haar interferometers, output distributions, shots.

Exact output distributions come from Perceval's SLOS backend. A plain Ryser
permanent is kept as an independent reference for tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import factorial

import numpy as np
import perceval as pcvl

from .fock import fock_basis


def haar_unitary(modes: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a Haar-random ``U in U(M)`` (QR of a Ginibre matrix with phase fix).

    Parameters
    ----------
    modes : int
        Matrix size ``M``.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    numpy.ndarray
        Complex ``(M, M)`` unitary.
    """
    ginibre = (
        rng.standard_normal((modes, modes)) + 1j * rng.standard_normal((modes, modes))
    ) / np.sqrt(2.0)
    q_factor, r_factor = np.linalg.qr(ginibre)
    phases = np.diag(r_factor) / np.abs(np.diag(r_factor))
    return q_factor * phases


def output_distribution(
    unitary: np.ndarray, occupation: Sequence[int], basis: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact output distribution of a Fock input through ``unitary`` (Perceval SLOS).

    Parameters
    ----------
    unitary : numpy.ndarray
        ``(M, M)`` interferometer matrix, acting as ``a_i^+ -> sum_j U_ij a_j^+``
        (paper Eq. 5).
    occupation : sequence of int
        Input Fock state.
    basis : numpy.ndarray | None
        Output patterns to report, normally ``fock_basis(n, M)``; built if
        omitted. Default value is None.

    Returns
    -------
    patterns : numpy.ndarray
        ``(D, M)`` integer patterns.
    probabilities : numpy.ndarray
        ``(D,)`` probabilities aligned with ``patterns``.
    """
    modes = len(occupation)
    photons = int(sum(occupation))
    patterns = fock_basis(photons, modes) if basis is None else basis
    backend = pcvl.BackendFactory.get_backend("SLOS")
    # Perceval's Unitary maps input mode j to output mode i with U[i, j]; the
    # paper's convention a_i^+ -> sum_j U_ij a_j^+ is the transpose. Haar
    # averages do not depend on this, but single-unitary tests do.
    backend.set_circuit(pcvl.Unitary(pcvl.Matrix(np.ascontiguousarray(unitary.T))))
    backend.set_input_state(pcvl.BasicState(list(occupation)))
    distribution = backend.prob_distribution()
    lookup = {
        tuple(state): float(probability) for state, probability in distribution.items()
    }
    probabilities = np.array(
        [lookup.get(tuple(int(value) for value in row), 0.0) for row in patterns]
    )
    return patterns, probabilities


def permanent(matrix: np.ndarray) -> complex:
    """Permanent by Ryser's formula; exponential cost, test-sized matrices only."""
    size = matrix.shape[0]
    if size == 0:
        return 1.0 + 0.0j
    total = 0.0 + 0.0j
    for subset in product((0, 1), repeat=size):
        if not any(subset):
            continue
        columns = [index for index, chosen in enumerate(subset) if chosen]
        row_sums = matrix[:, columns].sum(axis=1)
        total += (-1) ** len(columns) * np.prod(row_sums)
    return (-1) ** size * total


def output_probability_permanent(
    unitary: np.ndarray, occupation: Sequence[int], pattern: Sequence[int]
) -> float:
    """``|Perm(U_{s,t})|^2 / (prod s_i! prod t_i!)`` in the paper's convention.

    Row ``i`` of ``U`` is repeated ``s_i`` times and column ``j`` ``t_j`` times.
    """
    rows = [mode for mode, count in enumerate(occupation) for _ in range(count)]
    columns = [mode for mode, count in enumerate(pattern) for _ in range(count)]
    submatrix = unitary[np.ix_(rows, columns)]
    normalisation = np.prod([factorial(count) for count in occupation]) * np.prod(
        [factorial(count) for count in pattern]
    )
    return float(abs(permanent(submatrix)) ** 2 / normalisation)


def sample_patterns(
    patterns: np.ndarray,
    probabilities: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``shots`` measurement outcomes from an exact output distribution.

    Parameters
    ----------
    patterns : numpy.ndarray
        ``(D, M)`` output patterns.
    probabilities : numpy.ndarray
        ``(D,)`` probabilities; renormalised to absorb rounding error.
    shots : int
        Number of photon-number-resolving measurement shots.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    numpy.ndarray
        ``(shots, M)`` sampled patterns.
    """
    weights = np.clip(probabilities, 0.0, None)
    weights = weights / weights.sum()
    indices = rng.choice(len(patterns), size=shots, p=weights)
    return patterns[indices]
