"""Fock-state amplitudes and post-selected conditional states for linear optics.

Modes are 0-indexed here. The paper numbers modes from 1; ``circuits.py`` does the
translation. A Fock configuration is a tuple of occupation numbers, one per mode.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np


def permanent(matrix: np.ndarray) -> complex:
    """Return the permanent of a square complex matrix (Ryser formula).

    Parameters
    ----------
    matrix : numpy.ndarray
        Square matrix of shape ``(n, n)``.

    Returns
    -------
    complex
        Permanent of ``matrix``; 1 for the empty ``(0, 0)`` matrix.
    """
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        raise ValueError(f"permanent needs a square matrix, got {matrix.shape}")
    if size == 0:
        return 1.0 + 0.0j
    total = 0.0 + 0.0j
    for subset_size in range(1, size + 1):
        sign = (-1) ** subset_size
        for columns in itertools.combinations(range(size), subset_size):
            total += sign * np.prod(matrix[:, columns].sum(axis=1))
    return (-1) ** size * total


def modes_from_occupation(occupation: Sequence[int]) -> list[int]:
    """Expand an occupation tuple into the list of occupied mode indices.

    ``(0, 2, 1)`` becomes ``[1, 1, 2]``: the row/column multiset used to build the
    submatrix whose permanent gives the transition amplitude.
    """
    return [mode for mode, count in enumerate(occupation) for _ in range(count)]


def transition_amplitude(
    unitary: np.ndarray,
    input_occupation: Sequence[int],
    output_occupation: Sequence[int],
) -> complex:
    """Return <output| U |input> for indistinguishable photons.

    Uses amplitude = perm(U[out, in]) / sqrt(prod(out!) prod(in!)), where ``out``
    and ``in`` are the occupied-mode multisets.

    Parameters
    ----------
    unitary : numpy.ndarray
        Mode unitary of shape ``(m, m)``; column ``j`` maps input mode ``j``.
    input_occupation, output_occupation : sequence of int
        Occupation numbers of length ``m``.

    Returns
    -------
    complex
        Transition amplitude; 0 when the photon numbers differ.
    """
    if sum(input_occupation) != sum(output_occupation):
        return 0.0 + 0.0j
    rows = modes_from_occupation(output_occupation)
    columns = modes_from_occupation(input_occupation)
    norm = math.prod(math.factorial(n) for n in output_occupation) * math.prod(
        math.factorial(n) for n in input_occupation
    )
    return permanent(unitary[np.ix_(rows, columns)]) / math.sqrt(norm)


def fock_basis(n_modes: int, n_photons: int) -> list[tuple[int, ...]]:
    """Return all occupations of ``n_photons`` over ``n_modes`` in a fixed order.

    The order is reverse-lexicographic in the occupation tuple, so for one photon
    over modes (a, b, c) the basis is (1,0,0), (0,1,0), (0,0,1): the photon in the
    first listed mode comes first. Dimension is C(n_modes + n_photons - 1, n_photons).
    """
    basis = [
        occupation
        for occupation in itertools.product(range(n_photons + 1), repeat=n_modes)
        if sum(occupation) == n_photons
    ]
    return sorted(basis, reverse=True)


def conditional_state(
    unitary: np.ndarray,
    input_modes: Sequence[int],
    detected_modes: Sequence[int],
    detected_counts: Sequence[int],
    residual_modes: Sequence[int],
) -> tuple[np.ndarray, float]:
    """Return the residual-mode state conditioned on a detection pattern.

    This is the ABS feature map: inject one photon in each ``input_modes`` entry,
    evolve with ``unitary``, post-select ``detected_counts`` on ``detected_modes``
    and keep the normalised state of the remaining photons on ``residual_modes``.

    Parameters
    ----------
    unitary : numpy.ndarray
        Full ``(m, m)`` mode unitary for the circuit programmed for this outcome.
    input_modes : sequence of int
        One single photon per listed mode (0-indexed; no repeats).
    detected_modes : sequence of int
        Modes measured by the adaptive detectors.
    detected_counts : sequence of int
        Photon number recorded on each detected mode (the string ``p``).
    residual_modes : sequence of int
        Unmeasured modes that carry the encoded state. Detected and residual modes
        must partition all ``m`` modes.

    Returns
    -------
    state : numpy.ndarray
        Normalised complex amplitudes over ``fock_basis(len(residual_modes), n-r)``.
    probability : float
        Post-selection probability of the detection pattern (state norm squared).

    Raises
    ------
    ValueError
        If the modes do not partition the circuit or the pattern has zero
        probability.
    """
    n_modes = unitary.shape[0]
    if sorted([*detected_modes, *residual_modes]) != list(range(n_modes)):
        raise ValueError("detected and residual modes must partition all modes")
    if len(set(input_modes)) != len(input_modes):
        raise ValueError("input_modes must hold one photon per mode")
    if len(detected_modes) != len(detected_counts):
        raise ValueError("one detected count per detected mode is required")
    n_residual = len(input_modes) - sum(detected_counts)
    if n_residual < 0:
        raise ValueError("more photons detected than injected")

    input_occupation = [0] * n_modes
    for mode in input_modes:
        input_occupation[mode] = 1

    basis = fock_basis(len(residual_modes), n_residual)
    amplitudes = np.zeros(len(basis), dtype=np.complex128)
    for index, residual_occupation in enumerate(basis):
        output_occupation = [0] * n_modes
        for mode, count in zip(detected_modes, detected_counts):
            output_occupation[mode] = count
        for mode, count in zip(residual_modes, residual_occupation):
            output_occupation[mode] = count
        amplitudes[index] = transition_amplitude(
            unitary, input_occupation, output_occupation
        )
    probability = float(np.vdot(amplitudes, amplitudes).real)
    if probability <= 1e-14:
        raise ValueError(
            f"detection pattern {tuple(detected_counts)} has zero probability"
        )
    return amplitudes / math.sqrt(probability), probability
