"""Independent small-photon probability reference using permutation sums."""

import itertools
import math

import numpy as np


def fock_states(modes: int, photons: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate fixed-number Fock occupations in lexicographic order.

    Parameters
    ----------
    modes : int
        Positive mode count.
    photons : int
        Nonnegative photon number.

    Returns
    -------
    tuple
        Complete, sorted occupation basis.
    """
    if modes < 1 or photons < 0:
        raise ValueError("Invalid mode or photon count")
    result = []
    for positions in itertools.combinations_with_replacement(range(modes), photons):
        state = [0] * modes
        for position in positions:
            state[position] += 1
        result.append(tuple(state))
    return tuple(sorted(result))


def theoretical_distribution(
    unitary: np.ndarray,
    input_state: tuple[int, ...],
    survival: float = 1.0,
    hom: float = 1.0,
    threshold: bool = True,
) -> dict[tuple[int, ...], float]:
    """Compute a source-loss mixture without using either optical simulator.

    Parameters
    ----------
    unitary : numpy.ndarray
        Complex mode transformation, rows output and columns input.
    input_state : tuple
        Single-photon inputs (each occupation zero or one).
    survival : float
        Independent photon survival probability. Default value is 1.0.
    hom : float
        Pairwise squared internal overlap (HOM visibility). Default value is 1.0.
    threshold : bool
        Collapse occupations to clicks after loss. Default value is True.

    Returns
    -------
    dict
        Probabilities including vacuum if loss is active.

    Notes
    -----
    For surviving labelled photons the double permutation formula uses
    Gram off-diagonals sqrt(hom). At hom=1 this is the permanent formula.
    This exponential reference is restricted to small photon counts and g2=0.
    """
    modes = len(input_state)
    unitary = np.asarray(unitary, dtype=complex)
    if unitary.shape != (modes, modes):
        raise ValueError("Unitary dimensions differ from input")
    if not np.allclose(unitary.conj().T @ unitary, np.eye(modes), atol=1e-10, rtol=0):
        raise ValueError("Matrix is not unitary")
    if any(value not in (0, 1) for value in input_state):
        raise ValueError("Reference supports only single-photon input modes")
    if not 0 <= survival <= 1 or not 0 <= hom <= 1:
        raise ValueError("survival and hom must be probabilities")
    occupied = np.flatnonzero(input_state)
    photons = len(occupied)
    distribution = {}
    for count in range(photons + 1):
        weight = survival**count * (1 - survival) ** (photons - count)
        if weight == 0:
            continue
        permutations = np.array(list(itertools.permutations(range(count))), dtype=int)
        if count:
            mismatches = (permutations[:, None, :] != permutations[None, :, :]).sum(
                axis=2
            )
            distinguishability = np.sqrt(hom) ** mismatches
        for inputs in itertools.combinations(occupied, count):
            for state in fock_states(modes, count):
                if count:
                    outputs = np.repeat(np.arange(modes), state)
                    submatrix = unitary[np.ix_(outputs, inputs)]
                    paths = submatrix[np.arange(count)[None, :], permutations].prod(
                        axis=1
                    )
                    probability = float(
                        np.real(paths @ distinguishability @ paths.conj())
                    )
                    probability /= math.prod(math.factorial(value) for value in state)
                    if probability < -1e-12:
                        raise RuntimeError(
                            "Negative probability in permutation reference"
                        )
                else:
                    probability = 1.0
                key = tuple(int(value > 0) for value in state) if threshold else state
                distribution[key] = distribution.get(key, 0.0) + weight * max(
                    probability, 0.0
                )
    if not np.isclose(sum(distribution.values()), 1, atol=1e-9):
        raise RuntimeError("Permutation reference is not normalized")
    return distribution


def total_variation(first: dict, second: dict) -> float:
    """Return total variation distance for distributions with arbitrary keys.

    Parameters
    ----------
    first, second : dict
        Normalized distributions over the same physical outcome space.

    Returns
    -------
    float
        Half the sum of absolute probability differences.
    """
    return 0.5 * sum(
        abs(first.get(key, 0) - second.get(key, 0))
        for key in first.keys() | second.keys()
    )
