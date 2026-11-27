"""Exact low-order photon-number expectations from permanents (no state vector).

For a collision-free Fock input (at most one photon per mode) the normally
ordered moment is a sum over input-photon subsets,

    <a^+_{j_1} ... a^+_{j_p} a_{j_p} ... a_{j_1}> = sum_{I, |I| = p} |Perm(U[I, J])|^2,

with ``U`` in the paper convention (rows are input modes) and ``J`` the list of
output modes (repeats allowed). Powers of a number operator expand into normally
ordered products with Stirling numbers, ``n^a = sum_p S(a, p) (a^+)^p a^p``, so
any PNM of order ``r`` costs ``C(n, p)`` permanents of size ``p <= r``. This is
polynomial for fixed order; it is the exact classical evaluation behind
baseline B2 and the exact loss monitor of the large training runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations, product

import numpy as np

from .combinatorics import stirling2
from .observables import Monomial
from .photonic_sim import permanent


def normally_ordered_moment(
    unitary: np.ndarray, occupation: Sequence[int], output_modes: Sequence[int]
) -> float:
    """``<prod_t a^+_{j_t} prod_t a_{j_t}>`` for a collision-free Fock input.

    Parameters
    ----------
    unitary : numpy.ndarray
        Paper-convention interferometer.
    occupation : sequence of int
        Fock input with entries 0 or 1.
    output_modes : sequence of int
        Output mode list ``J`` (repeats allowed); empty gives 1.

    Returns
    -------
    float
        The normally ordered moment.

    Raises
    ------
    ValueError
        If the input has a mode with more than one photon.
    """
    if max(occupation) > 1:
        raise ValueError("normally_ordered_moment assumes a collision-free input")
    order = len(output_modes)
    if order == 0:
        return 1.0
    occupied = [mode for mode, count in enumerate(occupation) if count == 1]
    columns = list(output_modes)
    total = 0.0
    for subset in combinations(occupied, order):
        total += abs(permanent(unitary[np.ix_(list(subset), columns)])) ** 2
    return float(total)


def monomial_expectation(
    unitary: np.ndarray, occupation: Sequence[int], monomial: Monomial
) -> float:
    """Exact ``<prod_i n_i^{alpha_i}>`` via Stirling expansion into normal order."""
    expansion_choices = [
        [(power, stirling2(exponent, power)) for power in range(1, exponent + 1)]
        for _, exponent in monomial
    ]
    total = 0.0
    for choice in product(*expansion_choices):
        weight = 1
        output_modes: list[int] = []
        for (mode, _), (power, stirling) in zip(monomial, choice):
            weight *= stirling
            output_modes.extend([mode] * power)
        if len(output_modes) > sum(occupation):
            continue
        total += weight * normally_ordered_moment(unitary, occupation, output_modes)
    return total


def polynomial_expectation(
    unitary: np.ndarray, occupation: Sequence[int], polynomial: Mapping[Monomial, float]
) -> float:
    """Exact ``<C>`` of a photon-number polynomial (sum of monomial expectations)."""
    return float(
        sum(
            coefficient * monomial_expectation(unitary, occupation, monomial)
            for monomial, coefficient in polynomial.items()
        )
    )
