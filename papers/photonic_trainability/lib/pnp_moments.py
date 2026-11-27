"""Exact Haar moments of large photon-number polynomials (paper Fig. 5).

``moments.polynomial_variances`` enumerates every lowered pattern
``a in Phi^l_M`` and every term, which is exact but too slow for the Fig. 5
sizes (up to ~4000 terms at M = 90, or 300 order-4 terms at M = 60). This
module computes the same exact quantities faster:

* ``first_moment_of_square`` groups the ``|A|^2`` term pairs: disjoint pairs only
  depend on the two exponent patterns, so they are summed per pattern pair, and
  only overlapping pairs are visited one by one.
* ``g_values_sparse`` (levels ``l <= 2``) writes ``w_C(a)`` as a common baseline
  plus corrections from the few terms whose support touches ``a``.
* ``g_values_pairs`` (any level) uses
  ``sum_a w_C(a)^2 = sum_{alpha,beta} c_alpha c_beta G_l(alpha, beta)``, where
  ``G_l`` depends only on the multiset of exponent pairs over the union of the
  two supports (mode relabelling invariance), so it is cached per key.

Coefficients are converted exactly to fractions and then to integers with a
common denominator, so every result stays exact. ``tests/test_pnp_moments.py``
checks both paths against the enumeration code.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import cache
from math import comb, factorial, lcm

from . import moments
from .combinatorics import compositions
from .observables import (
    Monomial,
    monomial_exponents,
    multiply_monomials,
    polynomial_order,
)


def _integer_coefficients(
    polynomial: Mapping[Monomial, float | Fraction],
) -> tuple[list[tuple[Monomial, int]], int]:
    """Scale coefficients to integers; return ``(terms, common_denominator)``."""
    fractions = {monomial: Fraction(value) for monomial, value in polynomial.items()}
    denominator = lcm(*(value.denominator for value in fractions.values()))
    terms = [
        (monomial, int(value * denominator))
        for monomial, value in fractions.items()
        if value != 0
    ]
    if any(not monomial for monomial, _ in terms):
        raise ValueError("Constant terms carry no variance; drop them before calling")
    return terms, denominator


def first_moment_of_square(
    polynomial: Mapping[Monomial, float | Fraction], photons: int, modes: int
) -> Fraction:
    """Exact ``mu1(C^2) = sum_{alpha,beta} c_alpha c_beta mu1(alpha + beta)``.

    Parameters
    ----------
    polynomial : Mapping[Monomial, float | Fraction]
        Photon-number polynomial without a constant term.
    photons, modes : int
        Sector ``(n, M)``.

    Returns
    -------
    fractions.Fraction
        The exact first moment of the squared polynomial.
    """
    terms, denominator = _integer_coefficients(polynomial)
    pattern_sums: dict[tuple[int, ...], int] = defaultdict(int)
    terms_by_mode: dict[int, list[int]] = defaultdict(list)
    patterns = []
    for index, (monomial, coefficient) in enumerate(terms):
        pattern = tuple(sorted(monomial_exponents(monomial)))
        patterns.append(pattern)
        pattern_sums[pattern] += coefficient
        for mode, _ in monomial:
            terms_by_mode[mode].append(index)

    @cache
    def first_moment(pattern: tuple[int, ...]) -> Fraction:
        return moments.first_moment_exponents(pattern, photons, modes)

    total = Fraction(0)
    # Treat every ordered pair as disjoint first ...
    for left_pattern, left_sum in pattern_sums.items():
        for right_pattern, right_sum in pattern_sums.items():
            merged = tuple(sorted(left_pattern + right_pattern))
            if len(merged) > modes:
                continue
            total += left_sum * right_sum * first_moment(merged)
    # ... then correct the ordered pairs whose supports overlap.
    for left_index, (left, left_coefficient) in enumerate(terms):
        partners = {
            right_index for mode, _ in left for right_index in terms_by_mode[mode]
        }
        for right_index in partners:
            right, right_coefficient = terms[right_index]
            disjoint_pattern = tuple(
                sorted(patterns[left_index] + patterns[right_index])
            )
            correction = first_moment(
                tuple(sorted(monomial_exponents(multiply_monomials(left, right))))
            )
            if len(disjoint_pattern) <= modes:
                correction -= first_moment(disjoint_pattern)
            total += left_coefficient * right_coefficient * correction
    return total / denominator**2


def _weight(
    exponents: tuple[int, ...],
    lowered_support: tuple[int, ...],
    lowered_rest: int,
    photons: int,
    modes: int,
) -> int:
    return moments._monomial_lowering_weight(
        exponents, lowered_support, lowered_rest, photons, modes
    )


def g_values_sparse(
    polynomial: Mapping[Monomial, float | Fraction],
    photons: int,
    modes: int,
    max_level: int,
) -> list[Fraction]:
    """Exact ``g_l(C)`` for ``l <= max_level <= 2`` via baseline-plus-corrections.

    Parameters
    ----------
    polynomial : Mapping[Monomial, float | Fraction]
        Photon-number polynomial without a constant term.
    photons, modes : int
        Sector ``(n, M)``.
    max_level : int
        Largest level, at most 2.

    Returns
    -------
    list of fractions.Fraction
        ``g_0, ..., g_max_level``.
    """
    if max_level > 2:
        raise ValueError("The sparse path covers levels up to 2; use g_values_pairs")
    terms, denominator = _integer_coefficients(polynomial)
    parsed = [
        (tuple(monomial_exponents(monomial)), tuple(mode for mode, _ in monomial), c)
        for monomial, c in terms
    ]
    values = []
    # Level 0: a = 0.
    weight = sum(c * _weight(e, (0,) * len(e), 0, photons, modes) for e, _, c in parsed)
    values.append(factorial(photons) ** 2 * weight * weight)
    if max_level >= 1:
        baseline = 0
        corrections = [0] * modes
        for exponents, support, c in parsed:
            outside = _weight(exponents, (0,) * len(exponents), 1, photons, modes)
            baseline += c * outside
            for position, mode in enumerate(support):
                lowered = tuple(int(i == position) for i in range(len(support)))
                corrections[mode] += c * (
                    _weight(exponents, lowered, 0, photons, modes) - outside
                )
        total = sum((baseline + correction) ** 2 for correction in corrections)
        values.append(factorial(photons - 1) ** 2 * total)
    if max_level >= 2:
        baseline = 0
        diagonal = [0] * modes  # a = 2 e_j
        row_offsets = [0] * modes  # a = e_j + e_k with j in S, k outside S
        pair_corrections: dict[tuple[int, int], int] = defaultdict(int)
        for exponents, support, c in parsed:
            size = len(support)
            outside = _weight(exponents, (0,) * size, 2, photons, modes)
            baseline += c * outside
            for position, mode in enumerate(support):
                double = tuple(2 * int(i == position) for i in range(size))
                single = tuple(int(i == position) for i in range(size))
                diagonal[mode] += c * (
                    _weight(exponents, double, 0, photons, modes) - outside
                )
                one_inside = c * (
                    _weight(exponents, single, 1, photons, modes) - outside
                )
                # Every partner k outside S gets this correction; partners inside S
                # are fixed below, after removing what this row offset added.
                row_offsets[mode] += one_inside
                for other_position, other in enumerate(support):
                    if other_position == position:
                        continue
                    both = tuple(
                        int(i == position or i == other_position) for i in range(size)
                    )
                    key = (min(mode, other), max(mode, other))
                    if position < other_position:
                        pair_corrections[key] += c * (
                            _weight(exponents, both, 0, photons, modes) - outside
                        )
                    pair_corrections[key] -= one_inside
        total = sum((baseline + value) ** 2 for value in diagonal)
        # Off-diagonal pairs j < k: correction = row_offsets[j] + row_offsets[k]
        # (each mode's "one photon inside" correction) plus explicit fixes.
        for first in range(modes):
            for second in range(first + 1, modes):
                value = (
                    baseline
                    + row_offsets[first]
                    + row_offsets[second]
                    + pair_corrections.get((first, second), 0)
                )
                total += value * value
        values.append(factorial(photons - 2) ** 2 * total)
    return [Fraction(value, denominator**2) for value in values]


def _pair_key(left: Monomial, right: Monomial) -> tuple[tuple[int, int], ...]:
    left_map, right_map = dict(left), dict(right)
    union = set(left_map) | set(right_map)
    return tuple(
        sorted((left_map.get(mode, 0), right_map.get(mode, 0)) for mode in union)
    )


@cache
def _pair_overlap(
    key: tuple[tuple[int, int], ...], level: int, photons: int, modes: int
) -> int:
    """``G_l = sum_{a in Phi^l_M} w_alpha(a) w_beta(a)`` for a canonical pair key."""
    union_size = len(key)
    rest_modes = modes - union_size
    left_positions = [i for i, (e, _) in enumerate(key) if e > 0]
    right_positions = [i for i, (_, e) in enumerate(key) if e > 0]
    left_exponents = tuple(key[i][0] for i in left_positions)
    right_exponents = tuple(key[i][1] for i in right_positions)
    total = 0
    for union_level in range(level + 1):
        rest_level = level - union_level
        if rest_modes == 0:
            multiplicity = 1 if rest_level == 0 else 0
        else:
            multiplicity = comb(rest_level + rest_modes - 1, rest_modes - 1)
        if multiplicity == 0:
            continue
        for lowered in compositions(union_level, union_size):
            left_lowered = tuple(lowered[i] for i in left_positions)
            right_lowered = tuple(lowered[i] for i in right_positions)
            left_weight = _weight(
                left_exponents, left_lowered, level - sum(left_lowered), photons, modes
            )
            if left_weight == 0:
                continue
            right_weight = _weight(
                right_exponents,
                right_lowered,
                level - sum(right_lowered),
                photons,
                modes,
            )
            total += multiplicity * left_weight * right_weight
    return total


def g_values_pairs(
    polynomial: Mapping[Monomial, float | Fraction],
    photons: int,
    modes: int,
    max_level: int,
) -> list[Fraction]:
    """Exact ``g_l(C)`` for any level via cached pairwise overlaps.

    Cost is ``|A|^2`` key lookups plus one overlap evaluation per distinct key.

    Parameters
    ----------
    polynomial : Mapping[Monomial, float | Fraction]
        Photon-number polynomial without a constant term.
    photons, modes : int
        Sector ``(n, M)``.
    max_level : int
        Largest level required.

    Returns
    -------
    list of fractions.Fraction
        ``g_0, ..., g_max_level``.
    """
    terms, denominator = _integer_coefficients(polynomial)
    key_weights: dict[tuple[tuple[int, int], ...], int] = defaultdict(int)
    for left, left_coefficient in terms:
        for right, right_coefficient in terms:
            key_weights[_pair_key(left, right)] += left_coefficient * right_coefficient
    values = []
    for level in range(max_level + 1):
        total = sum(
            weight * _pair_overlap(key, level, photons, modes)
            for key, weight in key_weights.items()
            if weight != 0
        )
        values.append(Fraction(factorial(photons - level) ** 2 * total, denominator**2))
    return values


def polynomial_variances(
    polynomial: Mapping[Monomial, float | Fraction],
    occupation: Sequence[int],
    method: str = "auto",
) -> dict[str, Fraction]:
    """Exact moments and variances of a large PNP for a Fock input.

    Parameters
    ----------
    polynomial : Mapping[Monomial, float | Fraction]
        Photon-number polynomial without a constant term.
    occupation : sequence of int
        Fock input; its length is ``M``.
    method : str
        ``"sparse"`` (order <= 2), ``"pairs"`` or ``"auto"`` (sparse when it
        applies). Default value is ``"auto"``.

    Returns
    -------
    dict
        Same keys as :func:`moments.monomial_variances`.
    """
    photons, modes = sum(occupation), len(occupation)
    top_level = min(photons, polynomial_order(polynomial))
    if method == "auto":
        method = "sparse" if top_level <= 2 else "pairs"
    if method == "sparse":
        observable_g = g_values_sparse(polynomial, photons, modes, top_level)
    elif method == "pairs":
        observable_g = g_values_pairs(polynomial, photons, modes, top_level)
    else:
        raise ValueError(f"Unknown method {method!r}")
    mu1 = moments.first_moment_polynomial(polynomial, photons, modes)
    mu2 = moments.second_moment_from_g(
        moments.g_values_fock_state(occupation, top_level), observable_g, photons, modes
    )
    mu1_square = first_moment_of_square(polynomial, photons, modes)
    return {
        "mu1": mu1,
        "mu1_of_square": mu1_square,
        "mu2": mu2,
        "sample_variance": mu1_square - mu2,
        "circuit_variance": mu2 - mu1 * mu1,
    }
