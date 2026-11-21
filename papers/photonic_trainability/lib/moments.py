"""Exact Haar moments of photon-number observables in passive linear optics.

Notation follows the paper (arXiv:2607.21544) and Mhiri et al.
(arXiv:2604.14323), "[51]":

* ``mu_k(O) = E_U[ tr(O U rho U^dagger)^k ]`` over the Haar measure on U(M),
  with ``rho`` a Fock state of ``n`` photons in ``M`` modes.
* ``Var_samp(O) = mu1(O^2) - mu2(O)`` and ``Var_circ(O) = mu2(O) - mu1(O)^2``
  (paper Eqs. 7-8).
* First moment: the Haar-averaged output is the maximally mixed n-photon state
  (paper Theorem 3), which gives the closed form of paper Theorem 1 / Eq. 89.
* Second moment: ``mu2 = sum_k ||P_k(rho)||^2 ||P_k(O)||^2 / d_k`` (paper Eq. 12,
  [51] Eq. 33), with the irrep norms of [51] Corollary 1 (Eqs. 37-38) built from
  ``g_l(X) = tr[(L^{n-l} X)^2]`` and the lowering map ``L(X) = sum_j a_j X a_j^+``.

All quantities are exact integers or :class:`fractions.Fraction`. Floats appear
only when callers convert final results.

A PNM of order ``|alpha|`` is a combination of normally ordered p-body
operators with ``p <= |alpha|``, so its projections onto irreps ``k > |alpha|``
vanish. The second moment therefore only needs ``k <= min(n, |alpha|)``, which
keeps the paper-scale curves (n up to ~170) tractable. ``tests/test_moments.py``
checks this truncation against the full sum on small systems.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import cache
from math import comb, factorial

from .combinatorics import compositions, fock_dimension, rising_factorial, stirling2
from .observables import (
    Monomial,
    monomial_exponents,
    multiply_monomials,
    polynomial_order,
)

# ---------------------------------------------------------------------------
# First moment (paper Theorem 1)
# ---------------------------------------------------------------------------


def _ordered_bell_sequence(exponent: int) -> list[int]:
    """Sequence ``F^a = (j! S(a, j))_{j=0..a}`` of paper Eq. 88."""
    return [factorial(j) * stirling2(exponent, j) for j in range(exponent + 1)]


def _convolve(left: Sequence[int], right: Sequence[int]) -> list[int]:
    """Full discrete convolution of two integer sequences."""
    result = [0] * (len(left) + len(right) - 1)
    for i, left_value in enumerate(left):
        if left_value == 0:
            continue
        for j, right_value in enumerate(right):
            result[i + j] += left_value * right_value
    return result


@cache
def _support_convolution(exponents: tuple[int, ...]) -> tuple[int, ...]:
    """``F^{a_1} * ... * F^{a_K}`` (paper Eq. 87) for a sorted exponent tuple."""
    sequence = [1]
    for exponent in exponents:
        sequence = _convolve(sequence, _ordered_bell_sequence(exponent))
    return tuple(sequence)


def first_moment_exponents(
    exponents: Sequence[int], photons: int, modes: int
) -> Fraction:
    """Haar first moment ``mu1(alpha; n)`` of a PNM (paper Eq. 9 / Eq. 89).

    Parameters
    ----------
    exponents : sequence of int
        Positive exponents ``(alpha_1, ..., alpha_K)`` on ``K`` distinct modes.
        The result does not depend on which modes they sit on.
    photons : int
        Total photon number ``n`` of the input.
    modes : int
        Number of modes ``M``.

    Returns
    -------
    fractions.Fraction
        Exact first moment; ``1`` for the empty monomial.

    Raises
    ------
    ValueError
        If an exponent is not positive or the support exceeds ``M``.
    """
    exponents = tuple(sorted(int(value) for value in exponents))
    if any(value <= 0 for value in exponents):
        raise ValueError(f"Exponents must be positive, got {exponents}")
    if len(exponents) > modes:
        raise ValueError(f"Support {len(exponents)} exceeds {modes} modes")
    if not exponents:
        return Fraction(1)
    convolution = _support_convolution(exponents)
    denominator = comb(photons + modes - 1, photons)
    total = Fraction(0)
    for removed in range(len(exponents), sum(exponents) + 1):
        if removed > photons:
            break
        total += Fraction(
            comb(photons + modes - 1, photons - removed) * convolution[removed],
            denominator,
        )
    return total


def first_moment_monomial(monomial: Monomial, photons: int, modes: int) -> Fraction:
    """Haar first moment of a PNM key (see :func:`first_moment_exponents`)."""
    if not monomial:
        return Fraction(1)
    return first_moment_exponents(monomial_exponents(monomial), photons, modes)


def first_moment_limit(exponents: Sequence[int], density: float) -> float:
    """Large-M limit of ``mu1`` at fixed density ``nu = n / M`` (paper Eq. 10/95).

    The limit is a product of ordered Bell (Fubini) polynomials
    ``F_a(nu) = sum_p p! S(a, p) nu^p``.

    Parameters
    ----------
    exponents : sequence of int
        Positive exponents of the PNM.
    density : float
        Photon density ``nu``.

    Returns
    -------
    float
        ``prod_j F_{alpha_j}(nu)``.
    """
    value = 1.0
    for exponent in exponents:
        value *= sum(
            coefficient * density**power
            for power, coefficient in enumerate(_ordered_bell_sequence(int(exponent)))
        )
    return value


def first_moment_polynomial(
    polynomial: Mapping[Monomial, float | Fraction], photons: int, modes: int
) -> Fraction:
    """Haar first moment of a PNP: ``sum_alpha c_alpha mu1(alpha)`` (paper Eq. 79)."""
    return sum(
        (
            Fraction(coefficient) * first_moment_monomial(monomial, photons, modes)
            for monomial, coefficient in polynomial.items()
        ),
        Fraction(0),
    )


def first_moment_of_square(
    polynomial: Mapping[Monomial, float | Fraction], photons: int, modes: int
) -> Fraction:
    """Haar first moment of ``C^2``: ``sum_{alpha,beta} c_alpha c_beta mu1(alpha+beta)``.

    This is the term that makes the sample variance of an unbounded photonic
    observable large (paper Eq. 17 and App. D.3).
    """
    items = [(monomial, Fraction(c)) for monomial, c in polynomial.items()]
    total = Fraction(0)
    for i, (left, left_coefficient) in enumerate(items):
        for right, right_coefficient in items[i:]:
            weight = 1 if left == right else 2
            total += (
                weight
                * left_coefficient
                * right_coefficient
                * first_moment_monomial(multiply_monomials(left, right), photons, modes)
            )
    return total


# ---------------------------------------------------------------------------
# Irrep decomposition (Mhiri et al. Eqs. 12, 37, 38)
# ---------------------------------------------------------------------------


def irrep_dimension(level: int, modes: int) -> int:
    """Dimension ``d_k`` of irrep ``lambda_k`` of U(M) in the n-photon operator space.

    [51] Eq. 12 gives ``d_k = (2k + M - 1) / (M - 1) * C(k + M - 2, k)^2``; the
    equivalent integer form ``C(k + M - 1, k)^2 - C(k + M - 2, k - 1)^2`` is used.

    Parameters
    ----------
    level : int
        Irrep label ``k >= 0``.
    modes : int
        Number of modes ``M >= 2``.

    Returns
    -------
    int
        The irrep dimension (independent of ``n``).
    """
    if modes < 2:
        raise ValueError("Irrep dimensions need at least two modes")
    if level == 0:
        return 1
    return comb(level + modes - 1, level) ** 2 - comb(level + modes - 2, level - 1) ** 2


def _irrep_normalisation(level: int, photons: int, modes: int) -> int:
    """``alpha_{k,k+1,n} = (n - k)! (M + 2k)_{n-k}`` ([51] Eq. 38)."""
    return factorial(photons - level) * rising_factorial(
        modes + 2 * level, photons - level
    )


def irrep_norms(
    g_values: Sequence[int | Fraction], photons: int, modes: int
) -> list[Fraction]:
    """Squared Hilbert-Schmidt norms ``||P_k(X)||^2`` for ``k = 0..len(g_values) - 1``.

    [51] Corollary 1 (Eq. 37):
    ``||P_k||^2 = (1 / alpha_{k,k+1,n}) sum_{l=0}^{k} (-1)^{k-l} g_l /
    ((k - l)! (k + l + M - 1)_{k-l})``.

    Parameters
    ----------
    g_values : sequence of int or Fraction
        ``g_l(X)`` for ``l = 0, 1, ..., k_max``.
    photons : int
        Photon number ``n`` of the sector.
    modes : int
        Number of modes ``M``.

    Returns
    -------
    list of fractions.Fraction
        One norm per irrep label.
    """
    norms = []
    for level in range(len(g_values)):
        if level > photons:
            raise ValueError(f"Irrep label {level} exceeds photon number {photons}")
        total = Fraction(0)
        for lower in range(level + 1):
            sign = -1 if (level - lower) % 2 else 1
            total += Fraction(
                sign * g_values[lower],
                factorial(level - lower)
                * rising_factorial(level + lower + modes - 1, level - lower),
            )
        norms.append(total / _irrep_normalisation(level, photons, modes))
    return norms


# ---------------------------------------------------------------------------
# g_l values: g_l(X) = tr[(L^{n-l} X)^2]
#
# For a diagonal operator X = sum_m x(m)|m><m| on the n-photon sector,
# L^{n-l}(|m><m|) = (n-l)! sum_{a in Phi^l, a <= m} prod_i C(m_i, a_i) |a><a|,
# so g_l = (n-l)!^2 sum_{a in Phi^l} w(a)^2 with w(a) = sum_m x(m) prod_i C(m_i, a_i).
# ---------------------------------------------------------------------------


def g_values_fock_state(occupation: Sequence[int], max_level: int) -> list[int]:
    """``g_l(|s><s|)`` for ``l = 0..max_level`` ([51] Lemma 4; paper Eq. 121).

    ``g_l = (n - l)!^2 [x^l] prod_i sum_a C(s_i, a)^2 x^a``.

    Parameters
    ----------
    occupation : sequence of int
        Photon count per mode of the Fock state ``|s>``.
    max_level : int
        Largest ``l`` required.

    Returns
    -------
    list of int
        ``g_0, ..., g_max_level``.
    """
    photons = sum(occupation)
    if max_level > photons:
        raise ValueError(f"max_level {max_level} exceeds photon number {photons}")
    sequence = [1]
    for count in occupation:
        sequence = _convolve(sequence, [comb(count, a) ** 2 for a in range(count + 1)])
    return [
        factorial(photons - level) ** 2 * sequence[level]
        for level in range(max_level + 1)
    ]


@cache
def _monomial_lowering_weight(
    exponents: tuple[int, ...],
    lowered_support: tuple[int, ...],
    lowered_rest: int,
    photons: int,
    modes: int,
) -> int:
    """``w(a)`` for a PNM, given ``a`` on the support and its total off the support.

    ``w(a) = sum_{m >= a} prod_{i in S} m_i^{alpha_i} C(m_i, a_i) prod_{i not in S}
    C(m_i, a_i)``. The off-support factor sums in closed form,
    ``sum_{m_R in Phi^N_r} prod C(m_i, a_i) = C(N + r - 1, A_R + r - 1)`` with
    ``r = M - K`` and ``A_R = sum_{i not in S} a_i``, so ``w`` depends on ``a`` only
    through ``(a_S, A_R)``.
    """
    rest_modes = modes - len(exponents)
    support_sum = [1]
    for exponent, lowered in zip(exponents, lowered_support):
        per_mode = [
            count**exponent * comb(count, lowered) for count in range(photons + 1)
        ]
        support_sum = _convolve(support_sum, per_mode)[: photons + 1]
    total = 0
    for support_photons, value in enumerate(support_sum):
        if value == 0:
            continue
        rest_photons = photons - support_photons
        if rest_modes == 0:
            rest_weight = 1 if rest_photons == 0 and lowered_rest == 0 else 0
        elif rest_photons < lowered_rest:
            rest_weight = 0
        else:
            rest_weight = comb(
                rest_photons + rest_modes - 1, lowered_rest + rest_modes - 1
            )
        total += value * rest_weight
    return total


def _rest_multiplicity(lowered_rest: int, rest_modes: int) -> int:
    """Number of ways to spread ``lowered_rest`` photons over ``rest_modes`` modes."""
    if rest_modes == 0:
        return 1 if lowered_rest == 0 else 0
    return comb(lowered_rest + rest_modes - 1, rest_modes - 1)


def g_values_monomial(
    exponents: Sequence[int], photons: int, modes: int, max_level: int
) -> list[int]:
    """``g_l(M_alpha)`` for ``l = 0..max_level`` on the n-photon sector.

    The sum over ``a in Phi^l_M`` is grouped by the support part ``a_S`` and the
    off-support total ``A_R``; see :func:`_monomial_lowering_weight`.

    Parameters
    ----------
    exponents : sequence of int
        Positive PNM exponents on ``K`` distinct modes.
    photons : int
        Photon number ``n``.
    modes : int
        Number of modes ``M``.
    max_level : int
        Largest ``l`` required (at most ``n``).

    Returns
    -------
    list of int
        ``g_0, ..., g_max_level``.
    """
    exponents = tuple(sorted(int(value) for value in exponents))
    if max_level > photons:
        raise ValueError(f"max_level {max_level} exceeds photon number {photons}")
    rest_modes = modes - len(exponents)
    if rest_modes < 0:
        raise ValueError(f"Support {len(exponents)} exceeds {modes} modes")
    values = []
    for level in range(max_level + 1):
        total = 0
        for support_level in range(level + 1):
            lowered_rest = level - support_level
            multiplicity = _rest_multiplicity(lowered_rest, rest_modes)
            if multiplicity == 0:
                continue
            for lowered_support in compositions(support_level, len(exponents)):
                weight = _monomial_lowering_weight(
                    exponents, lowered_support, lowered_rest, photons, modes
                )
                total += multiplicity * weight * weight
        values.append(factorial(photons - level) ** 2 * total)
    return values


def g_values_polynomial(
    polynomial: Mapping[Monomial, float | Fraction],
    photons: int,
    modes: int,
    max_level: int,
) -> list[Fraction]:
    """``g_l(C)`` for a PNP, by explicit enumeration of ``a in Phi^l_M``.

    ``w_C(a) = sum_alpha c_alpha w_alpha(a)``. Cost grows as
    ``C(l + M - 1, l) * |A|``, which is fine for the orders used here
    (``l <= 4``) at moderate ``M``.

    Parameters
    ----------
    polynomial : Mapping[Monomial, float | Fraction]
        PNP coefficients; floats are converted exactly to fractions.
    photons : int
        Photon number ``n``.
    modes : int
        Number of modes ``M``.
    max_level : int
        Largest ``l`` required.

    Returns
    -------
    list of fractions.Fraction
        ``g_0, ..., g_max_level``.
    """
    terms = [
        (
            monomial_exponents(monomial),
            tuple(mode for mode, _ in monomial),
            Fraction(coefficient),
        )
        for monomial, coefficient in polynomial.items()
    ]
    values = []
    for level in range(max_level + 1):
        total = Fraction(0)
        for lowered in compositions(level, modes):
            weight = Fraction(0)
            for exponents, support_modes, coefficient in terms:
                lowered_support = tuple(lowered[mode] for mode in support_modes)
                lowered_rest = level - sum(lowered_support)
                # Exponents and lowered counts stay paired in mode order.
                weight += coefficient * _monomial_lowering_weight(
                    exponents,
                    lowered_support,
                    lowered_rest,
                    photons,
                    modes,
                )
            total += weight * weight
        values.append(factorial(photons - level) ** 2 * total)
    return values


def g_values_diagonal(
    values_by_pattern: Mapping[tuple[int, ...], int | Fraction],
    photons: int,
    modes: int,
    max_level: int,
) -> list[Fraction]:
    """``g_l`` of an arbitrary diagonal operator, by brute-force enumeration.

    Used for projector observables ``|s><s|`` and as an independent check of the
    structured formulas.

    Parameters
    ----------
    values_by_pattern : Mapping[tuple of int, int | Fraction]
        Diagonal entries ``x(m)`` keyed by n-photon pattern; missing patterns are 0.
    photons, modes : int
        Sector ``(n, M)``.
    max_level : int
        Largest ``l`` required.

    Returns
    -------
    list of fractions.Fraction
        ``g_0, ..., g_max_level``.
    """
    values = []
    for level in range(max_level + 1):
        total = Fraction(0)
        for lowered in compositions(level, modes):
            weight = Fraction(0)
            for pattern, entry in values_by_pattern.items():
                product = 1
                for count, lowered_count in zip(pattern, lowered):
                    product *= comb(count, lowered_count)
                    if product == 0:
                        break
                weight += Fraction(entry) * product
            total += weight * weight
        values.append(factorial(photons - level) ** 2 * total)
    return values


# ---------------------------------------------------------------------------
# Second moment and variances
# ---------------------------------------------------------------------------


def second_moment_from_g(
    input_g: Sequence[int | Fraction],
    observable_g: Sequence[int | Fraction],
    photons: int,
    modes: int,
) -> Fraction:
    """``mu2 = sum_k ||P_k(rho)||^2 ||P_k(O)||^2 / d_k`` (paper Eq. 12).

    The sum runs over ``k = 0..len(observable_g) - 1``; callers truncate at the
    observable's order, beyond which ``P_k(O) = 0``.
    """
    levels = len(observable_g)
    if len(input_g) < levels:
        raise ValueError("Input g-values must cover every observable level")
    input_norms = irrep_norms(input_g[:levels], photons, modes)
    observable_norms = irrep_norms(observable_g, photons, modes)
    return sum(
        (
            input_norm * observable_norm / irrep_dimension(level, modes)
            for level, (input_norm, observable_norm) in enumerate(
                zip(input_norms, observable_norms)
            )
        ),
        Fraction(0),
    )


def monomial_variances(
    exponents: Sequence[int], occupation: Sequence[int]
) -> dict[str, Fraction]:
    """First/second moments and both variances of a PNM for a Fock input.

    Parameters
    ----------
    exponents : sequence of int
        Positive PNM exponents.
    occupation : sequence of int
        Fock input ``|s>``; its length is ``M``.

    Returns
    -------
    dict
        ``mu1``, ``mu1_of_square``, ``mu2``, ``sample_variance``,
        ``circuit_variance`` as exact fractions.
    """
    photons, modes = sum(occupation), len(occupation)
    exponents = tuple(sorted(int(value) for value in exponents))
    top_level = min(photons, sum(exponents))
    mu1 = first_moment_exponents(exponents, photons, modes)
    mu1_square = first_moment_exponents(
        tuple(2 * value for value in exponents), photons, modes
    )
    mu2 = second_moment_from_g(
        g_values_fock_state(occupation, top_level),
        g_values_monomial(exponents, photons, modes, top_level),
        photons,
        modes,
    )
    return _variance_record(mu1, mu1_square, mu2)


def polynomial_variances(
    polynomial: Mapping[Monomial, float | Fraction], occupation: Sequence[int]
) -> dict[str, Fraction]:
    """Same as :func:`monomial_variances` for a photon-number polynomial."""
    photons, modes = sum(occupation), len(occupation)
    top_level = min(photons, polynomial_order(polynomial))
    mu1 = first_moment_polynomial(polynomial, photons, modes)
    mu1_square = first_moment_of_square(polynomial, photons, modes)
    mu2 = second_moment_from_g(
        g_values_fock_state(occupation, top_level),
        g_values_polynomial(polynomial, photons, modes, top_level),
        photons,
        modes,
    )
    return _variance_record(mu1, mu1_square, mu2)


def projector_variances(
    output_pattern: Sequence[int], occupation: Sequence[int]
) -> dict[str, Fraction]:
    """Moments and variances of the projector ``|s><s|`` onto one output pattern.

    The projector has components in every irrep, so the full sum ``k = 0..n`` is
    used. Exponential in system size; small systems only (paper App. G.1).
    """
    photons, modes = sum(occupation), len(occupation)
    if sum(output_pattern) != photons or len(output_pattern) != modes:
        raise ValueError("Output pattern must lie in the input's photon sector")
    dimension = fock_dimension(photons, modes)
    mu1 = Fraction(1, dimension)
    mu2 = second_moment_from_g(
        g_values_fock_state(occupation, photons),
        g_values_fock_state(output_pattern, photons),
        photons,
        modes,
    )
    return _variance_record(mu1, mu1, mu2)


def _variance_record(
    mu1: Fraction, mu1_square: Fraction, mu2: Fraction
) -> dict[str, Fraction]:
    return {
        "mu1": mu1,
        "mu1_of_square": mu1_square,
        "mu2": mu2,
        "sample_variance": mu1_square - mu2,
        "circuit_variance": mu2 - mu1 * mu1,
    }


def collision_probability(modes: int, photons: int) -> Fraction:
    """Normalised average outcome-collision probability ``P2(M, n)``.

    [51] Theorem 1, Eq. 40:
    ``P2 = sum_{k=0}^{n} (M + k - 1)_k / (M + n)_k * 2F1(-k, n - k + 1; 2 - M - 2k; -1)``.
    """
    total = Fraction(0)
    for level in range(photons + 1):
        hypergeometric = Fraction(0)
        upper = photons - level + 1
        lower = 2 - modes - 2 * level
        # 2F1(-k, a; b; z) = sum_p (-1)^p C(k, p) (a)_p / (b)_p z^p; at z = -1 the
        # two signs cancel.
        for power in range(level + 1):
            hypergeometric += Fraction(
                comb(level, power) * rising_factorial(upper, power),
                rising_factorial(lower, power),
            )
        total += (
            Fraction(
                rising_factorial(modes + level - 1, level),
                rising_factorial(modes + photons, level),
            )
            * hypergeometric
        )
    return total
