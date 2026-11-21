"""Checks of the exact Haar-moment engine against independent computations."""

from __future__ import annotations

from fractions import Fraction
from math import comb, sqrt

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the package on sys.path)
from photonic_trainability.lib import moments
from photonic_trainability.lib.combinatorics import fock_dimension, stirling2
from photonic_trainability.lib.fock import fock_basis, input_occupation
from photonic_trainability.lib.observables import make_monomial


def _diagonal_from_monomial(exponents_by_mode, photons, modes):
    values = {}
    for row in fock_basis(photons, modes):
        value = 1
        for mode, exponent in exponents_by_mode.items():
            value *= int(row[mode]) ** exponent
        values[tuple(int(entry) for entry in row)] = Fraction(value)
    return values


def _explicit_g_values(diagonal, photons, modes):
    """g_l = tr[(L^{n-l} X)^2] with explicit annihilation matrices (no formula)."""
    bases = [fock_basis(level, modes) for level in range(photons + 1)]
    index = [
        {tuple(int(v) for v in row): i for i, row in enumerate(basis)}
        for basis in bases
    ]

    def annihilation(level, mode):
        matrix = np.zeros((len(bases[level - 1]), len(bases[level])))
        for column, row in enumerate(bases[level]):
            if row[mode] == 0:
                continue
            lowered = row.copy()
            lowered[mode] -= 1
            matrix[index[level - 1][tuple(int(v) for v in lowered)], column] = sqrt(
                row[mode]
            )
        return matrix

    operator = np.diag(
        [float(diagonal.get(tuple(int(v) for v in row), 0)) for row in bases[photons]]
    )
    g_values = [0.0] * (photons + 1)
    g_values[photons] = float(np.trace(operator @ operator))
    current = operator
    for level in range(photons, 0, -1):
        current = sum(
            annihilation(level, mode) @ current @ annihilation(level, mode).T
            for mode in range(modes)
        )
        g_values[level - 1] = float(np.trace(current @ current))
    return g_values


def test_stirling_numbers():
    assert [stirling2(4, k) for k in range(5)] == [0, 1, 7, 6, 1]
    assert stirling2(6, 3) == 90


@pytest.mark.parametrize("modes", [2, 3, 5, 8])
def test_irrep_dimensions_sum_to_operator_space_dimension(modes):
    for photons in range(5):
        total = sum(moments.irrep_dimension(k, modes) for k in range(photons + 1))
        assert total == fock_dimension(photons, modes) ** 2


def test_irrep_dimension_matches_mhiri_eq12():
    for modes in (2, 3, 6):
        for level in range(1, 6):
            value = (
                Fraction(2 * level + modes - 1, modes - 1)
                * comb(level + modes - 2, level) ** 2
            )
            assert moments.irrep_dimension(level, modes) == value


@pytest.mark.parametrize(
    "exponents_by_mode, photons, modes",
    [({0: 1}, 3, 6), ({0: 2, 1: 1}, 3, 4), ({0: 1, 1: 1, 2: 1}, 4, 5), ({0: 5}, 2, 3)],
)
def test_first_moment_matches_maximally_mixed_average(
    exponents_by_mode, photons, modes
):
    diagonal = _diagonal_from_monomial(exponents_by_mode, photons, modes)
    brute_force = sum(diagonal.values(), Fraction(0)) / fock_dimension(photons, modes)
    exact = moments.first_moment_monomial(
        make_monomial(exponents_by_mode), photons, modes
    )
    assert exact == brute_force


def test_first_moment_of_number_operator_is_density():
    assert moments.first_moment_exponents((1,), 7, 13) == Fraction(7, 13)


def test_first_moment_approaches_fubini_limit():
    exponents = (2, 3)
    limit = moments.first_moment_limit(exponents, 0.5)
    deviations = [
        abs(float(moments.first_moment_exponents(exponents, modes // 2, modes)) - limit)
        for modes in (100, 400, 1600)
    ]
    assert deviations[0] > deviations[1] > deviations[2]
    assert deviations[2] * 1600 < 50 * limit  # O(1/M) convergence


@pytest.mark.parametrize(
    "exponents_by_mode, photons, modes",
    [({0: 2}, 3, 3), ({0: 1, 1: 2}, 3, 4), ({0: 1, 1: 1, 2: 1}, 3, 3)],
)
def test_g_values_match_explicit_lowering_map(exponents_by_mode, photons, modes):
    diagonal = _diagonal_from_monomial(exponents_by_mode, photons, modes)
    explicit = _explicit_g_values(diagonal, photons, modes)
    structured = moments.g_values_monomial(
        tuple(exponents_by_mode.values()), photons, modes, photons
    )
    brute = moments.g_values_diagonal(diagonal, photons, modes, photons)
    np.testing.assert_allclose([float(v) for v in structured], explicit, rtol=1e-10)
    assert [Fraction(v) for v in structured] == brute


def test_fock_state_g_values_match_bruteforce():
    occupation = (2, 0, 1, 1)
    brute = moments.g_values_diagonal({occupation: 1}, 4, 4, 4)
    assert [Fraction(v) for v in moments.g_values_fock_state(occupation, 4)] == brute


def test_polynomial_g_values_match_bruteforce():
    polynomial = {
        make_monomial({0: 1}): 0.5,
        make_monomial({1: 1, 2: 1}): -1.25,
        make_monomial({2: 2}): 0.75,
    }
    photons, modes = 3, 4
    diagonal = {}
    for row in fock_basis(photons, modes):
        value = Fraction(0)
        for monomial, coefficient in polynomial.items():
            product = 1
            for mode, exponent in monomial:
                product *= int(row[mode]) ** exponent
            value += Fraction(coefficient) * product
        diagonal[tuple(int(v) for v in row)] = value
    assert moments.g_values_polynomial(polynomial, photons, modes, photons) == (
        moments.g_values_diagonal(diagonal, photons, modes, photons)
    )


@pytest.mark.parametrize(
    "exponents_by_mode, photons, modes",
    [({0: 2}, 4, 5), ({0: 1, 1: 1}, 4, 4), ({0: 3}, 5, 4)],
)
def test_irrep_norms_sum_to_trace_and_vanish_beyond_order(
    exponents_by_mode, photons, modes
):
    diagonal = _diagonal_from_monomial(exponents_by_mode, photons, modes)
    g_values = moments.g_values_diagonal(diagonal, photons, modes, photons)
    norms = moments.irrep_norms(g_values, photons, modes)
    assert sum(norms, Fraction(0)) == sum(v * v for v in diagonal.values())
    order = sum(exponents_by_mode.values())
    assert all(norm == 0 for norm in norms[order + 1 :])
    assert all(norm >= 0 for norm in norms)


def test_truncated_second_moment_equals_full_sum():
    occupation = input_occupation(6, "alternating")
    photons, modes = 3, 6
    exponents_by_mode = {0: 1, 1: 1}
    diagonal = _diagonal_from_monomial(exponents_by_mode, photons, modes)
    full = moments.second_moment_from_g(
        moments.g_values_fock_state(occupation, photons),
        moments.g_values_diagonal(diagonal, photons, modes, photons),
        photons,
        modes,
    )
    truncated = moments.monomial_variances((1, 1), occupation)["mu2"]
    assert full == truncated


def test_identity_observable_has_no_circuit_variance():
    occupation = (1, 1, 0, 0, 1)
    record = moments.polynomial_variances({(): 2.0}, occupation)
    assert record["circuit_variance"] == 0
    assert record["sample_variance"] == 0


def test_collision_probability_closed_form():
    # Single photon: P2 = M * sum_j E|U_1j|^4 = 2M / (M + 1).
    for modes in (2, 5, 9):
        assert moments.collision_probability(modes, 1) == Fraction(2 * modes, modes + 1)


@pytest.mark.parametrize("modes, photons", [(3, 2), (4, 2), (5, 3)])
def test_collision_probability_matches_projector_second_moments(modes, photons):
    occupation = tuple([1] * photons + [0] * (modes - photons))
    total = sum(
        moments.projector_variances(tuple(int(v) for v in row), occupation)["mu2"]
        for row in fock_basis(photons, modes)
    )
    assert fock_dimension(photons, modes) * total == moments.collision_probability(
        modes, photons
    )


def test_fig4_anchor_value():
    # Paper Fig. 4, alpha = 2 panel: first point at M ~ 10 reads ~0.128.
    record = moments.monomial_variances((2,), input_occupation(10, "alternating"))
    assert abs(float(record["circuit_variance"]) - 0.128) < 0.002


def test_local_slopes_distinguish_power_law_from_exponential():
    from photonic_trainability.lib.fits import local_loglog_slopes

    modes = [100, 200, 400, 800]
    power = [m**-1.0 for m in modes]
    exponential = [np.exp(-m / 100) for m in modes]
    assert all(abs(s["slope"] + 1.0) < 1e-12 for s in local_loglog_slopes(modes, power))
    slopes = [s["slope"] for s in local_loglog_slopes(modes, exponential)]
    assert slopes[0] > slopes[1] > slopes[2]
