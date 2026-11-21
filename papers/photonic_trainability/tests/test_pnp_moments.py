from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the package on sys.path)
from photonic_trainability.lib import moments, pnp_moments
from photonic_trainability.lib.fock import input_occupation
from photonic_trainability.lib.observables import all_monomials, random_polynomial


def _random_pnp(modes, min_order, max_order, term_count, seed):
    rng = np.random.default_rng(seed)
    return random_polynomial(
        all_monomials(modes, min_order, max_order), term_count, -1.0, 1.0, rng
    )


@pytest.mark.parametrize("modes, pattern", [(4, "alternating"), (5, "all_ones")])
def test_sparse_path_matches_enumeration(modes, pattern):
    polynomial = _random_pnp(modes, 1, 2, None, seed=modes)
    occupation = input_occupation(modes, pattern)
    photons = sum(occupation)
    level = min(2, photons)
    assert pnp_moments.g_values_sparse(polynomial, photons, modes, level) == (
        moments.g_values_polynomial(polynomial, photons, modes, level)
    )


@pytest.mark.parametrize("modes, max_order, terms", [(4, 3, 8), (5, 4, 10), (6, 2, 12)])
def test_pair_path_matches_enumeration(modes, max_order, terms):
    polynomial = _random_pnp(modes, 1, max_order, terms, seed=10 + modes)
    occupation = input_occupation(modes, "all_ones")
    photons = sum(occupation)
    level = min(max_order, photons)
    assert pnp_moments.g_values_pairs(polynomial, photons, modes, level) == (
        moments.g_values_polynomial(polynomial, photons, modes, level)
    )


@pytest.mark.parametrize("modes, max_order, terms", [(4, 2, None), (5, 4, 9)])
def test_first_moment_of_square_matches_naive_pairs(modes, max_order, terms):
    polynomial = _random_pnp(modes, 1, max_order, terms, seed=20 + modes)
    photons = modes // 2 + 1
    assert pnp_moments.first_moment_of_square(polynomial, photons, modes) == (
        moments.first_moment_of_square(polynomial, photons, modes)
    )


def test_fast_variances_match_reference():
    polynomial = _random_pnp(6, 1, 2, None, seed=3)
    occupation = input_occupation(6, "alternating")
    fast = pnp_moments.polynomial_variances(polynomial, occupation)
    reference = moments.polynomial_variances(polynomial, occupation)
    assert fast == reference


def test_partition_schemes_draw_distinct_monomials_with_valid_orders():
    from photonic_trainability.lib.observables import (
        integer_partitions,
        sample_distinct_monomials,
    )

    assert sorted(len(integer_partitions(order)) for order in range(1, 5)) == [
        1,
        2,
        3,
        5,
    ]
    rng = np.random.default_rng(0)
    for scheme in ("uniform_pattern", "order_then_partition"):
        monomials = sample_distinct_monomials(12, 1, 4, 30, scheme, rng)
        assert len(set(monomials)) == 30
        assert all(1 <= sum(e for _, e in m) <= 4 for m in monomials)
