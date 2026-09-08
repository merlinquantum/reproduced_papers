"""Anchors for the cost functions and their exact optima."""

import itertools

import numpy as np

from lib.problems import (
    build_problem,
    knapsack_cost,
    knapsack_cost_batch,
    knapsack_instance,
    knapsack_optimum,
    locations_for_modes,
    tsp_cost,
    tsp_cost_batch,
    tsp_instance,
    tsp_optimum,
)

ALL_BITS_10 = np.array(list(itertools.product([0, 1], repeat=10)))


def test_knapsack_dynamic_programme_matches_brute_force():
    for seed in range(4):
        instance = knapsack_instance(10, seed)
        brute = min(knapsack_cost(instance, bits) for bits in ALL_BITS_10)
        assert knapsack_optimum(instance) == brute


def test_knapsack_batch_matches_scalar():
    instance = knapsack_instance(10, 7)
    expected = np.array([knapsack_cost(instance, bits) for bits in ALL_BITS_10])
    assert np.allclose(knapsack_cost_batch(instance, ALL_BITS_10), expected)


def test_tsp_sizes_invert_to_the_paper_locations():
    # The paper's TSP sizes are numbers of binary variables, m = ceil(log2((n-1)!)).
    assert (locations_for_modes(10), locations_for_modes(19), locations_for_modes(29)) == (7, 10, 13)


def test_tsp_optimum_is_reachable_and_batch_matches_scalar():
    instance = tsp_instance(10, 2)
    expected = np.array([tsp_cost(instance, bits) for bits in ALL_BITS_10])
    assert np.allclose(tsp_cost_batch(instance, ALL_BITS_10), expected)
    # Held-Karp optimum must be attainable through the binary encoding, otherwise
    # "% optimal" could never reach 100 and the metric would be meaningless.
    assert np.isclose(expected.min(), tsp_optimum(instance))


def test_build_problem_contract():
    for family in ("knapsack", "tsp"):
        instance, cost_batch, optimum = build_problem(family, 10, 0)
        costs = cost_batch(ALL_BITS_10)
        assert costs.shape == (len(ALL_BITS_10),)
        assert np.isclose(costs.min(), optimum)
