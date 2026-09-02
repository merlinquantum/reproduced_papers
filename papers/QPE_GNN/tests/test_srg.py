"""Numerical tests for the paper-scale SRG experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

from lib.srg import (  # noqa: E402
    load_srg_catalogs,
    rrwp_correlation,
    sorted_correlation_distance,
    two_particle_walk_correlations,
)


def test_complete_paper_srg_catalogs_are_loaded():
    families = load_srg_catalogs(PAPER_ROOT / "data")
    assert len(families["srg(25,12,5,6)"]) == 15
    assert len(families["srg(26,10,3,4)"]) == 10


def test_two_particle_correlations_distinguish_and_rrwp_does_not():
    families = load_srg_catalogs(PAPER_ROOT / "data")
    for adjacency_matrices in families.values():
        first_adjacency, second_adjacency = adjacency_matrices[:2]
        quantum_distance = sorted_correlation_distance(
            two_particle_walk_correlations(first_adjacency, 1.0),
            two_particle_walk_correlations(second_adjacency, 1.0),
        )
        rrwp_distance = sorted_correlation_distance(
            rrwp_correlation(first_adjacency, 20),
            rrwp_correlation(second_adjacency, 20),
        )
        assert quantum_distance > 1e-9
        assert rrwp_distance < 1e-9


def test_sorted_correlations_are_invariant_to_node_permutations():
    adjacency = load_srg_catalogs(PAPER_ROOT / "data")["srg(26,10,3,4)"][0]
    permutation = np.random.default_rng(0).permutation(adjacency.shape[0])
    permuted_adjacency = adjacency[np.ix_(permutation, permutation)]
    distance = sorted_correlation_distance(
        two_particle_walk_correlations(adjacency, 1.0),
        two_particle_walk_correlations(permuted_adjacency, 1.0),
    )
    assert distance < 1e-9
