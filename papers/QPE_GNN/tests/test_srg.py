"""Numerical tests for the paper-scale SRG experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PAPER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.srg import (  # noqa: E402
    load_srg_catalogs,
    rrwp_correlation,
    sorted_correlation_distance,
    two_particle_walk_correlations,
)

from runtime_lib.data_paths import paper_data_dir  # noqa: E402

# The catalogs live under the shared data root (data/QPE_GNN/srg/), which is
# not committed; README "Data" gives the two-line download command.
SRG_DIRECTORY = paper_data_dir("QPE_GNN", ensure_exists=False) / "srg"
pytestmark = pytest.mark.skipif(
    not (SRG_DIRECTORY / "sr251256.g6").is_file(),
    reason=f"SRG catalogs not downloaded to {SRG_DIRECTORY} (see README, Data)",
)


def test_complete_paper_srg_catalogs_are_loaded():
    families = load_srg_catalogs(SRG_DIRECTORY)
    assert len(families["srg(25,12,5,6)"]) == 15
    assert len(families["srg(26,10,3,4)"]) == 10


def test_two_particle_correlations_distinguish_and_rrwp_does_not():
    families = load_srg_catalogs(SRG_DIRECTORY)
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
    adjacency = load_srg_catalogs(SRG_DIRECTORY)["srg(26,10,3,4)"][0]
    permutation = np.random.default_rng(0).permutation(adjacency.shape[0])
    permuted_adjacency = adjacency[np.ix_(permutation, permutation)]
    distance = sorted_correlation_distance(
        two_particle_walk_correlations(adjacency, 1.0),
        two_particle_walk_correlations(permuted_adjacency, 1.0),
    )
    assert distance < 1e-9
