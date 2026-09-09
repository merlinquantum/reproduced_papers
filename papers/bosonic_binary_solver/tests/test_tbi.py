"""Anchors for the interferometer model.

The two tests that matter scientifically are the convention lock and the
equivalence with ORCA's released simulator: both failures would silently change
every number this package produces rather than raise an error.
"""

import numpy as np
import perceval as pcvl
import pytest
import torch
from lib.tbi import (
    alternating_input,
    beamsplitter_layout,
    candidate_budget,
    mode_count,
    orca_single_loop_distribution,
    perceval_unitary,
    unitary,
)
from perceval.backends import SLOSBackend


def test_beamsplitter_count_matches_appendix_b():
    # Appendix B counts sum_i (m - l_i) beamsplitters; at m=30 with delays
    # (1, 3, 9) that is 77, and 77 is what makes its stated bound come out at
    # 2.15e6 cost evaluations.
    assert len(beamsplitter_layout(30, (1, 3, 9))) == 77
    assert candidate_budget(30, (1, 3, 9), 200, 50) == 2_150_000
    assert candidate_budget(20, (1, 3, 9), 200, 50) == 1_350_000


def test_beamsplitter_convention_is_balanced_at_half_pi():
    # R = cos^2(theta / 2), the Perceval and MerLin convention. ORCA's released
    # toolkit uses cos^2(theta) instead; mixing them rescales every gradient.
    u = unitary(torch.tensor([np.pi / 2], dtype=torch.float64), 2, (1,))
    assert float(u[0, 0] ** 2) == pytest.approx(0.5)
    u0 = unitary(torch.tensor([0.0], dtype=torch.float64), 2, (1,))
    assert float(u0[0, 0] ** 2) == pytest.approx(1.0)


@pytest.mark.parametrize("topology", ["chain", "loop"])
def test_unitary_is_orthogonal_and_batches(topology):
    m, delays = 9, (1, 3)
    k = len(beamsplitter_layout(m, delays, topology))
    n = mode_count(m, delays, topology)
    theta = torch.rand(k, dtype=torch.float64) * 2 * np.pi
    u = unitary(theta, m, delays, topology)
    assert u.shape == (n, n)
    assert torch.allclose(u @ u.T, torch.eye(n, dtype=torch.float64), atol=1e-12)
    stacked = unitary(torch.stack([theta, theta / 2]), m, delays, topology)
    assert torch.allclose(stacked[0], u)


@pytest.mark.parametrize("n_bins", [3, 4, 5])
def test_loop_topology_matches_orca_simulator(n_bins):
    """The loop model reproduces ORCA's released single-loop sampler exactly.

    ORCA's ``tbi_sampler`` walks the pulse train, interfering each bin with the
    photons circulating in the loop. Enumerating that recursion and comparing it
    against the distribution of our rail unitary pins both the topology and the
    beamsplitter orientation.
    """
    thetas = list(np.random.default_rng(n_bins).random(n_bins) * 2 * np.pi)
    input_state = (1,) * n_bins
    reference = orca_single_loop_distribution(input_state, thetas)

    backend = SLOSBackend()
    backend.set_circuit(
        pcvl.Unitary(pcvl.Matrix(perceval_unitary(thetas, n_bins, (1,), "loop")))
    )
    backend.set_input_state(pcvl.BasicState(list(input_state) + [0]))
    ours = {
        tuple(state): float(p)
        for state, p in backend.prob_distribution().items()
        if p > 1e-12
    }

    total_variation = 0.5 * sum(
        abs(reference.get(k, 0.0) - ours.get(k, 0.0))
        for k in set(reference) | set(ours)
    )
    assert total_variation < 1e-12


def test_chain_and_loop_are_different_circuits():
    # They are not two descriptions of one circuit, which is why the paper's
    # parameter count cannot come from the loop model.
    assert len(beamsplitter_layout(30, (1, 3, 9), "chain")) == 77
    assert len(beamsplitter_layout(30, (1, 3, 9), "loop")) == 90
    assert mode_count(30, (1, 3, 9), "loop") == 33


def test_alternating_input_matches_the_paper():
    assert alternating_input(6) == (1, 0, 1, 0, 1, 0)
    assert sum(alternating_input(30)) == 15
    assert alternating_input(4, (1, 3), "loop") == (1, 0, 1, 0, 0, 0)
