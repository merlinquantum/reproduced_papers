"""Anchors for the MerLin photonic variant."""

import numpy as np
import pytest
import torch
from lib.bbs_merlin import MerlinBinarySolver, build_circuit
from lib.problems import build_problem
from lib.tbi import unitary


@pytest.mark.parametrize("m,delays", [(9, (1, 3)), (12, (1, 3, 9)), (10, (1,))])
def test_perceval_circuit_matches_the_torch_unitary(m, delays):
    """The MerLin circuit and lib.tbi must be the same interferometer.

    Two traps live here: Perceval's default beamsplitter is BS.Rx, which is
    complex and would change every interference term, and the permutation
    sandwich needed for the length-3 and length-9 delay lines is easy to get
    wrong in a way that still produces a valid unitary.
    """
    circuit, names = build_circuit(m, delays)
    angles = np.random.default_rng(0).random(len(names)) * 2 * np.pi
    for name, value in zip(names, angles):
        circuit.param(name).set_value(value)
    from_perceval = np.array(circuit.compute_unitary(), dtype=complex)
    from_torch = unitary(torch.tensor(angles), m, delays).numpy()
    assert np.abs(from_perceval - from_torch).max() < 1e-12


def test_click_distribution_is_normalised():
    solver = MerlinBinarySolver(8, delays=(1, 3), updates=1, seed=0)
    probabilities = solver.click_probabilities()
    assert probabilities.shape == (2**8,)
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-10)


def test_exact_gradient_training_reduces_the_objective():
    _, cost_batch, optimum = build_problem("knapsack", 10, 0)
    solver = MerlinBinarySolver(10, updates=60, flip_samples=50, seed=0)
    result = solver.solve(cost_batch)
    assert np.mean(result["history"][-5:]) < np.mean(result["history"][:5])
    # one circuit evaluation per update, against 1 + 2K for the parameter-shift path
    assert result["circuit_evaluations"] == 60
    assert result["candidates_sampled"] == 60 * 50
