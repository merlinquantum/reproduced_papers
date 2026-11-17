from __future__ import annotations

import math

import common  # noqa: F401  (puts the repository and paper roots on sys.path)
import numpy as np
import perceval as pcvl
import pytest
from lib.fock import (
    conditional_state,
    fock_basis,
    permanent,
    transition_amplitude,
)
from scipy.stats import unitary_group


def test_permanent_small_matrices():
    assert permanent(np.zeros((0, 0))) == 1
    assert permanent(np.array([[2.0]])) == pytest.approx(2.0)
    assert permanent(np.array([[1.0, 2.0], [3.0, 4.0]])) == pytest.approx(10.0)
    assert permanent(np.ones((3, 3))) == pytest.approx(6.0)


@pytest.mark.parametrize(
    "output", [(2, 0, 1, 0, 0), (0, 1, 1, 0, 1), (0, 0, 0, 0, 3), (1, 0, 1, 1, 0)]
)
def test_transition_amplitude_matches_perceval_slos(output):
    unitary = unitary_group.rvs(5, random_state=1)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    backend.set_circuit(pcvl.Unitary(pcvl.Matrix(unitary)))
    backend.set_input_state(pcvl.BasicState([1, 0, 1, 1, 0]))
    expected = backend.prob_amplitude(pcvl.BasicState(list(output)))
    assert transition_amplitude(unitary, (1, 0, 1, 1, 0), output) == pytest.approx(
        expected, abs=1e-12
    )


def test_hong_ou_mandel_suppresses_coincidence():
    splitter = np.array([[1, 1j], [1j, 1]]) / math.sqrt(2)
    assert abs(transition_amplitude(splitter, (1, 1), (1, 1))) < 1e-12
    assert abs(transition_amplitude(splitter, (1, 1), (2, 0))) ** 2 == pytest.approx(
        0.5
    )


def test_fock_basis_dimension_and_order():
    assert fock_basis(3, 1) == [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    assert len(fock_basis(6, 2)) == math.comb(7, 2)


def test_conditional_state_probabilities_sum_to_one():
    unitary = unitary_group.rvs(4, random_state=3)
    detected, residual = (0, 1), (2, 3)
    total = 0.0
    for counts in [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]:
        try:
            state, probability = conditional_state(
                unitary, (0, 2), detected, counts, residual
            )
        except ValueError:
            continue
        assert np.linalg.norm(state) == pytest.approx(1.0)
        total += probability
    assert total == pytest.approx(1.0)


def test_conditional_state_rejects_bad_partition():
    with pytest.raises(ValueError, match="partition"):
        conditional_state(np.eye(3), (0,), (0,), (1,), (1,))
