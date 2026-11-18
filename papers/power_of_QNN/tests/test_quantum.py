from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from power_of_QNN.lib.quantum import QubitQNN, StatevectorBatch, entangled_pairs


def test_entangled_pairs_follow_qiskit_order():
    assert entangled_pairs(4, "full") == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    assert entangled_pairs(4, "linear") == [(0, 1), (1, 2), (2, 3)]


def test_cx_flips_target_only_when_control_is_one():
    # Qubit q is tensor axis q, so basis index = sum_q bit_q * 2^(n-1-q).
    for control, target in [(0, 1), (1, 0), (0, 2), (2, 0)]:
        state = StatevectorBatch(1, 3, "cpu", torch.float64)
        state.amplitudes = torch.zeros(1, 8, dtype=torch.complex128)
        bits = [0, 0, 0]
        bits[control] = 1
        index = sum(bit << (2 - qubit) for qubit, bit in enumerate(bits))
        state.amplitudes[0, index] = 1
        state.cx(control, target)
        bits[target] = 1
        expected = sum(bit << (2 - qubit) for qubit, bit in enumerate(bits))
        assert state.amplitudes[0, expected].real.item() == pytest.approx(1.0)


@pytest.mark.parametrize("feature_map,reps", [("zz", 2), ("z", 1), ("ry_angle", 1)])
def test_probabilities_are_normalized(feature_map, reps):
    model = QubitQNN(4, feature_map, reps, 2)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(5, 4, dtype=torch.float64, generator=generator)
    theta = torch.rand(5, model.d, dtype=torch.float64, generator=generator)
    probabilities = model.probabilities(x, theta)
    assert probabilities.shape == (5, 2)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(5, dtype=torch.float64))


def test_parameter_count_matches_paper_formula():
    assert QubitQNN(4, "zz", 2, 9).d == 40
    assert QubitQNN(10, "zz", 2, 9).d == 100
    assert QubitQNN(4, "zz", 2, 1).d == 8


def test_autograd_equals_parameter_shift():
    # The authors differentiate with the +-pi/2 parameter shift, exact for RY gates.
    model = QubitQNN(3, "zz", 2, 2)
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(1, 3, dtype=torch.float64, generator=generator)
    theta = torch.rand(
        1, model.d, dtype=torch.float64, generator=generator
    ).requires_grad_(True)
    (autograd,) = torch.autograd.grad(model.probabilities(x, theta)[0, 0], theta)
    shift = torch.zeros_like(theta)
    for index in range(model.d):
        shift.zero_()
        shift[0, index] = math.pi / 2
        with torch.no_grad():
            difference = (
                model.probabilities(x, theta + shift)[0, 0]
                - model.probabilities(x, theta - shift)[0, 0]
            )
        assert autograd[0, index].item() == pytest.approx(
            0.5 * difference.item(), abs=1e-12
        )


def test_rows_use_their_own_parameters():
    model = QubitQNN(2, "zz", 2, 1)
    x = torch.tensor([[0.3, -0.2], [0.3, -0.2]], dtype=torch.float64)
    theta = torch.tensor(
        np.array([[0.1, 0.2, 0.3, 0.4], [1.0, 0.5, -0.5, 2.0]]), dtype=torch.float64
    )
    batched = model.probabilities(x, theta)
    single = torch.cat(
        [model.probabilities(x[i : i + 1], theta[i : i + 1]) for i in range(2)]
    )
    assert torch.allclose(batched, single)
