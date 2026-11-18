from __future__ import annotations

import math

import torch
from lib import statevector as sv
from lib.models import VariationalAnsatz, angle_encode


def test_ry_encoding_gives_cosine_expectation():
    angles = torch.tensor([[0.3, 1.2], [math.pi, 0.0]], dtype=torch.float64)
    expectations = sv.expectation_z(angle_encode(angles))
    assert torch.allclose(expectations, torch.cos(angles), atol=1e-12)


def test_cz_phase_flips_only_the_11_component():
    phase = sv.cz_phase(2, [(0, 1)], torch.float64, "cpu").reshape(-1)
    assert torch.equal(
        phase.real, torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=torch.float64)
    )


def test_big_endian_amplitude_order():
    vector = torch.zeros(1, 4, dtype=torch.float64)
    vector[0, 1] = 1.0  # |01>: qubit 0 in |0>, qubit 1 in |1>
    expectations = sv.expectation_z(sv.amplitude_state(vector, 2))
    assert torch.allclose(
        expectations, torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    )


def test_ansatz_preserves_norm_and_is_differentiable():
    ansatz = VariationalAnsatz(3, 2).double()
    state = angle_encode(torch.rand(5, 3, dtype=torch.float64))
    out = ansatz(state)
    assert torch.allclose(
        sv.flatten(out).abs().pow(2).sum(1), torch.ones(5, dtype=torch.float64)
    )
    sv.expectation_z(out).sum().backward()
    assert ansatz.angles.grad is not None and ansatz.angles.grad.abs().sum() > 0
