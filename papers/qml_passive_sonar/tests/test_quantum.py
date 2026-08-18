"""Unit tests for the pure-PyTorch PQC simulator."""

from __future__ import annotations

import numpy as np
import torch
from lib.quantum import PQC


def test_pqc_output_shape_and_range():
    torch.manual_seed(0)
    pqc = PQC(n_qubits=4, n_layers=2)
    angles = torch.rand(8, 4) * np.pi
    z = pqc(angles)
    assert z.shape == (8, 4)
    assert torch.all(z.real >= -1.000001)
    assert torch.all(z.real <= 1.000001)


def test_pqc_zero_state_with_zero_angles_returns_plus_one():
    """All-zero encoding and all-zero variational params leaves |0...0>, so <Z>=+1."""
    pqc = PQC(n_qubits=3, n_layers=2)
    with torch.no_grad():
        pqc.theta.zero_()
    angles = torch.zeros(1, 3)
    z = pqc(angles)
    assert torch.allclose(z, torch.ones_like(z), atol=1e-5)


def test_pqc_gradients_flow():
    pqc = PQC(n_qubits=3, n_layers=2)
    angles = torch.rand(4, 3, requires_grad=True) * np.pi
    z = pqc(angles)
    loss = z.sum()
    loss.backward()
    assert pqc.theta.grad is not None
    assert torch.isfinite(pqc.theta.grad).all()
