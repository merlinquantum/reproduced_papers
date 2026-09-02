"""Tests for the MerLin photonic adaptation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.photonic import (  # noqa: E402
    PhotonicMZIReadout,
    graph_unitary,
    photonic_cqrw_features,
)
from lib.qpe import adjacency_from_edges, cqrw_features  # noqa: E402


def test_photonic_unitary_is_unitary():
    A = adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    U = graph_unitary(A, t=0.7)
    np.testing.assert_allclose(U @ U.conj().T, np.eye(4), atol=1e-10)


def test_photonic_1cqrw_matches_xy_cqrw():
    """Single-photon photonic walk IS the 1-CQRW under the XY hamiltonian."""
    A = adjacency_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)])
    times = [0.2, 0.7, 1.4]
    P_phot = photonic_cqrw_features(A, 1, times)
    P_qpe = cqrw_features(A, 1, times)
    np.testing.assert_allclose(P_phot, P_qpe, atol=1e-10)


def test_photonic_2cqrw_rows_sum_to_one():
    A = adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    P = photonic_cqrw_features(A, 2, [0.3])
    # The marginalised distribution should sum to 1 per row.
    np.testing.assert_allclose(P[0].sum(axis=1), 1.0, atol=1e-9)


def test_merlin_readout_forward_grad():
    head = PhotonicMZIReadout(
        n_modes=6,
        n_photons=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        n_phase_error_samples=1,
    )
    assert head.execution_scope == "standalone_analytic_simulator"
    assert head.computation_space == "unbunched"
    assert head.n_phase_error_samples == 1
    x = torch.randn(4, head.input_dim, requires_grad=False)
    y = head(x)
    assert y.shape == (4, 1)
    # Backprop should reach the photonic parameters.
    y.sum().backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()
    )
    assert has_grad
