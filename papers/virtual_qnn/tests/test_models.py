"""Focused behaviour of the VQEM, VQC, VVQE models and the SDP baseline."""

from __future__ import annotations

import logging
import math

import pytest
import torch
from common import PROJECT_DIR  # noqa: F401  (puts the repo root on sys.path)
from virtual_qnn.lib import sdp, vqc, vqem, vvqe
from virtual_qnn.lib.training import optimize

DTYPE = torch.complex128
DEVICE = torch.device("cpu")


def test_budget_coefficients_preserve_trace_and_overhead():
    c1, c2 = vqem.budget_coefficients(2.5)
    assert c1 + c2 == pytest.approx(1.0)
    assert abs(c1) + abs(c2) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        vqem.budget_coefficients(0.5)


def test_vqem_unmitigated_error_and_alg1_initial_overhead():
    model = vqem.VQEMModel(
        "amplitude_damping",
        0.3,
        2,
        "trainable",
        2.0,
        1.0,
        (0.0, 0.0),
        torch.Generator().manual_seed(0),
        DTYPE,
    )
    with torch.no_grad():
        outputs = model(DEVICE)
    assert float(outputs["overhead"]) == pytest.approx(3.0)
    assert float(torch.real(torch.trace(outputs["sigma"]))) == pytest.approx(1.0)
    # Reference Fig. 4 data: HS^2 = 0.0582 when the penalty is too weak to mitigate.
    assert model.unmitigated_hs_squared(DEVICE) == pytest.approx(0.0583, abs=2e-4)


def test_vqem_alg2_learns_inverse_above_minimum_overhead():
    logging.disable(logging.INFO)
    try:
        model = vqem.VQEMModel(
            "depolarizing",
            0.2,
            5,
            "budget",
            2.0,
            1.5,
            (0.0, 0.0),
            torch.Generator().manual_seed(1),
            DTYPE,
        )
        history = optimize(
            model.parameters(),
            lambda: (model(DEVICE)["hs_squared"], {}),
            {
                "iterations": 150,
                "lr": 0.05,
                "scheduler_factor": 0.5,
                "scheduler_patience": 10,
            },
        )
    finally:
        logging.disable(logging.NOTSET)
    # Budget 1.5 exceeds the depolarizing optimum 1.375, so near-exact inversion exists.
    assert history[-1]["loss"] < 1e-5


def test_vqc_parameter_counts():
    generator = torch.Generator().manual_seed(0)
    two_branch = vqc.VirtualQuantumClassifier(2, 2, 2.0, (0.02, 0.05), generator, DTYPE)
    conventional = vqc.VirtualQuantumClassifier(
        2, 2, 1.0, (0.02, 0.05), generator, DTYPE
    )
    matched = vqc.VirtualQuantumClassifier(2, 4, 1.0, (0.02, 0.05), generator, DTYPE)
    assert (
        two_branch.n_parameters(),
        conventional.n_parameters(),
        matched.n_parameters(),
    ) == (16, 8, 16)
    x = torch.rand(5, 2, generator=generator, dtype=torch.float64) * 2 - 1
    assert conventional(x).shape == (5,)


def test_vqc_encoding_produces_pure_states():
    x = torch.tensor([[0.3, -0.6], [0.9, 0.1]], dtype=torch.float64)
    rho = vqc.encode_points(x, 2, DTYPE)
    purity = torch.real(torch.einsum("bij,bji->b", rho, rho))
    assert torch.allclose(purity, torch.ones(2, dtype=torch.float64))


def test_chebyshev_coefficients_match_quadrature_and_bound():
    order = 100
    analytic = vvqe.negativity_chebyshev_coefficients(order)
    theta = torch.linspace(0, math.pi, 20001, dtype=torch.float64)
    f = (torch.cos(theta).abs() - torch.cos(theta)) / 2
    for n in (0, 1, 2, 4, 50):
        quadrature = (2 / math.pi) * torch.trapz(f * torch.cos(n * theta), theta).item()
        quadrature *= 0.5 if n == 0 else 1.0
        assert analytic[n] == pytest.approx(quadrature, abs=1e-6)
    xs = torch.linspace(-1, 1, 101, dtype=torch.float64)
    series = sum(a * torch.cos(n * torch.arccos(xs)) for n, a in enumerate(analytic))
    assert float((series - (xs.abs() - xs) / 2).abs().max()) <= 2 / (
        math.pi * (order + 1)
    )


def test_exact_negativity():
    sigma = torch.diag(torch.tensor([0.8, 0.5, -0.2, -0.1], dtype=DTYPE))
    negativity, minimum = vvqe.exact_negativity(sigma)
    assert negativity == pytest.approx(0.3)
    assert minimum == pytest.approx(-0.2)


def test_sdp_minimum_overhead_matches_closed_forms():
    pytest.importorskip("cvxpy")
    for level in (0.1, 0.3):
        result = sdp.min_sampling_overhead("depolarizing", level)
        assert result["gamma_min"] == pytest.approx(
            sdp.depolarizing_min_overhead_analytic(level), abs=1e-5
        )
    assert sdp.min_sampling_overhead("amplitude_damping", 0.3)[
        "gamma_min"
    ] == pytest.approx(1.3 / 0.7, abs=1e-5)
