from __future__ import annotations

import math

import torch
from common import PROJECT_DIR  # noqa: F401
from lib.circuit import ParamLayout, num_parameters
from lib.model import DQMLConfig, DQMLModel
from lib.simulator import (
    apply_cnot,
    apply_h,
    apply_rx,
    init_state,
    marginal_probabilities,
)


def test_param_layout_counts_match_models():
    for scheme in ["non", "nc", "cc", "qc"]:
        for L in [3, 5, 9]:
            layout = ParamLayout.for_scheme(scheme, L)
            model = DQMLModel(DQMLConfig(scheme=scheme, n_layers=L))
            assert num_parameters(scheme, L) == layout.total
            assert layout.total == model.num_parameters()


def test_bell_state_via_simulator():
    state = init_state(2, batch_size=1, dtype=torch.complex64)
    state = apply_h(state, 0, torch.complex64)
    state = apply_cnot(state, 0, 1, torch.complex64)
    probs = marginal_probabilities(state, [0, 1])[0]
    assert torch.allclose(probs, torch.tensor([[0.5, 0.0], [0.0, 0.5]]), atol=1e-6)


def test_rx_rotation_probabilities():
    state = init_state(1, batch_size=1, dtype=torch.complex64)
    theta = torch.tensor(0.7)
    state = apply_rx(state, theta, 0, torch.complex64)
    probs = marginal_probabilities(state, [0])[0]
    expected = torch.tensor([math.cos(0.35) ** 2, math.sin(0.35) ** 2])
    assert torch.allclose(probs, expected, atol=1e-6)


def test_model_forward_is_differentiable():
    torch.manual_seed(0)
    model = DQMLModel(DQMLConfig(scheme="cc", n_layers=2))
    x = torch.randn(4, 8, requires_grad=False) * 0.5
    y = model(x).sum()
    y.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
