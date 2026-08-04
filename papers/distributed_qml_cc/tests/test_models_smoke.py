"""Per-pipeline import + forward-pass smoke tests."""

from __future__ import annotations

import torch
from common import PROJECT_DIR  # noqa: F401
from lib.classical_model import TinyMLP, num_mlp_parameters
from lib.model import DQMLConfig, DQMLModel


def test_classical_param_count():
    assert num_mlp_parameters(8, 8) == TinyMLP(8, 8).net[0].weight.numel() \
        + TinyMLP(8, 8).net[0].bias.numel() \
        + TinyMLP(8, 8).net[2].weight.numel() \
        + TinyMLP(8, 8).net[2].bias.numel() \
        + TinyMLP(8, 8).net[4].weight.numel() \
        + TinyMLP(8, 8).net[4].bias.numel()


def test_tinymlp_forward_shape():
    mlp = TinyMLP(8, 8)
    x = torch.randn(7, 8)
    assert mlp(x).shape == (7,)


def test_dqml_double_precision():
    """Run a small forward pass in float64 to exercise the complex128 path."""
    torch.manual_seed(0)
    m = DQMLModel(DQMLConfig(scheme="cc", n_layers=2), dtype=torch.float64)
    x = torch.randn(3, 8, dtype=torch.float64) * 0.5
    y = m(x)
    assert y.dtype == torch.float64
    assert y.shape == (3,)
