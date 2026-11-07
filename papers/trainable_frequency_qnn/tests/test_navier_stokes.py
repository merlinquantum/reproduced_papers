from __future__ import annotations

import common  # noqa: F401  (puts the paper package on sys.path)
import numpy as np
import pytest
import torch
from torch import nn
from trainable_frequency_qnn.lib.navier_stokes import (
    StreamPressureModel,
    maerm,
    momentum_residuals,
    velocity_pressure,
)

REYNOLDS = 100.0


class AnalyticStream(nn.Module):
    """psi(x, y, t) = sin(x) sin(y) exp(-t) evaluated on transformed inputs."""

    def __init__(self, scales, offsets):
        super().__init__()
        self.scales, self.offsets = scales, offsets

    def forward(self, inputs):
        x, y, t = ((inputs - self.offsets) / self.scales).unbind(1)
        return torch.sin(x) * torch.sin(y) * torch.exp(-t)


class AnalyticPressure(nn.Module):
    """p(x, y, t) = x * y * t evaluated on transformed inputs."""

    def __init__(self, scales, offsets):
        super().__init__()
        self.scales, self.offsets = scales, offsets

    def forward(self, inputs):
        x, y, t = ((inputs - self.offsets) / self.scales).unbind(1)
        return x * y * t


def _expected_residuals(x, y, t):
    decay = torch.exp(-t)
    u = torch.sin(x) * torch.cos(y) * decay
    v = -torch.cos(x) * torch.sin(y) * decay
    u_x = torch.cos(x) * torch.cos(y) * decay
    u_y = -torch.sin(x) * torch.sin(y) * decay
    v_x = torch.sin(x) * torch.sin(y) * decay
    v_y = -torch.cos(x) * torch.cos(y) * decay
    laplace_u = -2 * u
    laplace_v = -2 * v
    zeta_x = -u + u * u_x + v * u_y - laplace_u / REYNOLDS + y * t
    zeta_y = -v + u * v_x + v * v_y - laplace_v / REYNOLDS + x * t
    return u, v, zeta_x, zeta_y


@pytest.mark.parametrize("transform", ["raw", "minmax"])
def test_residuals_match_analytic_flow(transform):
    # For "minmax" the networks see z = a * coordinate + b and the code must
    # recover physical derivatives through the chain-rule scales.
    if transform == "raw":
        scales = torch.ones(3, dtype=torch.float64)
        offsets = torch.zeros(3, dtype=torch.float64)
    else:
        scales = torch.tensor([2 / 7, 2 / 4, 2 / 5.5], dtype=torch.float64)
        offsets = torch.tensor([-9 / 7, 0.0, -1.0], dtype=torch.float64)
    model = StreamPressureModel(
        AnalyticStream(scales, offsets), AnalyticPressure(scales, offsets)
    )
    generator = torch.Generator().manual_seed(0)
    physical = torch.rand(
        (16, 3), generator=generator, dtype=torch.float64
    ) * torch.tensor([7.0, 4.0, 5.5], dtype=torch.float64) + torch.tensor(
        [1.0, -2.0, 0.0], dtype=torch.float64
    )
    inputs = physical * scales + offsets

    fields = velocity_pressure(model, inputs.clone(), scales, create_graph=False)
    zeta_x, zeta_y = momentum_residuals(model, inputs.clone(), scales, REYNOLDS)
    u, v, expected_x, expected_y = _expected_residuals(*physical.unbind(1))
    assert torch.allclose(fields["u"], u)
    assert torch.allclose(fields["v"], v)
    assert torch.allclose(zeta_x, expected_x)
    assert torch.allclose(zeta_y, expected_y)


def test_maerm_definition():
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    prediction = reference + np.array([0.3, -0.3, 0.3, -0.3, 0.3])
    assert maerm(prediction, reference)["maerm"] == pytest.approx(10.0)
    signed = maerm(-prediction, -reference)
    assert signed["maerm"] == pytest.approx(10.0)
    assert signed["maerm_abs_median"] == pytest.approx(10.0)
