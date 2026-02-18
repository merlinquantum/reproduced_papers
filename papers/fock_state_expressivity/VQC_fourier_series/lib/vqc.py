from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import perceval as pcvl
import torch
import torch.nn as nn
from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer


class ScaleLayer(nn.Module):
    """Feature scaling layer matching the notebook implementation."""

    _CONSTANTS = {
        "2pi": lambda dim: torch.full((dim,), 2 * torch.pi),
        "/2pi": lambda dim: torch.full((dim,), 1 / (2 * torch.pi)),
        "/2": lambda dim: torch.full((dim,), 0.5),
        "2": lambda dim: torch.full((dim,), 2.0),
        "pi": lambda dim: torch.full((dim,), torch.pi),
        "/pi": lambda dim: torch.full((dim,), 1 / torch.pi),
        "/3pi": lambda dim: torch.full((dim,), 1 / (3 * torch.pi)),
        "1": lambda dim: torch.ones(dim),
    }

    def __init__(self, dim: int, scale_type: str = "learned"):
        super().__init__()
        scale_type = scale_type.lower()
        if scale_type == "learned":
            self.scale = nn.Parameter(torch.rand(dim))
        elif scale_type in self._CONSTANTS:
            self.register_buffer("scale", self._CONSTANTS[scale_type](dim))
        else:
            raise ValueError(f"Unsupported scale_type: {scale_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def create_vqc_general(num_modes: int, input_size: int) -> pcvl.Circuit:
    wl = pcvl.GenericInterferometer(
        num_modes,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_li{i}_ps"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_lo{i}_ps"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c_var = pcvl.Circuit(num_modes)
    for i in range(input_size):
        px = pcvl.P(f"px{i + 1}")
        c_var.add(i + (num_modes - input_size) // 2, pcvl.PS(px))

    wr = pcvl.GenericInterferometer(
        num_modes,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_ri{i}_ps"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_ro{i}_ps"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    circuit = pcvl.Circuit(num_modes)
    circuit.add(0, wl, merge=True)
    circuit.add(0, c_var, merge=True)
    circuit.add(0, wr, merge=True)
    return circuit


def build_model(
    initial_state: Iterable[int],
    cfg: dict,
) -> nn.Module:
    num_modes = int(cfg.get("num_modes", 3))
    input_size = int(cfg.get("input_size", 1))
    scale_type = cfg.get("scale_type", "/pi")
    trainable_params = cfg.get("trainable_parameters", ["theta"])
    input_parameters = cfg.get("input_parameters", ["px"])
    no_bunching = bool(cfg.get("no_bunching", False))
    computation_space = (
        ComputationSpace.UNBUNCHED if no_bunching else ComputationSpace.FOCK
    )

    scale_layer = ScaleLayer(input_size, scale_type=scale_type)
    vqc = QuantumLayer(
        input_size=input_size,
        circuit=create_vqc_general(num_modes, input_size),
        trainable_parameters=trainable_params,
        input_parameters=input_parameters,
        input_state=list(initial_state),
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=computation_space
        ),
    )

    linear_layer = nn.Linear(vqc.output_size, 1)
    model = nn.Sequential(scale_layer, vqc, linear_layer)

    return model


class VQCFactory:
    """Factory to create identical VQC instances for different initial Fock states."""

    def __init__(self, cfg: dict):
        self.cfg = deepcopy(cfg)

    def build(self, initial_state: Iterable[int]) -> nn.Module:
        return build_model(initial_state, self.cfg)
