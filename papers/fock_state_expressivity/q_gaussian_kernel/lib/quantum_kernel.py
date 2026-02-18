from __future__ import annotations

import random

import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer


def _ps_random():
    return pcvl.PS(phi=random.random() * np.pi)


def _generic_block(trainable: bool, label: str):
    if trainable:
        return pcvl.GenericInterferometer(
            2,
            lambda i: (
                pcvl.BS(theta=pcvl.P(f"theta_{label}1{i}"))
                // pcvl.PS(phi=pcvl.P(f"theta_ps{label}1{i}"))
                // pcvl.BS(theta=pcvl.P(f"theta_{label}2{i}"))
                // pcvl.PS(phi=pcvl.P(f"theta_ps{label}2{i}"))
            ),
            shape=pcvl.InterferometerShape.RECTANGLE,
        )
    return pcvl.GenericInterferometer(
        2, lambda i: pcvl.BS() // _ps_random(), shape=pcvl.InterferometerShape.RECTANGLE
    )


def build_circuit(cfg: dict) -> pcvl.Circuit:
    circuit_type = cfg.get("circuit", "mzi")
    train = cfg.get("train_circuit", False)

    circuit = pcvl.Circuit(2)
    if circuit_type == "mzi":
        left = (
            pcvl.BS()
            if not train
            else pcvl.BS(
                theta=pcvl.P("theta_l_1"),
                phi_bl=pcvl.P("theta_l_2"),
                phi_tl=pcvl.P("theta_l_3"),
                phi_br=pcvl.P("theta_l_4"),
                phi_tr=pcvl.P("theta_l"),
            )
        )
        right = (
            pcvl.BS()
            if not train
            else pcvl.BS(
                theta=pcvl.P("theta_r_1"),
                phi_bl=pcvl.P("theta_r_2"),
                phi_tl=pcvl.P("theta_r_3"),
                phi_br=pcvl.P("theta_r_4"),
                phi_tr=pcvl.P("theta_r"),
            )
        )
    elif circuit_type == "general_all_angles":
        if train:

            def unit(label: str):
                return pcvl.GenericInterferometer(
                    2,
                    lambda i: (
                        pcvl.BS(
                            theta=pcvl.P(f"theta_{label}1{i}"),
                            phi_tr=pcvl.P(f"theta_{label}2{i}"),
                            phi_br=pcvl.P(f"theta_{label}3{i}"),
                            phi_tl=pcvl.P(f"theta_{label}4{i}"),
                            phi_bl=pcvl.P(f"theta_{label}5{i}"),
                        )
                        // pcvl.PS(phi=pcvl.P(f"theta_ps{label}1{i}"))
                        // pcvl.BS(
                            theta=pcvl.P(f"theta_{label}6{i}"),
                            phi_tr=pcvl.P(f"theta_{label}7{i}"),
                            phi_br=pcvl.P(f"theta_{label}8{i}"),
                            phi_tl=pcvl.P(f"theta_{label}9{i}"),
                            phi_bl=pcvl.P(f"theta_{label}10{i}"),
                        )
                        // pcvl.PS(phi=pcvl.P(f"theta_ps{label}2{i}"))
                    ),
                    shape=pcvl.InterferometerShape.RECTANGLE,
                )

            left = unit("l")
            right = unit("r")
        else:
            left = _generic_block(False, "l")
            right = _generic_block(False, "r")
    elif circuit_type == "general":
        left = _generic_block(train, "l")
        right = _generic_block(train, "r")
    elif circuit_type == "spiral":
        if train:
            left = _generic_block(True, "l")
            right = pcvl.GenericInterferometer(
                2,
                lambda i: (
                    pcvl.BS()
                    // pcvl.PS(phi=pcvl.P(f"theta_psr1{i}"))
                    // pcvl.BS()
                    // pcvl.PS(phi=pcvl.P(f"theta_psr2{i}"))
                ),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
        else:
            left = _generic_block(False, "l")
            right = _generic_block(False, "r")
    elif circuit_type == "ps_based":
        if train:
            left = pcvl.GenericInterferometer(
                2,
                lambda i: (
                    pcvl.BS()
                    // pcvl.PS(phi=pcvl.P(f"theta_l1{i}"))
                    // pcvl.BS()
                    // pcvl.PS(phi=pcvl.P(f"theta_{i}"))
                ),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            right = pcvl.GenericInterferometer(
                2,
                lambda i: (
                    pcvl.BS()
                    // pcvl.PS(phi=pcvl.P(f"theta_r1{i}"))
                    // pcvl.BS()
                    // pcvl.PS(phi=pcvl.P(f"theta_r2{i}"))
                ),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
        else:
            left = _generic_block(False, "l")
            right = _generic_block(False, "r")
    elif circuit_type == "bs_based":
        if train:
            left = pcvl.GenericInterferometer(
                2,
                lambda i: (
                    pcvl.BS(theta=pcvl.P(f"theta_l1{i}_bs"))
                    // pcvl.PS(phi=random.random() * np.pi)
                    // pcvl.BS(theta=pcvl.P(f"theta_l2{i}_bs"))
                    // pcvl.PS(phi=random.random() * np.pi)
                ),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            right = pcvl.GenericInterferometer(
                2,
                lambda i: (
                    pcvl.BS(theta=pcvl.P(f"theta_r1{i}_bs"))
                    // pcvl.PS(phi=random.random() * np.pi)
                    // pcvl.BS(theta=pcvl.P(f"theta_r2{i}_bs"))
                    // pcvl.PS(phi=random.random() * np.pi)
                ),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
        else:
            left = _generic_block(False, "l")
            right = _generic_block(False, "r")
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")

    circuit.add(0, left, merge=True)
    circuit.add(0, pcvl.PS(pcvl.P("δ")), merge=True)
    circuit.add(0, right, merge=True)
    return circuit


class ScaleLayer(nn.Module):
    def __init__(self, dim: int, scale_type: str = "learned"):
        super().__init__()
        self.scale_type = scale_type
        if scale_type == "learned":
            self.scale = nn.Parameter(torch.full((dim,), 1 / (2 * torch.pi)))
        elif scale_type == "paper":
            self.register_buffer("scale", torch.full((dim,), 1 / torch.pi))
        else:
            constants = {
                "2pi": 2 * torch.pi,
                "pi": torch.pi,
                "/pi": 1 / torch.pi,
                "1": 1.0,
                "0.1": 0.1,
                "/2pi": 1 / (2 * torch.pi),
                "/3pi": 1 / (3 * torch.pi),
                "/4pi": 1 / (4 * torch.pi),
            }
            value = constants.get(scale_type)
            if value is None:
                raise ValueError(f"Unknown scale_type: {scale_type}")
            self.register_buffer("scale", torch.full((dim,), value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_type == "paper":
            return x * self.scale - torch.pi / 2
        return x * self.scale


def build_quantum_kernel(num_photons: int, cfg: dict) -> nn.Module:
    """Instantiate the quantum kernel model for a specific photon count."""
    if num_photons % 2 == 0:
        input_state = [num_photons // 2, num_photons // 2]
    else:
        input_state = [num_photons // 2 + 1, num_photons // 2]

    scale_layer = ScaleLayer(1, cfg.get("scale_type", "learned"))
    circuit = build_circuit(cfg)
    train_params: list[str] = ["theta"] if cfg.get("train_circuit", False) else []
    no_bunching = bool(cfg.get("no_bunching", False))
    computation_space = (
        ComputationSpace.UNBUNCHED if no_bunching else ComputationSpace.FOCK
    )

    quantum_layer = QuantumLayer(
        input_size=1,
        circuit=circuit,
        trainable_parameters=train_params,
        input_parameters=["δ"],
        input_state=input_state,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=computation_space
        ),
    )

    linear = nn.Linear(quantum_layer.output_size, 1)
    return nn.Sequential(scale_layer, quantum_layer, linear)
