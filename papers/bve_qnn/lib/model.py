"""Photonic dual-rail QNN architecture (MerLin) for the BVE reproduction.

Encodes the paper's N=6 qubit, l=32 layer Hardware-Efficient Ansatz (HEA)
QNN (main.tex, Section IV + Section V.A) using dual-rail photonic encoding:
1 logical qubit = 2 photonic modes, 1 photon.

Since linear optics cannot natively implement a fixed CNOT (KLM theorem),
inter-qubit entanglement is provided by trainable photonic mixing
beamsplitters instead of the qubit circuit's fixed CNOT gates.
"""

from __future__ import annotations

from typing import Any

import perceval as pcvl
import torch
import torch.nn as nn
from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer

QNN_INPUT_FEATURES = ["t", "x", "y", "z"]


def add_dual_rail_single_qubit_block(
    circuit: pcvl.Circuit, qubit: int, prefix: str
) -> None:
    # A dual-rail single-qubit trainable block with 3 parameters, mirroring
    # the HEA count of 3 trainable parameters per qubit per layer.
    left = 2 * qubit
    right = 2 * qubit + 1

    circuit.add(
        (left, right),
        pcvl.BS(
            theta=pcvl.P(f"{prefix}_theta_q{qubit}"),
            phi_tr=pcvl.P(f"{prefix}_phi_q{qubit}"),
        ),
    )
    circuit.add(left, pcvl.PS(pcvl.P(f"{prefix}_phase_q{qubit}")))


def add_trainable_photonic_mixing(
    circuit: pcvl.Circuit, prefix: str, n_qubits: int
) -> None:
    # Trainable inter-qubit coupling, compensating for the absence of a
    # native CNOT in linear optics. Adds 2*(n_qubits-1) parameters/layer.
    for qubit in range(n_qubits - 1):
        circuit.add(
            (2 * qubit + 1, 2 * (qubit + 1)),
            pcvl.BS(
                theta=pcvl.P(f"{prefix}_mix_theta_q{qubit}"),
                phi_tr=pcvl.P(f"{prefix}_mix_phi_q{qubit}"),
            ),
        )


def build_merlin_dual_rail_circuit(n_qubits: int, depth: int) -> pcvl.Circuit:
    n_modes = 2 * n_qubits
    circuit = pcvl.Circuit(n_modes)
    input_index = 1

    # main.tex, Section IV.c-d: serial trainable-frequency feature map.
    for feature_id, _feature_name in enumerate(QNN_INPUT_FEATURES):
        for qubit in range(n_qubits):
            left = 2 * qubit
            right = 2 * qubit + 1
            circuit.add(
                (left, right),
                pcvl.BS(theta=pcvl.P(f"input{input_index}"), phi_tr=0.0),
            )
            input_index += 1

        if feature_id < 3:
            for qubit in range(n_qubits):
                add_dual_rail_single_qubit_block(
                    circuit, qubit, prefix=f"fm_l{feature_id}"
                )
            add_trainable_photonic_mixing(
                circuit, prefix=f"fm_l{feature_id}", n_qubits=n_qubits
            )

    # main.tex, Section IV + Section V.A: main HEA ansatz with `depth` layers.
    for layer in range(depth):
        for qubit in range(n_qubits):
            add_dual_rail_single_qubit_block(circuit, qubit, prefix=f"hea_l{layer}")
        add_trainable_photonic_mixing(
            circuit, prefix=f"hea_l{layer}", n_qubits=n_qubits
        )

    return circuit


class MerlinDualRailPaperQNN(nn.Module):
    def __init__(self, n_qubits: int = 6, depth: int = 32):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth

        self.gamma = nn.Parameter(
            torch.ones(len(QNN_INPUT_FEATURES), n_qubits, dtype=torch.float64)
        )

        circuit = build_merlin_dual_rail_circuit(n_qubits, depth)

        self.quantum_layer = QuantumLayer(
            input_size=len(QNN_INPUT_FEATURES) * n_qubits,
            circuit=circuit,
            input_state=[1, 0] * n_qubits,
            trainable_parameters=["fm", "hea"],
            input_parameters=["input"],
            measurement_strategy=MeasurementStrategy.mode_expectations(
                computation_space=ComputationSpace.DUAL_RAIL
            ),
            dtype=torch.float64,
        )

        # main.tex, Section III: qubit observable C = sum_m Z_m.
        # Dual-rail equivalent: sum_m (<n_left_m> - <n_right_m>).
        self.register_buffer(
            "magnetisation_weights",
            torch.tensor([1.0, -1.0] * n_qubits, dtype=torch.float64),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features has shape (batch_size, 4): columns are (t, x, y, z).
        features = features.to(dtype=torch.float64)
        encoded_features = features.unsqueeze(-1) * self.gamma.unsqueeze(0)
        encoded_features = encoded_features.reshape(features.shape[0], -1)

        mode_expectations = self.quantum_layer(encoded_features)
        return mode_expectations @ self.magnetisation_weights


class OutputScaledQNN(nn.Module):
    def __init__(self, qnn: nn.Module, initial_scale: float, initial_shift: float):
        super().__init__()
        self.qnn = qnn
        # Learnable affine output map (main.tex): psi = alpha_scale * QNN(...) + alpha_shift
        self.output_scale = nn.Parameter(
            torch.tensor(float(initial_scale), dtype=torch.float64)
        )
        self.output_shift = nn.Parameter(
            torch.tensor(float(initial_shift), dtype=torch.float64)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw_output = self.qnn(features)
        if raw_output.ndim > 1:
            raw_output = raw_output.squeeze(-1)
        return self.output_scale * raw_output + self.output_shift


def build_model(cfg: dict[str, Any], targets_tensor: torch.Tensor) -> OutputScaledQNN:
    model_params = cfg.get("model", {}).get("params", {})
    n_qubits = int(model_params.get("n_qubits", 6))
    depth = int(model_params.get("depth", 32))

    qnn = MerlinDualRailPaperQNN(n_qubits=n_qubits, depth=depth)
    return OutputScaledQNN(
        qnn=qnn,
        initial_scale=float(targets_tensor.std()),
        initial_shift=float(targets_tensor.mean()),
    )


__all__ = [
    "QNN_INPUT_FEATURES",
    "MerlinDualRailPaperQNN",
    "OutputScaledQNN",
    "build_merlin_dual_rail_circuit",
    "build_model",
]
