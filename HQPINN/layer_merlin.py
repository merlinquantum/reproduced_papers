# layer_merlin.py

import numpy as np
import torch
import torch.nn as nn

import merlin as ML
from merlin import LexGrouping, QuantumLayer

import perceval as pcvl
from perceval import PS, BS


# ============================================================
#  Perceval building blocks
# ============================================================


def entangling_chain_all_modes(n_qubits: int) -> pcvl.Circuit:
    """
    Linear (non-circular) entangling chain across all 2 * n_qubits modes.

    Structure
    ---------
    - n_modes = 2 * n_qubits (dual-rail encoding)
    - Apply BS.H between (m, m+1) for m = 0 .. n_modes - 2
    """
    n_modes = 2 * n_qubits
    circ = pcvl.Circuit(n_modes)
    for m_idx in range(n_modes - 1):
        circ // (m_idx, BS.H()) # type: ignore
    return circ


def ansatz_layer(prefix: str, n_qubits: int) -> pcvl.Circuit:
    """
    Perceval implementation of an ansatz layer in dual-rail encoding.

    Dual-rail encoding
    ------------------
    - Logical qubit i ↦ spatial modes (2*i, 2*i+1).

    Parameters (symbolic)
    ---------------------
    For each logical qubit i we introduce 3 parameters:
      theta_{prefix}_{i}_0 : "RZ-like" rotation
      theta_{prefix}_{i}_1 : "RX-like" rotation
      theta_{prefix}_{i}_2 : "RZ-like" rotation
    """
    circ = pcvl.Circuit(2 * n_qubits)
    for i in range(n_qubits):
        m0 = 2 * i
        m1 = 2 * i + 1

        theta_z1 = pcvl.P(f"theta_{prefix}_{i}_0")
        theta_x = pcvl.P(f"theta_{prefix}_{i}_1")
        theta_z2 = pcvl.P(f"theta_{prefix}_{i}_2")

        circ // (m1, PS(theta_z1)) # type: ignore
        circ // (m0, BS.Rx(theta_x)) # type: ignore
        circ // (m1, PS(theta_z2)) # type: ignore

    return circ // entangling_chain_all_modes(n_qubits)


def feature_layer(prefix: str, n_qubits: int) -> pcvl.Circuit:
    """
    Feature map layer implemented as BS.Ry rotations.

    Parameters (symbolic)
    ---------------------
    For each logical qubit i, we introduce:
      phi_{prefix}_{i}
    """
    circ = pcvl.Circuit(2 * n_qubits)
    for i in range(n_qubits):
        phi = pcvl.P(f"phi_{prefix}_{i}")
        circ // (2 * i, BS.Ry(phi)) # type: ignore
    return circ


def build_merlin_circuit(n_qubits: int) -> pcvl.Circuit:
    """
    Full photonic circuit:

        ansatz("layer0") → feature("layer1") →
        ansatz("layer2") → feature("layer3") →
        ansatz("layer4")
    """
    circ = pcvl.Circuit(2 * n_qubits)
    circ = circ // ansatz_layer("layer0", n_qubits)
    circ = circ // feature_layer("layer1", n_qubits)
    circ = circ // ansatz_layer("layer2", n_qubits)
    circ = circ // feature_layer("layer3", n_qubits)
    circ = circ // ansatz_layer("layer4", n_qubits)
    return circ


# ============================================================
#  QuantumLayer factory
# ============================================================


def make_merlin_qlayer(
    n_qubits: int,
    dtype: torch.dtype = torch.float32,
) -> QuantumLayer:
    """
    Build one QuantumLayer for the given MerLin circuit.

    Important
    ---------
    Call this function twice if you need two independent branches.
    """
    circuit = build_merlin_circuit(n_qubits)
    input_size = 2 * n_qubits

    qlayer = QuantumLayer(
        input_size=input_size,
        circuit=circuit,
        input_state=[1, 0, 1, 0, 1, 0],  # dual-rail |1> for each logical qubit
        trainable_parameters=["theta"],
        input_parameters=["phi"],
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK
        ),
        dtype=dtype,
    )
    return qlayer


# ============================================================
#  MerLin quantum branch
# ============================================================


class MerlinQuantumBranch(nn.Module):
    """
    Quantum branch based on a MerLin QuantumLayer.

    Feature map:
        φ(t) = [π t, 2π t, 3π t]
    reused for the two feature layers, giving a 6-dimensional input.
    """

    def __init__(self, qlayer: QuantumLayer, n_qubits: int) -> None:
        super().__init__()
        self.qlayer = qlayer
        self.n_qubits = n_qubits

        self.group_dim = 2 * n_qubits
        self.group = LexGrouping(self.qlayer.output_size, self.group_dim)
        self.readout = nn.Linear(self.group_dim, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)

        scale = np.pi
        phi0 = scale * t
        phi1 = 2.0 * scale * t
        phi2 = 3.0 * scale * t

        # Two feature layers → concatenate [φ0, φ1, φ2] twice
        X = torch.stack(
            [
                phi0,
                phi1,
                phi2,  # layer1
                phi0,
                phi1,
                phi2,  # layer3
            ],
            dim=1,
        )

        q_out = self.qlayer(X)  # (N, output_size)
        feat = self.group(q_out)  # (N, 2 * n_qubits)
        u = self.readout(feat)  # (N, 1)

        return u
