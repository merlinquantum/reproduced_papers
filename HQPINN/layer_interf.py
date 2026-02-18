# layer_merlin.py

import numpy as np
import torch
import torch.nn as nn

import merlin as ML
from merlin import LexGrouping, QuantumLayer

import perceval as pcvl
from perceval import PS, BS

from math import comb


# ============================================================
#  QuantumLayer factory
# ============================================================


def make_merlin_qlayer(
    n_qubits: int,
    dtype: torch.dtype = torch.float32,
) -> QuantumLayer:
    """
    Build one QuantumLayer for the given MerLin circuit.

    Grouping is handled inside MerLin via the MeasurementStrategy.
    Call this function twice if you need two independent branches.
    """

    # Dual-rail: 2 modes per logical qubit, 1 photon per qubit.
    n_modes = 2 * n_qubits
    input_state = [1, 0] * n_qubits
    n_photons = sum(input_state)

    # Build photonic circuit with interferometers and angle encoding.
    builder = ML.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer(trainable=True, name="layer0")
    encoding_modes = list(range(0, n_modes, 2))
    builder.add_angle_encoding(modes=encoding_modes, name="phi1_")
    builder.add_entangling_layer(trainable=True, name="layer2")
    builder.add_angle_encoding(modes=encoding_modes, name="phi3_")
    builder.add_entangling_layer(trainable=True, name="layer4")

    # Full Fock space dimension for n_photons over n_modes modes.
    fock_dim = comb(n_modes + n_photons - 1, n_photons)

    # Number of logical output features used by the classical readout.
    group_dim = 2 * n_qubits

    # Grouping from Fock basis to logical features.
    grouping = LexGrouping(fock_dim, group_dim)

    qlayer = QuantumLayer(
        builder=builder,
        input_state=input_state,
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK,
            grouping=grouping,
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

        # QuantumLayer already outputs grouped features of this size.
        self.group_dim = 2 * n_qubits
        self.readout = nn.Linear(self.group_dim, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Ensure shape (N,) before encoding.
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

        # QuantumLayer output is already grouped: (N, 2 * n_qubits).
        q_out = self.qlayer(X)  # (N, output_size)
        u = self.readout(q_out)  # (N, 1)

        return u
