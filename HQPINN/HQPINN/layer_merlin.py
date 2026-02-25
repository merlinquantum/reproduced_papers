# layer_merlin.py

import numpy as np
import torch
import torch.nn as nn

import merlin as ML
from merlin import LexGrouping, QuantumLayer

import perceval as pcvl
from perceval import PS, BS

from math import comb

from .config import N_QUBITS, N_LAYERS, DTYPE


# ============================================================
#  Perceval building blocks
# ============================================================


def entangling_chain_all_modes() -> pcvl.Circuit:
    """
    Linear (non-circular) entangling chain across all 2 * n_qubits modes.

    Structure
    ---------
    - n_modes = 2 * n_qubits (dual-rail encoding)
    - Apply BS.H between (m, m+1) for m = 0 .. n_modes - 2
    """
    n_modes = 2 * N_QUBITS
    circ = pcvl.Circuit(n_modes)
    for m_idx in range(n_modes - 1):
        circ // (m_idx, BS.H())  # type: ignore
    return circ


def ansatz_layer(prefix: str) -> pcvl.Circuit:
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
    circ = pcvl.Circuit(2 * N_QUBITS)
    for i in range(N_QUBITS):
        m0 = 2 * i
        m1 = 2 * i + 1

        theta_z1 = pcvl.P(f"theta_{prefix}_{i}_0")
        theta_x = pcvl.P(f"theta_{prefix}_{i}_1")
        theta_z2 = pcvl.P(f"theta_{prefix}_{i}_2")

        circ // (m1, PS(theta_z1))  # type: ignore
        circ // (m0, BS.Rx(theta_x))  # type: ignore
        circ // (m1, PS(theta_z2))  # type: ignore

    return circ // entangling_chain_all_modes()


def feature_layer(prefix: str) -> pcvl.Circuit:
    """
    Feature map layer implemented as BS.Ry rotations.

    Parameters (symbolic)
    ---------------------
    For each logical qubit i, we introduce:
      phi_{prefix}_{i}
    """
    circ = pcvl.Circuit(2 * N_QUBITS)
    for i in range(N_QUBITS):
        phi = pcvl.P(f"phi_{prefix}_{i}")
        circ // (2 * i, BS.Ry(phi))  # type: ignore
    return circ


def build_merlin_circuit() -> pcvl.Circuit:
    """
    Pattern:
        ansatz("layer0")
        feature("layer1")
        ansatz("layer2")
        feature("layer3")
        ...
        ansatz("layer{2*(N_LAYERS-1)}")
    """
    circ = pcvl.Circuit(2 * N_QUBITS)

    for l in range(N_LAYERS):
        # Ansatz layer with even prefix: layer0, layer2, ...
        circ = circ // ansatz_layer(f"layer{2 * l}")

        # Feature layer between ansatz layers, except after the last ansatz
        if l < N_LAYERS - 1:
            circ = circ // feature_layer(f"layer{2 * l + 1}")

    return circ


# ============================================================
#  QuantumLayers factory
# ============================================================

# Dual-rail: 2 modes per logical qubit, 1 photon per qubit.
n_modes = 2 * N_QUBITS
input_state = [1, 0] * N_QUBITS
n_photons = sum(input_state)

# Fock space dimension for n_photons over n_modes modes.
fock_dim = comb(n_modes + n_photons - 1, n_photons)

# Number of logical output features used by the classical readout.
group_dim = 2 * N_QUBITS

# Grouping from Fock basis to logical features.
grouping = LexGrouping(fock_dim, group_dim)


def make_perceval_qlayer() -> QuantumLayer:
    """
    Build one QuantumLayer for the given MerLin circuit.

    Grouping is handled inside MerLin via the MeasurementStrategy.
    Call this function twice if you need two independent branches.
    """
    circuit = build_merlin_circuit()

    qlayer = QuantumLayer(
        input_size=n_modes,
        circuit=circuit,
        input_state=input_state,
        trainable_parameters=["theta"],
        input_parameters=["phi"],
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK, grouping=grouping
        ),
        dtype=DTYPE,
    )
    return qlayer


def make_interf_qlayer(n_photons: int) -> QuantumLayer:
    """
    Build one QuantumLayer for the given MerLin circuit.

    Grouping is handled inside MerLin via the MeasurementStrategy.
    Call this function twice if you need two independent branches.
    """
    input_state = [n_photons] + [0] * (n_modes - 1)  # All photons in the first mode

    # Fock space dimension for n_photons over n_modes modes.
    fock_dim = comb(n_modes + n_photons - 1, n_photons)

    grouping = LexGrouping(fock_dim, group_dim)

    # Build photonic circuit with interferometers and angle encoding.
    builder = ML.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer(trainable=True, name="layer0")
    encoding_modes = list(range(0, n_modes, 2))
    builder.add_angle_encoding(modes=encoding_modes, name="phi1_")
    builder.add_entangling_layer(trainable=True, name="layer2")
    builder.add_angle_encoding(modes=encoding_modes, name="phi3_")
    builder.add_entangling_layer(trainable=True, name="layer4")

    qlayer = QuantumLayer(
        builder=builder,
        input_state=input_state,
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK,
            grouping=grouping,
        ),
        dtype=DTYPE,
    )

    return qlayer


# ============================================================
#  MerLin quantum branch
# ============================================================


class BranchMerlin(nn.Module):
    """
    Quantum branch based on a MerLin QuantumLayer.
    """

    def __init__(
        self,
        qlayer: QuantumLayer,
        n_outputs: int = 1,
        processor: ML.MerlinProcessor | None = None,
    ) -> None:
        super().__init__()
        self.qlayer = qlayer
        self.group_dim = 2 * N_QUBITS
        self.n_outputs = n_outputs
        self.processor: ML.MerlinProcessor | None = processor

        self.readout = nn.Linear(self.group_dim, n_outputs, dtype=DTYPE)

    def _feature_map(self, x_in: torch.Tensor) -> torch.Tensor:
        x_in = x_in.to(DTYPE)

        # [N] or [N,1] → DHO style 1D encoding (only t)
        if x_in.dim() == 1:  # x_in is already 1D, treat as t
            t = x_in
        elif x_in.shape[1] == 1:
            t = x_in[:, 0]  # treat single column as t
        else:
            # [N,2] → Euler style 2D encoding (x,t)
            x = x_in[:, 0]
            t = x_in[:, 1]
            # Feature map construction for 2D case (example: φ = [π x, π t, π (x-t)])
            phi0 = np.pi * x
            phi1 = np.pi * t
            phi2 = np.pi * (x - t)
            # Stack features into [N, 3]
            return torch.stack([phi0, phi1, phi2], dim=1).to(DTYPE)

        # DHO-style 1D encoding: φ = [π t, 2π t, 3π t]
        scale = np.pi
        phi0 = scale * t
        phi1 = 2.0 * scale * t
        phi2 = 3.0 * scale * t
        return torch.stack([phi0, phi1, phi2], dim=1).to(DTYPE)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        x_in:
            - DHO style: [N] or [N,1] with t values → 1D encoding
            - Euler style: [N,2] with (x,t) pairs → 2D encoding
        """

        # phi_single: [N, 3] with features for one layer of the MerLin circuit.
        phi_single = self._feature_map(x_in)

        # Number of angle encoding layers = N_LAYERS - 1 (one encoding layer between each pair of ansatz layers).
        n_feature_layers = max(N_LAYERS - 1, 0)

        if n_feature_layers > 0:
            # Repeat phi_single for each feature layer, giving [N, 3 * n_feature_layers].
            X = torch.cat([phi_single] * n_feature_layers, dim=1)
        else:
            # No feature layers, so X is empty with shape [N, 0].
            X = torch.empty(x_in.shape[0], 0, dtype=DTYPE, device=x_in.device)

        if self.processor is None:
            # Local Execution, differentiable (SLOS)
            q_out = self.qlayer(X).to(DTYPE)  # (N, output_size)
        else:
            # Remote Execution via MerlinProcessor → shots / simulator / QPU
            # No gradient here since we only use the processor for inference, not training.
            self.qlayer.eval()
            with torch.no_grad():
                q_out = self.processor.forward(self.qlayer, X).to(DTYPE)

        # QuantumLayer output is already grouped: (N, 2 * n_qubits).
        u = self.readout(q_out)  # (N, 1)

        return u


def make_merlin_processor(processor="sim:ascella") -> ML.MerlinProcessor:
    """
    Construit un MerlinProcessor connecté au simulateur Perceval 'sim:ascella'.
    """
    rp = pcvl.RemoteProcessor(processor)
    processor = ML.MerlinProcessor(
        rp,
        microbatch_size=32,
        timeout=3600.0,
        max_shots_per_call=None,
        chunk_concurrency=1,
    )
    return processor
