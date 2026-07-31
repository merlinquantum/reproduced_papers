"""MerLin photonic counterpart of the LatentQGAN quantum generator.

Each sub-generator in the original paper produces an 8-dim probability
vector (NG = 3 data qubits, after post-selection on the ancilla).
We reproduce this in MerLin by using ``DUAL_RAIL`` computation space
with 3 logical photonic qubits (6 modes, 3 photons), which yields
2 ** 3 = 8 outcomes per chip.

Input encoding: the four noise values per sub-generator are mapped to
angle encoders on four of the six modes. The remaining modes receive a
trainable bias only.

Hardware-aware reporting (per sub-generator):

    Computation space:     DUAL_RAIL
    Detector model:        threshold (UNBUNCHED-equivalent within DR subspace)
    Photon number:         3
    Number of modes:       6
    Input state:           [1, 0, 1, 0, 1, 0]
    Encoding:              angle, modes 0,2,3,5, scale=pi
    Measurement strategy:  MeasurementStrategy.PROBABILITIES
    Postselection:         dual-rail subspace
    Simulator / QPU path:  MerLin CPU simulator (analytic, shots=0)
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

try:
    import merlin as ml  # noqa: F401
    _HAS_MERLIN = True
except Exception:
    _HAS_MERLIN = False


class MerlinSubGenerator(nn.Module):
    """One photonic sub-generator returning an 8-dim probability row.

    Uses DUAL_RAIL with 3 logical qubits = 6 modes, 3 photons.
    """

    def __init__(self, N_inputs: int = 4, L: int = 7):
        super().__init__()
        if not _HAS_MERLIN:
            raise RuntimeError("merlinquantum is required for MerlinSubGenerator")
        import merlin as ml

        self.N_inputs = N_inputs
        self.L = L
        self.n_modes = 6
        self.n_photons = 3
        self.input_state = [1, 0, 1, 0, 1, 0]
        # Encode 4 input values on 4 of the 6 modes (one per dual-rail "qubit"
        # plus one extra on the second pair).
        self.input_modes: List[int] = [0, 2, 4, 1][:N_inputs]

        builder = ml.CircuitBuilder(n_modes=self.n_modes)
        # Initial trainable entangling mesh.
        builder.add_entangling_layer()
        builder.add_angle_encoding(modes=self.input_modes, scale=float(math.pi))
        # Repeat: ``L`` interleaved entangling + (1 extra) layers to roughly
        # match the depth of the gate-based generator.
        for _ in range(L):
            builder.add_entangling_layer()

        self.qlayer = ml.QuantumLayer(
            input_size=N_inputs,
            builder=builder,
            input_state=self.input_state,
            n_photons=self.n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.DUAL_RAIL,
            ),
        )
        # output_size should be 2**3 = 8 for 3 dual-rail qubits.
        assert self.qlayer.output_size == 8, (
            f"Expected output_size=8, got {self.qlayer.output_size}"
        )

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """alpha: (B, N_inputs) -> (B, 8) probabilities (already normalised)."""
        return self.qlayer(alpha)


class MerlinLatentGenerator(nn.Module):
    """T MerLin sub-generators, one per latent row (matches LatentQGenerator API)."""

    def __init__(self, T: int = 5, N: int = 4, NA: int = 1, L: int = 7):
        super().__init__()
        if not _HAS_MERLIN:
            raise RuntimeError("merlinquantum is required for MerlinLatentGenerator")
        self.T = T
        self.N = N
        self.NA = NA
        self.NG = N - NA
        self.L = L
        # Use N "input" angles per sub-gen to mirror the gate-based generator.
        self.subs = nn.ModuleList([MerlinSubGenerator(N_inputs=N, L=L) for _ in range(T)])

    def sample_noise(self, batch: int, device=None) -> torch.Tensor:
        device = device or torch.device("cpu")
        # Match the gate-model encoding range (uniform in [0, 1) for angle encoder).
        return torch.rand(batch, self.T, self.N, device=device)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        outs = []
        for t in range(self.T):
            outs.append(self.subs[t](alpha[:, t, :]))
        return torch.stack(outs, dim=1)
