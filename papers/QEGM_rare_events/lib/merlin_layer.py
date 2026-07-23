"""MerLin photonic counterpart of the gate-based VQC used by QEGM.

The paper itself is gate-based. To assess photonic feasibility we
replace ``HardwareEfficientVQC`` with a MerLin ``QuantumLayer`` built
on three-photon dual-rail-style mode pairs. Pattern A in
``MERLIN_COOKBOOK.md`` matches the role we need: a small latent vector
in -> a few probability outputs that modulate the variance of the
encoder's Gaussian latent sampling (Eq. 7 of the paper).
"""

from __future__ import annotations

import math

import merlin as ml
import torch
import torch.nn as nn


class MerlinPhotonicLayer(nn.Module):
    """Photonic surrogate for ``HardwareEfficientVQC``.

    Builds a MerLin circuit whose input encoding takes a small latent
    vector of size ``n_qubits`` and produces ``n_qubits`` quantum-randomness
    values in ``[0, 1]`` by pooling the unbunched-outcome probability
    distribution into ``n_qubits`` buckets.
    """

    def __init__(
        self,
        n_qubits: int,
        n_modes: int = 6,
        n_photons: int = 3,
        encoding_scale: float = math.pi,
    ):
        super().__init__()
        if n_modes < n_qubits:
            raise ValueError("n_modes must be >= n_qubits to support encoding")
        if n_photons < 1 or n_photons > n_modes:
            raise ValueError("n_photons must be in [1, n_modes]")
        self.n_qubits = int(n_qubits)
        self.n_modes = int(n_modes)
        self.n_photons = int(n_photons)

        input_modes = list(range(self.n_qubits))
        builder = ml.CircuitBuilder(n_modes=self.n_modes)
        builder.add_entangling_layer()
        builder.add_angle_encoding(modes=input_modes, scale=float(encoding_scale))
        builder.add_entangling_layer()
        builder.add_entangling_layer()

        input_state = [0] * self.n_modes
        placed = 0
        # Spread photons across active modes per cookbook design principle #1.
        step = max(1, self.n_modes // self.n_photons)
        for k in range(self.n_photons):
            input_state[min(self.n_modes - 1, k * step)] = 1
            placed += 1
        if placed != self.n_photons:
            for i in range(self.n_modes):
                if input_state[i] == 0 and placed < self.n_photons:
                    input_state[i] = 1
                    placed += 1

        # merlin >= 0.4: the computation space is owned by the measurement
        # strategy factory and the photon count is inferred from input_state.
        self.layer = ml.QuantumLayer(
            builder=builder,
            input_state=input_state,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.UNBUNCHED,
            ),
            dtype=torch.float32,
        )

        out_dim = int(self.layer.output_size)
        if out_dim < self.n_qubits:
            raise RuntimeError(
                "Photonic output size smaller than requested number of "
                f"quantum-randomness channels ({out_dim} < {self.n_qubits}). "
                "Increase n_modes or n_photons."
            )
        self.bucket_size = out_dim // self.n_qubits
        self.out_dim = out_dim

    @property
    def output_size(self) -> int:
        return self.out_dim

    def hardware_settings(self) -> dict:
        return {
            "computation_space": "UNBUNCHED",
            "detector_model": "threshold",
            "n_photons": self.n_photons,
            "n_modes": self.n_modes,
            "input_state": [int(s) for s in self.layer.input_state]
            if hasattr(self.layer, "input_state")
            else None,
            "encoding": f"angle, modes 0..{self.n_qubits - 1}",
            "measurement_strategy": "MeasurementStrategy.probs(UNBUNCHED)",
            "postselection": "none",
            "simulator": "MerLin CPU analytic",
            "shots": 0,
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        probs = self.layer(z)
        batch = probs.shape[0]
        buckets = probs[:, : self.bucket_size * self.n_qubits].reshape(
            batch, self.n_qubits, self.bucket_size
        )
        # Each bucket is a valid sub-distribution; pool to a single number in [0, 1].
        pooled = buckets.sum(dim=-1) * float(self.n_qubits)
        # Renormalize against the full output mass so values stay in [0, 1].
        denom = probs[:, : self.bucket_size * self.n_qubits].sum(dim=-1, keepdim=True)
        denom = torch.clamp(denom, min=1e-6)
        pooled = pooled / denom
        return torch.clamp(pooled, 0.0, 1.0)
