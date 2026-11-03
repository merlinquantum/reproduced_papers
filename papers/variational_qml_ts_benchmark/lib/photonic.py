"""Photonic (MerLin) counterpart of the dressed QNN for time-series regression.

The paper is gate-based (PennyLane).  To assess whether the photonic modality
changes the benchmark conclusion, we build the direct optical analogue of the
best-performing quantum model, the dressed QNN (d-QNN):

    window (l*d) --Linear--> n_modes angles --[photonic chip]--> probabilities
                 --Linear--> d prediction

The photonic chip is a standard MerLin interferometric mesh: a trainable
entangling layer, angle encoding of the classical features as phase shifts, and
a second trainable entangling layer, read out as Fock-basis probabilities under
threshold (UNBUNCHED) detection.  This mirrors the d-QNN's "PQC squeezed between
two classical linear layers" structure so the comparison is apples-to-apples.

Requires ``merlinquantum``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _output_dim(data_label: str) -> int:
    if data_label.startswith("lorenz"):
        return 3
    if data_label.startswith("henon"):
        return 2
    return 1


class PhotonicDressedQNN(nn.Module):
    """Dressed photonic QNN: linear -> photonic mesh -> linear.

    Parameters
    ----------
    data_label : str
        Dataset label (sets input/output dimension).
    seq_length : int
        Sliding-window length ``l``.
    n_modes : int
        Number of optical modes. Default value is 6.
    n_photons : int
        Number of photons injected (dual-rail-style spread). Default value is 3.
    random_id : int
        Seed for reproducible initial weights. Default value is 42.
    """

    def __init__(
        self,
        data_label: str,
        seq_length: int,
        n_modes: int = 6,
        n_photons: int = 3,
        random_id: int = 42,
        **_,
    ) -> None:
        super().__init__()
        import merlin as ml

        self.data_label = data_label
        self.seq_length = seq_length
        self.n_modes = n_modes
        self.n_photons = n_photons
        d = _output_dim(data_label)

        torch.manual_seed(random_id)
        input_modes = list(range(n_modes))
        builder = ml.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer()
        builder.add_angle_encoding(modes=input_modes, scale=float(np.pi))
        builder.add_entangling_layer()

        # Deterministic input state: photons spread across modes (dual-rail style)
        # so that every mode lies in the photon light-cone.
        input_state = [0] * n_modes
        step = max(1, n_modes // n_photons)
        placed = 0
        for m in range(0, n_modes, step):
            if placed >= n_photons:
                break
            input_state[m] = 1
            placed += 1
        for m in range(n_modes):  # top up if rounding left photons unplaced
            if placed >= n_photons:
                break
            if input_state[m] == 0:
                input_state[m] = 1
                placed += 1
        self.input_state = input_state

        self.qlayer = ml.QuantumLayer(
            input_size=len(input_modes),
            builder=builder,
            input_state=input_state,
            n_photons=n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.UNBUNCHED
            ),
        )
        self.input_layer = nn.Linear(d * seq_length, len(input_modes))
        self.output_layer = nn.Linear(self.qlayer.output_size, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.reshape(x, (x.size(0), -1))
        x = self.input_layer(x)
        probs = self.qlayer(x)
        return self.output_layer(probs)
