"""Classical autoencoder used as the feature-compression front end of the
Guided Quantum Compressor (GQC).

Encoder: ``input_dim -> 256 -> 128 -> 64 -> n_qubits`` (ReLU after each
hidden layer, linear output at the latent layer — the latent feeds the VQC
as one angle-encoded feature per qubit).

Decoder mirrors the encoder: ``n_qubits -> 64 -> 128 -> 256 -> input_dim``
(ReLU hidden, linear output).
"""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Feed-forward encoder mapping raw features to the VQC's latent space."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Feed-forward decoder mirroring :class:`Encoder`."""

    def __init__(
        self, output_dim: int, hidden_dims: list[int], latent_dim: int
    ) -> None:
        super().__init__()
        reversed_hidden_dims = list(reversed(hidden_dims))
        layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in reversed_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


__all__ = ["Encoder", "Decoder"]
