"""Classical autoencoder used as the feature-compression front end of the
Guided Quantum Compressor (GQC).

Encoder: ``input_dim -> 256 -> 128 -> 64 -> n_qubits`` (ReLU after each
hidden layer, linear output at the latent layer — the latent feeds the VQC
as one angle-encoded feature per qubit).

Decoder mirrors the encoder: ``n_qubits -> 64 -> 128 -> 256 -> input_dim``
(ReLU hidden, linear output).

The latent dimension must be strictly smaller than the input dimension. The
paper states this as a property of the architecture (Section 2.1: "a
feedforward neural net (FNN) with ReLU activation functions is used to encode
the data in a lower latent dimension"), and a latent at least as wide as the
input would let the pair learn the identity, defeating the compression the
GQC depends on. Both classes reject such a configuration rather than train a
model that silently does nothing useful.

Note that this constrains only the latent against the input. It does not
require the hidden dimensions to decrease: the paper's own encoder widens to
256 before narrowing, from a 29-feature input.
"""

from __future__ import annotations

import torch
from torch import nn


def _validate_latent_dim(latent_dim: int, outer_dim: int, outer_name: str) -> None:
    """Reject a latent dimension that is not a compression of ``outer_dim``.

    Raises
    ------
    ValueError
        If ``latent_dim`` is below 1, or is not strictly smaller than
        ``outer_dim``.
    """
    if latent_dim < 1:
        raise ValueError(f"latent_dim must be >= 1; got {latent_dim}")
    if latent_dim >= outer_dim:
        raise ValueError(
            f"latent_dim={latent_dim} must be strictly smaller than "
            f"{outer_name}={outer_dim}: the autoencoder has to compress. "
            "The paper specifies a lower latent dimension (Section 2.1); a "
            "latent at least as wide as the data lets the encoder/decoder "
            "pair learn the identity and the GQC loses the compression it "
            "relies on."
        )


class Encoder(nn.Module):
    """Feed-forward encoder mapping raw features to the VQC's latent space."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()
        _validate_latent_dim(latent_dim, input_dim, "input_dim")
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
        _validate_latent_dim(latent_dim, output_dim, "output_dim")
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
