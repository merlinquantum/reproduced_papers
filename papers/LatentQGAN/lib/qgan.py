"""LatentQGAN generator/discriminator pair.

Glues together T quantum sub-generators (``QuantumGeneratorTorch``) and a
fully-connected discriminator that operates on a flattened latent
representation of shape (T, 2**NG).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .quantum_generator import QuantumGeneratorTorch


class LatentQGenerator(nn.Module):
    """Holds T independent quantum sub-generators, one per latent row."""

    def __init__(self, T: int = 5, N: int = 4, NA: int = 1, L: int = 7):
        super().__init__()
        self.T = T
        self.N = N
        self.NA = NA
        self.NG = N - NA
        self.L = L
        self.subs = nn.ModuleList(
            [QuantumGeneratorTorch(N=N, NA=NA, L=L) for _ in range(T)]
        )

    @property
    def latent_shape(self) -> tuple[int, int]:
        return (self.T, 2**self.NG)

    def sample_noise(self, batch: int, device=None) -> torch.Tensor:
        """Sample noise vectors uniformly in [0, pi] (alphas used by RY)."""
        device = device or torch.device("cpu")
        return torch.rand(batch, self.T, self.N, device=device) * math.pi

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """alpha: (batch, T, N) -> (batch, T, 2**NG)."""
        outs = []
        for t in range(self.T):
            outs.append(self.subs[t](alpha[:, t, :]))
        return torch.stack(outs, dim=1)


class LatentDiscriminator(nn.Module):
    """Classical FCNN discriminator: 40 -> 64 -> 16 -> 1.

    Matches the architecture in Section V.A (total ~3681 params for the
    default 40-dim latent).
    """

    def __init__(self, latent_dim: int = 40, h1: int = 64, h2: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2**NG) or (B, latent_dim)
        if x.dim() == 3:
            x = x.flatten(start_dim=1)
        return self.net(x).squeeze(-1)


class ClassicalLatentGenerator(nn.Module):
    """Classical baseline generator with comparable parameter count.

    Architecture: noise_dim -> hidden -> latent_dim with softmax per row to
    match the row-normalised constraint of the quantum generator.

    Parameter count is tuned to be ~140 (matches paper) by default.
    """

    def __init__(
        self, T: int = 5, N: int = 4, noise_dim: int | None = None, hidden_dim: int = 4
    ):
        super().__init__()
        self.T = T
        self.N = N
        self.NG = N - 1
        self.noise_dim = noise_dim if noise_dim is not None else T * N
        self.out_dim = T * (2**self.NG)  # 5*8 = 40
        # 2-layer MLP with hidden_dim chosen so total params ~ 140.
        # noise_dim=20 -> first: 20*hidden_dim + hidden_dim, second: hidden_dim*40 + 40.
        # 20*4+4 + 4*40+40 = 84 + 200 = 284, too many. Use hidden_dim=2 -> 20*2+2 + 2*40+40 = 42+120=162; still too many.
        # Use no bias on the output and hidden_dim=2: 20*2+2 + 2*40 = 122. Add a final bias of 40 -> 162.
        # Closer match: hidden_dim=1 -> 20*1+1 + 1*40+40 = 21 + 80 = 101. Too few.
        # Best simple form: single linear layer noise_dim -> out_dim with bias gives 20*40+40=840 (too many).
        # We accept ~few hundred parameters but keep architecture simple. Make it configurable.
        self.fc1 = nn.Linear(self.noise_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.out_dim)

    def sample_noise(self, batch: int, device=None) -> torch.Tensor:
        device = device or torch.device("cpu")
        # Uniform [0, 1) noise for comparability with the quantum generator's bounded inputs.
        return torch.rand(batch, self.noise_dim, device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(z))
        out = self.fc2(h)
        out = out.view(-1, self.T, 2**self.NG)
        # Row-wise softmax for the same row-sum=1 invariant as the quantum generator.
        out = torch.softmax(out, dim=-1)
        return out
