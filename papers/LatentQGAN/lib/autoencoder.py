"""Convolutional autoencoder used in LatentQGAN.

The encoder ends in a (T, 2**NG) tensor that is normalised row-wise so each
row is a probability distribution compatible with the quantum-generator
post-selection output. By default T = 5, NG = 3 → latent shape (5, 8).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder: image -> (T, 2**NG) latent.

    The exact channel widths follow Fig.2 of the paper qualitatively;
    the final flattened size is exposed as ``flat_features``.
    """

    def __init__(self, T: int = 5, NG: int = 3):
        super().__init__()
        self.T = T
        self.NG = NG
        self.latent_dim = T * (2 ** NG)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 28 -> 24
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)  # 24 -> 20
        self.flat_features = 10 * 20 * 20  # 4000
        self.fc1 = nn.Linear(self.flat_features, 400)
        self.fc2 = nn.Linear(400, self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # reshape to (B, T, 2**NG) and normalise row-wise so each row sums to 1
        x = x.view(-1, self.T, 2 ** self.NG)
        x = x + 1e-8  # avoid zero denominator
        x = x / x.sum(dim=-1, keepdim=True)
        return x


class Decoder(nn.Module):
    def __init__(self, T: int = 5, NG: int = 3):
        super().__init__()
        self.T = T
        self.NG = NG
        self.latent_dim = T * (2 ** NG)
        self.flat_features = 10 * 20 * 20  # 4000
        self.fc1 = nn.Linear(self.latent_dim, 400)
        self.fc2 = nn.Linear(400, self.flat_features)
        # ConvTranspose to invert the encoder kernels
        self.deconv1 = nn.ConvTranspose2d(10, 10, kernel_size=5)  # 20 -> 24
        self.deconv2 = nn.ConvTranspose2d(10, 1, kernel_size=5)   # 24 -> 28

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T, 2**NG) or (B, latent_dim)
        if z.dim() == 3:
            z = z.flatten(start_dim=1)
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 10, 20, 20)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, T: int = 5, NG: int = 3):
        super().__init__()
        self.encoder = Encoder(T, NG)
        self.decoder = Decoder(T, NG)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
