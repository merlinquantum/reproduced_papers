"""CNN backbone and HQ-CNN classifier for the QML passive-sonar reproduction.

Architecture summary (paper Fig. 7):

    Block A: Conv(3 -> 96, 5x5, stride 2)  + BN + ReLU + MaxPool(3x3, s=2)
    Block B: Conv(96 -> 256, 5x5)          + BN + ReLU + MaxPool(3x3, s=2)
    Block C: Conv(256 -> 256, 5x5, p=2)    + BN + ReLU + MaxPool(3x3, s=2)
    Block D: Conv(256 -> 256, 3x3, p=1)    + BN + ReLU + MaxPool(3x3, s=2)
    Flatten -> FC -> ReLU -> FC -> ReLU -> output

The paper reports a 4096-dim FC bottleneck. We expose ``fc_dim`` so smoke
runs can fit in CPU RAM with a smaller value (e.g. 256).
"""

from __future__ import annotations

import math

import torch
from lib.quantum import PQC
from torch import nn


class _Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        # Skip the pool gracefully on tiny smoke inputs so 4 blocks still fit.
        if x.shape[-1] >= 3 and x.shape[-2] >= 3:
            x = self.pool(x)
        return x


class CNNBackbone(nn.Module):
    """Convolutional backbone producing a ``fc_dim``-dim embedding."""

    def __init__(self, in_channels: int = 3, fc_dim: int = 4096, image_size: int = 224) -> None:
        super().__init__()
        self.block_a = _Block(in_channels, 96, kernel=5, stride=2)
        self.block_b = _Block(96, 256, kernel=5)
        self.block_c = _Block(256, 256, kernel=5, padding=2)
        self.block_d = _Block(256, 256, kernel=3, padding=1)
        self.flat_dim = self._infer_flat_dim(in_channels, image_size)
        self.fc1 = nn.Linear(self.flat_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc_dim = fc_dim

    def _infer_flat_dim(self, in_channels: int, image_size: int) -> int:
        # Use eval mode + a 2-sample batch so BatchNorm's training-time
        # "need >1 sample per channel" check does not fire during shape probe.
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.zeros(2, in_channels, image_size, image_size)
            x = self.block_a(x)
            x = self.block_b(x)
            x = self.block_c(x)
            x = self.block_d(x)
            flat_dim = int(math.prod(x.shape[1:]))
        if was_training:
            self.train()
        return flat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = x.flatten(1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x


class CNNClassifier(nn.Module):
    """Pure-classical baseline: CNN backbone + linear head."""

    def __init__(self, num_classes: int, in_channels: int = 3, fc_dim: int = 4096, image_size: int = 224) -> None:
        super().__init__()
        self.backbone = CNNBackbone(in_channels, fc_dim, image_size)
        self.head = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class HQCNN(nn.Module):
    """Hybrid Quantum CNN classifier.

    Pipeline: ``f_post ∘ Q_θq ∘ f_enc ∘ f_CNN`` exactly as in the paper.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        fc_dim: int = 4096,
        image_size: int = 224,
        n_qubits: int = 10,
        n_layers: int = 4,
        pqc_init: str = "uniform",
    ) -> None:
        super().__init__()
        self.backbone = CNNBackbone(in_channels, fc_dim, image_size)
        self.encoder = nn.Linear(fc_dim, n_qubits)  # W_K, b_K
        self.pqc = PQC(n_qubits=n_qubits, n_layers=n_layers, init=pqc_init)
        self.head = nn.Linear(n_qubits, num_classes)  # W_out, b_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        # Encoding interface: phi = pi * sigmoid(W_K h + b_K) in [0, pi]
        encoded = torch.pi * torch.sigmoid(self.encoder(h))
        q = self.pqc(encoded)
        return self.head(q)
