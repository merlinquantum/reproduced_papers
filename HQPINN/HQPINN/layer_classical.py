"""
Classical branch used in HQPINN models.

In the paper's hybrid architecture, each model is built from one or more
parallel branches whose outputs are combined downstream. This module provides
the pure-MLP branch used in:
- fully classical baselines (CC),
- hybrid classical+quantum variants (e.g., CI/CP),
- shared readout setups for DHO/SEE/DEE experiments.
"""

import torch
import torch.nn as nn

from .config import DTYPE, DHO_NUM_HIDDEN_LAYERS, DHO_HIDDEN_WIDTH


class BranchPyTorch(nn.Module):
    """
    High-level classical encoder from input coordinates to physical outputs.

    This branch is the "classical path" counterpart to quantum branches
    (PennyLane/Merlin) in the paper's comparative experiments.
    """

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        num_hidden_layers: int = DHO_NUM_HIDDEN_LAYERS,
        hidden_width: int = DHO_HIDDEN_WIDTH,
    ) -> None:
        super().__init__()

        layers = []

        layers.append(nn.Linear(in_features, hidden_width, dtype=DTYPE))
        layers.append(nn.Tanh())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width, dtype=DTYPE))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_width, out_features, dtype=DTYPE))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
