# layer_classcal.py

import torch
import torch.nn as nn

from .config import DTYPE, DHO_NUM_HIDDEN_LAYERS, DHO_HIDDEN_WIDTH


class BranchPyTorch(nn.Module):
    """
    Simple feed-forward network used as a branch in the PINN.
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
