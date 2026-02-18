# layer_classcal.py

import torch
import torch.nn as nn

from config import DTYPE


class BranchPyTorch(nn.Module):
    """
    Simple feed-forward network used as a branch in the PINN.

    Architecture:
        1 -> 16 -> 16 -> 1 with Tanh activations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16, dtype=DTYPE),
            nn.Tanh(),
            nn.Linear(16, 16, dtype=DTYPE),
            nn.Tanh(),
            nn.Linear(16, 1, dtype=DTYPE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
