"""Classical FNN head mapping the VQC's scalar expectation value to a fraud
probability.

Architecture: ``1 -> hidden_dim -> 1`` (ReLU hidden, sigmoid output).
"""

from __future__ import annotations

import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, vqc_expval: torch.Tensor) -> torch.Tensor:
        # vqc_expval: (batch,) -> (batch, 1) for the Linear(1, ...) input.
        x = vqc_expval.unsqueeze(-1) if vqc_expval.dim() == 1 else vqc_expval
        p_hat = self.net(x)
        return p_hat.squeeze(-1)


__all__ = ["Head"]
