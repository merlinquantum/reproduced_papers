from __future__ import annotations

import torch
from torch import nn


class RNNRegressor(nn.Module):
    """Simple RNN baseline for univariate or multivariate forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 64,
        layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(x)
        last_hidden = outputs[:, -1]
        prediction = self.head(last_hidden).squeeze(-1)
        return prediction


__all__ = ["RNNRegressor"]
