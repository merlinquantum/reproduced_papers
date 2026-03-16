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
        cell_type: str = "rnn",
        bidirectional: bool = False,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        cell_type_norm = str(cell_type).strip().lower()
        if cell_type_norm not in {"rnn", "gru", "lstm"}:
            raise ValueError(
                f"Unsupported cell_type={cell_type!r} (expected 'rnn', 'gru', or 'lstm')"
            )

        rnn_dropout = dropout if layers > 1 else 0.0
        rnn_cls: type[nn.Module]
        if cell_type_norm == "gru":
            rnn_cls = nn.GRU
        elif cell_type_norm == "lstm":
            rnn_cls = nn.LSTM
        else:
            rnn_cls = nn.RNN

        self.cell_type = cell_type_norm
        self.bidirectional = bool(bidirectional)
        self.input_dropout = nn.Dropout(float(input_dropout))
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=self.bidirectional,
        )
        head_in = hidden_dim * (2 if self.bidirectional else 1)
        self.head = nn.Linear(head_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        outputs, _ = self.rnn(x)
        last_hidden = outputs[:, -1]
        prediction = self.head(last_hidden).squeeze(-1)
        return prediction


__all__ = ["RNNRegressor"]
