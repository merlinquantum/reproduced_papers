"""Trainable feed-forward head that maps reservoir probabilities to feature logits."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ReservoirHead(nn.Module):
    """Single (or two-layer) linear readout, trained with cross-entropy."""

    def __init__(self, input_dim: int, num_features: int, hidden_dim: int = 0):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_features),
            )
        else:
            self.net = nn.Linear(input_dim, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_head(
    head: ReservoirHead,
    inputs: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    verbose: bool = False,
) -> list[float]:
    """Fit ``head`` so that ``softmax(head(inputs)) ~ one_hot(targets)``.

    ``inputs`` has shape ``(T, input_dim)``; ``targets`` has shape ``(T,)`` with
    integer class indices. The full sequence is used as a single batch -
    matching the small datasets in the paper (157 features for Mario 1-2).
    """
    head.train()
    optim = torch.optim.Adam(
        head.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    loss_fn = nn.CrossEntropyLoss()
    x = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.long)
    history: list[float] = []
    for epoch in range(int(epochs)):
        optim.zero_grad()
        logits = head(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        history.append(float(loss.item()))
        if verbose:
            print(f"epoch {epoch:03d}: loss={loss.item():.4f}")
    return history
