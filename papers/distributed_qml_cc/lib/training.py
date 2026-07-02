"""Training loop for the DQML models.

The paper trains with Adam (lr=0.05), batch size 512 (= entire training
batch per iteration), and a least-squares loss

    L ~ sum_n ( f_label_n - f_int_n )^2

over the dataset, with binary labels in ``{-1, +1}``. Predictions assign
class +1 when ``f_int >= 0`` and -1 otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn


@dataclass
class TrainingResult:
    iterations: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    final_val_acc: float = 0.0
    final_train_acc: float = 0.0
    final_loss: float = 0.0


def _accuracy(fint: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.where(fint >= 0.0, torch.ones_like(fint), -torch.ones_like(fint))
    return (pred == y).float().mean().item()


def train_dqml(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    n_iterations: int = 1000,
    lr: float = 0.05,
    batch_size: int = 512,
    eval_every: int = 25,
    seed: int = 42,
    progress_callback: Callable[[int, dict], None] | None = None,
) -> TrainingResult:
    """Run a full Adam training schedule and return per-iteration metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = x_train.shape[0]
    result = TrainingResult()

    for it in range(1, n_iterations + 1):
        idx = torch.randint(0, n_train, (batch_size,), generator=torch.Generator().manual_seed(seed + it))
        xb = x_train[idx]
        yb = y_train[idx]
        fint = model(xb)
        loss = ((yb - fint) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

        if it % eval_every == 0 or it == n_iterations:
            model.eval()
            with torch.no_grad():
                val_fint = model(x_val)
                val_acc = _accuracy(val_fint, y_val)
                train_fint = model(x_train)
                train_acc = _accuracy(train_fint, y_train)
            model.train()
            result.iterations.append(it)
            result.train_loss.append(float(loss.item()))
            result.train_acc.append(train_acc)
            result.val_acc.append(val_acc)
            if progress_callback is not None:
                progress_callback(it, {
                    "loss": float(loss.item()),
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                })

    result.final_val_acc = result.val_acc[-1] if result.val_acc else 0.0
    result.final_train_acc = result.train_acc[-1] if result.train_acc else 0.0
    result.final_loss = result.train_loss[-1] if result.train_loss else 0.0
    return result


__all__ = ["TrainingResult", "train_dqml"]
