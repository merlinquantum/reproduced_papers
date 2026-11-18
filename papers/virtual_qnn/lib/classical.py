"""Classical baselines for the circle classification task (Fair Baseline Plan, C4).

- ``logistic``: logistic regression on the raw features ``(x1, x2)``. It is a
  linear floor; the circle is not linearly separable.
- ``mlp``: 2-4-1 tanh MLP (17 parameters, close to the 16 parameters of the
  two-branch depth-2 VQC).

Both use binary cross-entropy, full-batch Adam, and the same iteration
budget as the quantum classifier.
"""

from __future__ import annotations

import torch
from torch import nn


def build_classical_model(
    kind: str, generator: torch.Generator, dtype: torch.dtype
) -> nn.Module:
    """Return an untrained classical baseline producing logits.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"logistic"`` or ``"mlp"``.
    """
    if kind == "logistic":
        model = nn.Linear(2, 1)
    elif kind == "mlp":
        model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    else:
        raise ValueError(f"Unknown classical model kind {kind!r}")
    model = model.to(dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            bound = (
                1.0 / max(parameter.shape[-1], 1) ** 0.5 if parameter.ndim > 1 else 0.5
            )
            parameter.copy_(
                (torch.rand(parameter.shape, generator=generator, dtype=dtype) * 2 - 1)
                * bound
            )
    return model


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of points whose ``sigmoid(logit) >= 0.5`` matches the label."""
    predictions = (logits >= 0).to(labels.dtype)
    return float((predictions == labels).to(torch.float64).mean())


def split_train_validation(
    x: torch.Tensor, y: torch.Tensor, fraction: float, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministically hold out ``fraction`` of the training set for selection."""
    permutation = torch.randperm(len(x), generator=generator)
    n_validation = int(round(fraction * len(x)))
    validation_index, train_index = (
        permutation[:n_validation],
        permutation[n_validation:],
    )
    return x[train_index], y[train_index], x[validation_index], y[validation_index]
