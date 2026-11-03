"""Shared training utilities: optimiser construction and endless batch streams."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader


def build_optimizer(parameters, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Return the configured optimiser; the paper uses Adam for every model (App. B)."""
    name = training_cfg["optimizer"]
    if name != "adam":
        raise ValueError(
            f"Only the paper's 'adam' optimizer is supported, got {name!r}."
        )
    return torch.optim.Adam(parameters, lr=training_cfg["lr"])


def endless_batches(loader: DataLoader, device) -> Iterator[tuple[torch.Tensor, ...]]:
    """Yield batches forever, reshuffling at every pass over the data.

    The paper trains for a number of optimiser iterations, not epochs (App. B),
    so one iteration consumes one batch and the loader restarts when exhausted.
    """
    while True:
        for batch in loader:
            yield tuple(tensor.to(device) for tensor in batch)
