"""Container shared by all experiments so one federated loop serves them all."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

TensorBatch = tuple[torch.Tensor, ...]


@dataclass
class Task:
    """One experiment: model, per-client data, loss and evaluation.

    Attributes
    ----------
    model : torch.nn.Module
        Global model; its parameters are what clients train and the server averages.
    client_data : list of tuple of torch.Tensor
        Training tensors held by each client (a single entry for centralised runs).
    loss : callable
        ``loss(model, batch) -> scalar tensor`` for a batch of ``client_data`` tensors.
    evaluate : callable
        ``evaluate(model) -> dict`` of held-out metrics, computed after every round.
    primary_metric : str
        Key of ``evaluate``'s output used as the headline number.
    final_report : callable
        ``final_report(model) -> dict`` with extra end-of-training metrics.
    summary : dict
        Dataset identity and sizes for ``run_status.json``.
    """

    model: nn.Module
    client_data: list[TensorBatch]
    loss: Callable[[nn.Module, TensorBatch], torch.Tensor]
    evaluate: Callable[[nn.Module], dict[str, float]]
    primary_metric: str
    final_report: Callable[[nn.Module], dict[str, Any]]
    summary: dict[str, Any] = field(default_factory=dict)


def tensor_batches(
    data: TensorBatch, batch_size: int, generator: torch.Generator
) -> Iterator[TensorBatch]:
    """Yield shuffled mini-batches covering ``data`` once (one local epoch)."""
    permutation = torch.randperm(len(data[0]), generator=generator)
    for start in range(0, len(permutation), batch_size):
        indices = permutation[start : start + batch_size]
        yield tuple(tensor[indices] for tensor in data)
