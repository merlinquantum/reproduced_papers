"""Federated averaging loop (paper Sec. II-D, Algorithm 2) and its centralised control.

One communication round: the server picks ``clients_per_round`` clients, each starts
from the global parameters, trains locally for ``local_epochs`` epochs with a fresh
Adam optimiser, and returns its parameters. The server replaces the global
parameters with the sample-count-weighted average of the returned ones. Only trainable
parameters are averaged; buffers (for example MerLin's constant measurement mask) keep
the global model's values, so weighted floating-point averaging cannot perturb them.

``server_lr`` scales the step from the old global parameters to that average
(``global += server_lr * (average - global)``). The paper's Eq. (18) and Algorithm 2
disagree on the server update (LOG.md, "Reproducibility Assessment"), so plain FedAvg
(``server_lr = 1``) is the default and other values are a controlled variant.

The centralised control trains one model on the pooled data; one "round" is then one
epoch with a persistent optimiser, which keeps the round axis comparable.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

import torch

from .task import Task, tensor_batches

logger = logging.getLogger(__name__)


def _train_epochs(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    task: Task,
    data: tuple[torch.Tensor, ...],
    batch_size: int,
    epochs: int,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    """Run ``epochs`` local epochs; return the mean batch loss."""
    model.train()
    loss_sum, batches = 0.0, 0
    for _ in range(epochs):
        for batch in tensor_batches(data, batch_size, generator):
            batch = tuple(tensor.to(device) for tensor in batch)
            optimizer.zero_grad()
            loss = task.loss(model, batch)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
            logger.debug("TRAIN_STEP | batch=%d | loss=%.6f", batches, loss.item())
    return loss_sum / batches


def _weighted_average(
    states: list[dict[str, torch.Tensor]], weights: list[float], names: list[str]
) -> dict[str, torch.Tensor]:
    total = sum(weights)
    return {
        name: sum(w / total * state[name] for w, state in zip(weights, states))
        for name in names
    }


def run_training(
    cfg: dict[str, Any],
    task: Task,
    device: torch.device,
    on_round: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    """Train ``task.model`` for ``federation.rounds`` rounds.

    Parameters
    ----------
    cfg : dict
        Resolved configuration; uses ``federation.*`` and ``training.*``.
    task : Task
        Experiment container.
    device : torch.device
        Device for model and batches.
    on_round : callable
        Called with each round's history entry right after its evaluation.

    Returns
    -------
    list of dict
        One entry per round: ``round``, ``train_loss`` and the task's evaluation metrics.

    Raises
    ------
    ValueError
        If ``federation.mode`` or the client selection settings are invalid.
    """
    federation, training = cfg["federation"], cfg["training"]
    seed = int(cfg["seed"])
    rounds = int(federation["rounds"])
    batch_size = int(training["batch_size"])
    local_epochs = int(federation["local_epochs"])
    mode = federation["mode"]
    if mode not in ("federated", "centralized"):
        raise ValueError(f"Unknown federation.mode {mode!r}")
    n_clients = len(task.client_data)
    requested = federation["clients_per_round"]
    clients_per_round = n_clients if requested == "all" else int(requested)
    if mode == "federated" and not 1 <= clients_per_round <= n_clients:
        raise ValueError(
            f"federation.clients_per_round={clients_per_round} must be in 1..{n_clients}"
        )
    server_lr = float(federation["server_lr"])

    model = task.model
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["lr"]))
    history: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        if mode == "centralized":
            generator = torch.Generator().manual_seed(seed * 100003 + round_index)
            train_loss = _train_epochs(
                model, optimizer, task, task.client_data[0], batch_size,
                local_epochs, device, generator,
            )  # fmt: skip
        else:
            selector = torch.Generator().manual_seed(seed * 7919 + round_index)
            chosen = torch.randperm(n_clients, generator=selector)[:clients_per_round]
            global_state = copy.deepcopy(model.state_dict())
            states, weights, losses = [], [], []
            for client in chosen.tolist():
                local_model = copy.deepcopy(model)
                local_optimizer = torch.optim.Adam(
                    local_model.parameters(), lr=float(training["lr"])
                )
                generator = torch.Generator().manual_seed(
                    seed * 100003 + round_index * 1009 + client
                )
                data = task.client_data[client]
                losses.append(
                    _train_epochs(
                        local_model,
                        local_optimizer,
                        task,
                        data,
                        batch_size,
                        local_epochs,
                        device,
                        generator,
                    )  # fmt: skip
                )
                states.append(copy.deepcopy(local_model.state_dict()))
                weights.append(float(len(data[0])))
            trainable = [name for name, _ in model.named_parameters()]
            average = _weighted_average(states, weights, trainable)
            updated = dict(global_state)
            for name in trainable:
                updated[name] = global_state[name] + server_lr * (
                    average[name] - global_state[name]
                )
            model.load_state_dict(updated)
            train_loss = sum(
                w * client_loss for w, client_loss in zip(weights, losses)
            ) / sum(weights)
        model.eval()
        with torch.no_grad():
            metrics = task.evaluate(model)
        entry = {"round": round_index, "train_loss": train_loss, **metrics}
        history.append(entry)
        on_round(entry)
    return history
