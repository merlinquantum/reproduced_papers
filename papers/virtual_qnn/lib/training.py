"""Shared optimization loop: Adam + ReduceLROnPlateau, as in the reference code.

The paper only states "Adam". The authors' notebooks pair Adam with
``ReduceLROnPlateau(mode="min", factor=0.5)`` (PyTorch defaults otherwise:
patience 10, relative threshold 1e-4) and step the scheduler on the training
loss after every iteration. Every task trains full-batch, so one optimizer
iteration is the natural progress unit and is logged at INFO.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import torch

logger = logging.getLogger(__name__)

LossFunction = Callable[[], tuple[torch.Tensor, dict[str, float]]]


def optimize(
    parameters: Iterable[torch.nn.Parameter],
    loss_fn: LossFunction,
    training_cfg: dict,
) -> list[dict[str, float]]:
    """Minimize ``loss_fn`` and return the per-iteration history.

    Parameters
    ----------
    parameters : Iterable[torch.nn.Parameter]
        Trainable parameters.
    loss_fn : Callable[[], tuple[torch.Tensor, dict[str, float]]]
        Returns the differentiable loss and a dict of scalar diagnostics that
        are logged and stored alongside the loss.
    training_cfg : dict
        Requires ``iterations``, ``lr``, ``scheduler_factor`` and
        ``scheduler_patience``. ``scheduler_factor = None`` disables the
        scheduler (constant learning rate).

    Returns
    -------
    list[dict[str, float]]
        One record per iteration with ``iteration``, ``loss``, ``lr`` and the
        diagnostics. Values are measured *before* that iteration's update, as
        in the reference notebooks.
    """
    iterations = int(training_cfg["iterations"])
    parameters = list(parameters)
    optimizer = torch.optim.Adam(parameters, lr=float(training_cfg["lr"]))
    scheduler = None
    if training_cfg["scheduler_factor"] is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(training_cfg["scheduler_factor"]),
            patience=int(training_cfg["scheduler_patience"]),
        )
    logger.info(
        "TRAINING_STARTED | iterations=%d | optimizer=adam | lr=%.4g | "
        "scheduler=ReduceLROnPlateau(factor=%s, patience=%s) | n_parameters=%d",
        iterations,
        float(training_cfg["lr"]),
        training_cfg["scheduler_factor"],
        training_cfg["scheduler_patience"],
        sum(p.numel() for p in parameters),
    )
    history: list[dict[str, float]] = []
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad()
        loss, diagnostics = loss_fn()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        if scheduler is not None:
            scheduler.step(loss_value)
        learning_rate = optimizer.param_groups[0]["lr"]
        record = {"iteration": iteration, "loss": loss_value, "lr": learning_rate}
        record.update(diagnostics)
        history.append(record)
        details = " | ".join(f"{key}={value:.6g}" for key, value in diagnostics.items())
        logger.info(
            "TRAIN_ITERATION_COMPLETED | iteration=%d/%d | loss=%.8g | lr=%.3g%s",
            iteration,
            iterations,
            loss_value,
            learning_rate,
            f" | {details}" if details else "",
        )
    return history
