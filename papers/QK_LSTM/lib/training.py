"""Training and prediction loops (Sec. III-B4 and Table II).

The paper trains every model for 20 epochs with batch size 1, learning rate
1e-3 and the MSE loss of Eq. 14. It names "SGD or Adam"; the optimizer is a
config choice (LOG.md, Q4). The final-epoch model is evaluated: no checkpoint
is selected on validation or test data.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def build_optimizer(
    model: nn.Module, training_cfg: dict[str, Any]
) -> torch.optim.Optimizer:
    """Create the configured optimizer (``adam`` or ``sgd``)."""
    name = training_cfg["optimizer"].lower()
    lr = float(training_cfg["lr"])
    weight_decay = float(training_cfg["weight_decay"])
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}; expected 'adam' or 'sgd'")


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a split.

    Returns
    -------
    tuple of numpy.ndarray
        ``(predictions, target_days)``: scaled predictions ``(n,)`` and the
        integer target-day index of each window ``(n,)``.
    """
    model.eval()
    predictions, target_days = [], []
    for x, _, days in loader:
        predictions.append(model(x.to(device)).squeeze(-1).cpu().numpy())
        target_days.append(days.numpy())
    return np.concatenate(predictions), np.concatenate(target_days)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    training_cfg: dict[str, Any],
    device: torch.device,
    *,
    validation_loader: DataLoader | None = None,
    validation_rmse_fn=None,
) -> list[dict[str, float]]:
    """Train with MSE on the scaled target and log every epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train in place.
    train_loader : torch.utils.data.DataLoader
        Shuffled training windows.
    training_cfg : dict
        ``training`` config section: ``epochs``, ``optimizer``, ``lr``,
        ``weight_decay``.
    device : torch.device
        Device for model and tensors.
    validation_loader : torch.utils.data.DataLoader | None
        Optional validation windows, monitored each epoch (never used to pick
        a checkpoint). Default value is None.
    validation_rmse_fn : callable | None
        Maps ``(scaled_predictions, target_days)`` to validation RMSE in AQI
        units. Required with ``validation_loader``. Default value is None.

    Returns
    -------
    list of dict
        One record per epoch: ``epoch``, ``train_mse`` (scaled units),
        ``validation_rmse_aqi`` (or None) and ``seconds``.
    """
    epochs = int(training_cfg["epochs"])
    optimizer = build_optimizer(model, training_cfg)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float]] = []
    logger.info(
        "TRAINING_STARTED | epochs=%d | optimizer=%s | lr=%.6g | train_windows=%d | batch_size=%d",
        epochs,
        training_cfg["optimizer"],
        float(training_cfg["lr"]),
        len(train_loader.dataset),
        train_loader.batch_size,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        total_loss, total_items = 0.0, 0
        for batch_index, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}, batch {batch_index}"
                )
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            total_loss += loss_value * x.shape[0]
            total_items += x.shape[0]
            logger.debug(
                "TRAIN_BATCH | epoch=%d | batch=%d | loss=%.6f",
                epoch,
                batch_index,
                loss_value,
            )
        record = {
            "epoch": epoch,
            "train_mse": total_loss / total_items,
            "validation_rmse_aqi": None,
            "seconds": time.perf_counter() - started,
        }
        if validation_loader is not None and len(validation_loader.dataset) > 0:
            predictions, days = predict(model, validation_loader, device)
            record["validation_rmse_aqi"] = float(validation_rmse_fn(predictions, days))
        history.append(record)
        logger.info(
            "TRAIN_EPOCH_COMPLETED | epoch=%d/%d | train_mse=%.6f | validation_rmse_aqi=%s | seconds=%.1f",
            epoch,
            epochs,
            record["train_mse"],
            "n/a"
            if record["validation_rmse_aqi"] is None
            else f"{record['validation_rmse_aqi']:.4f}",
            record["seconds"],
        )
    return history
