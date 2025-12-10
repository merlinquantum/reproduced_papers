from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: Callable, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch_x.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    run_dir: Path,
) -> dict:
    device = torch.device(cfg.get("device", "cpu"))
    model.to(device)

    training_cfg = cfg.get("training", {})
    epochs = int(training_cfg.get("epochs", 1))
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    grad_clip = training_cfg.get("clip_grad_norm")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        val_loss = evaluate(model, val_loader, criterion, device)
        LOGGER.info("Epoch %d/%d - train_loss=%.4f - val_loss=%.4f", epoch, epochs, train_loss, val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    metrics = {
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "history": history,
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved metrics to %s", metrics_path)
    return metrics


__all__ = ["fit", "train_one_epoch", "evaluate"]
