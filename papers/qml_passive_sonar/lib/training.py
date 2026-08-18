"""Training and evaluation loops."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    epoch_time_s: float


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == targets).float().mean().item())


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total += images.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    if total == 0:
        return 0.0, 0.0, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return (
        total_loss / total,
        total_correct / total,
        torch.cat(all_preds),
        torch.cat(all_labels),
    )


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    weight_decay: float = 0.0,
) -> list[EpochMetrics]:
    """Run an Adam training loop and return per-epoch metrics."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history: list[EpochMetrics] = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_total += images.size(0)

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, device, loss_fn)
        history.append(
            EpochMetrics(
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                epoch_time_s=time.time() - t0,
            )
        )
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  test_loss=%.4f  test_acc=%.4f  (%.1fs)",
            epoch,
            epochs,
            train_loss,
            train_acc,
            test_loss,
            test_acc,
            history[-1].epoch_time_s,
        )
    return history


def confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for pred, label in zip(predictions.tolist(), labels.tolist()):
        matrix[label, pred] += 1
    return matrix
