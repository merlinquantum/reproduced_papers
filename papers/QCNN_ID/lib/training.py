"""Shared training/evaluation loop for QCNN-ID models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    time_s: float


@dataclass
class TrainHistory:
    name: str
    epochs: list[EpochMetrics] = field(default_factory=list)
    final_accuracy: float = 0.0
    final_precision: float = 0.0
    final_recall: float = 0.0
    confusion_matrix: list[list[int]] | None = None
    param_count: int = 0
    train_time_s: float = 0.0
    roc_auc: float | None = None
    roc_curve: dict[str, list[float]] | None = None
    train_predictions: dict[str, list[float | int]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    test_predictions: dict[str, list[float | int]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "epochs": [vars(e) for e in self.epochs],
            "final_accuracy": self.final_accuracy,
            "final_precision": self.final_precision,
            "final_recall": self.final_recall,
            "confusion_matrix": self.confusion_matrix,
            "param_count": self.param_count,
            "train_time_s": self.train_time_s,
            "roc_auc": self.roc_auc,
            "roc_curve": self.roc_curve,
        }


def _confusion(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return cm


def _macro_precision_recall(cm: np.ndarray) -> tuple[float, float]:
    num_classes = cm.shape[0]
    precisions = []
    recalls = []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(precisions)), float(np.mean(recalls))


def _as_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    raise ValueError(
        "Binary QCNN-ID models must return one logit per sample; "
        f"got output shape {tuple(logits.shape)}."
    )


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    all_preds = []
    all_targets = []
    all_logits = []
    all_probs = []
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            targets = yb.to(device, dtype=torch.float32)
            logits = _as_binary_logits(model(xb))
            losses.append(loss_fn(logits, targets).item() * len(yb))
            probs = torch.sigmoid(logits)
            all_preds.append((probs >= 0.5).to(torch.int64).cpu().numpy())
            all_targets.append(targets.to(torch.int64).cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    n = sum(len(t) for t in all_targets)
    return (
        float(np.sum(losses) / max(n, 1)),
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_logits),
        np.concatenate(all_probs),
    )


def _binary_roc_from_probs(
    probs: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, dict[str, list[float]]] | tuple[None, None]:
    if len(np.unique(targets)) != 2:
        return None, None
    fpr, tpr, thresholds = roc_curve(targets, probs)
    return float(auc(fpr, tpr)), {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def _predictions_payload(
    logits: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
) -> dict[str, list[float | int]]:
    return {
        "sample_index": list(range(len(targets))),
        "y_true": targets.astype(int).tolist(),
        "y_pred": preds.astype(int).tolist(),
        "logit": logits.astype(float).tolist(),
        "probability": probs.astype(float).tolist(),
    }


def train_and_evaluate_model(
    name: str,
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    logger,
    num_classes: int = 2,
) -> TrainHistory:
    model = model.to(device)
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history = TrainHistory(
        name=name,
        param_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    overall_start = time.time()
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            logits = _as_binary_logits(model(xb))
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(yb)
            n_seen += len(yb)
        epoch_time = time.time() - t0
        train_loss = running_loss / max(n_seen, 1)
        test_loss, preds, targets, _logits, _probs = _evaluate(
            model, test_loader, device
        )
        cm = _confusion(preds, targets, num_classes)
        acc = float((preds == targets).mean())
        prec, rec = _macro_precision_recall(cm)
        history.epochs.append(
            EpochMetrics(
                loss=train_loss,
                accuracy=acc,
                precision=prec,
                recall=rec,
                time_s=epoch_time,
            )
        )
        logger.info(
            "[%s] epoch %d/%d | train_loss=%.4f | test_loss=%.4f | acc=%.4f"
            " | prec=%.4f | rec=%.4f | t=%.2fs",
            name,
            epoch + 1,
            epochs,
            train_loss,
            test_loss,
            acc,
            prec,
            rec,
            epoch_time,
        )

    test_loss, preds, targets, logits, probs = _evaluate(model, test_loader, device)
    cm = _confusion(preds, targets, num_classes)
    history.final_accuracy = float((preds == targets).mean())
    history.final_precision, history.final_recall = _macro_precision_recall(cm)
    history.confusion_matrix = cm.tolist()
    history.roc_auc, history.roc_curve = _binary_roc_from_probs(probs, targets)
    history.test_predictions = _predictions_payload(logits, probs, preds, targets)

    _, train_preds, train_targets, train_logits, train_probs = _evaluate(
        model, train_eval_loader, device
    )
    history.train_predictions = _predictions_payload(
        train_logits, train_probs, train_preds, train_targets
    )
    history.train_time_s = time.time() - overall_start
    return history
