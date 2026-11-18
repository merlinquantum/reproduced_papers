"""Shared mini-batch training loop for the binary direction classifiers (Tasks 1 and 2)."""

from __future__ import annotations

import copy
import logging
import math

import numpy as np
import torch
from torch import nn

from .metrics import safe_auc

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_proba(
    model: nn.Module, inputs: torch.Tensor, batch_size: int = 1024
) -> np.ndarray:
    """Probability of an up move for every row of ``inputs``."""
    model.eval()
    outputs = [
        torch.sigmoid(model(inputs[start : start + batch_size]))
        for start in range(0, len(inputs), batch_size)
    ]
    return torch.cat(outputs).double().cpu().numpy()


def fit_classifier(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    early_stop_inputs: torch.Tensor,
    early_stop_labels: torch.Tensor,
    training_cfg: dict,
    seed: int,
    context: str,
) -> dict:
    """Train with Adam on BCE and keep the weights with the best early-stop AUC.

    Training stops when the early-stop ROC-AUC has not improved for
    ``patience`` epochs (§III-D5 uses patience 30 for Task 2; decision R4 uses
    the same rule for Task 1) or after ``max_epochs``.

    Parameters
    ----------
    model : torch.nn.Module
        Classifier returning one logit per row; trained in place.
    train_inputs, train_labels : torch.Tensor
        Training rows and ``{0, 1}`` labels.
    early_stop_inputs, early_stop_labels : torch.Tensor
        Chronologically later rows used only for the stopping rule.
    training_cfg : dict
        Keys ``lr``, ``batch_size``, ``max_epochs``, ``patience`` and
        ``class_weighting`` (``"none"`` or ``"balanced"``; balanced sets
        ``pos_weight = n_negative / n_positive``, the weighted BCE of §III-D3).
    seed : int
        Seed for the batch shuffling order.
    context : str
        Short label (candidate and fold) prefixed to every epoch log line.

    Returns
    -------
    dict
        ``best_epoch``, ``best_early_stop_auc``, ``epochs_run`` and the final
        ``train_loss``.
    """
    lr = float(training_cfg["lr"])
    batch_size = int(training_cfg["batch_size"])
    max_epochs = int(training_cfg["max_epochs"])
    patience = int(training_cfg["patience"])
    weighting = training_cfg["class_weighting"]
    if weighting == "balanced":
        positives = float(train_labels.sum())
        pos_weight = torch.tensor(
            (len(train_labels) - positives) / max(positives, 1.0),
            dtype=train_inputs.dtype,
        )
    elif weighting == "none":
        pos_weight = None
    else:
        raise ValueError(
            f"class_weighting must be 'none' or 'balanced', got {weighting!r}"
        )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    shuffle_generator = torch.Generator().manual_seed(seed)

    best_auc, best_epoch, best_state = -math.inf, 0, copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    epoch_loss = math.nan
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_inputs), generator=shuffle_generator)
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(train_inputs[batch]), train_labels[batch])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        epoch_loss = loss_sum / len(order)
        early_stop_auc = safe_auc(
            early_stop_labels.cpu().numpy(), predict_proba(model, early_stop_inputs)
        )
        improved = np.isfinite(early_stop_auc) and early_stop_auc > best_auc
        if improved:
            best_auc, best_epoch, best_state = (
                early_stop_auc,
                epoch,
                copy.deepcopy(model.state_dict()),
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        logger.info(
            "TRAIN_EPOCH_COMPLETED | %s | epoch=%d/%d | loss=%.6f | early_stop_auc=%.6f | best_epoch=%d",
            context,
            epoch,
            max_epochs,
            epoch_loss,
            early_stop_auc,
            best_epoch,
        )
        if epochs_without_improvement >= patience:
            break
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_early_stop_auc": best_auc,
        "epochs_run": epoch,
        "train_loss": epoch_loss,
    }
