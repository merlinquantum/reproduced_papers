"""MNIST classification tasks (experiments E1 and E2) for the VQTN and classical models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn import metrics as skm
from torch import nn

from .classical import ClassicalClassifier
from .data import build_mnist_splits
from .task import Task, TensorBatch

EVALUATION_BATCH = 2000


def _build_model(cfg: dict[str, Any], n_classes: int, dtype: torch.dtype) -> nn.Module:
    model_cfg = cfg["model"]
    if model_cfg["name"] == "vqtn":
        from .vqtn import VQTNClassifier

        return VQTNClassifier(
            n_classes,
            float(model_cfg["init_std"]),
            dtype,
            int(model_cfg["layers"]),
            float(model_cfg["encoding_scale"]),
            model_cfg["encoding_gate"],
            model_cfg["readout"],
        )
    if model_cfg["name"] == "merlin":
        from .vqtn_merlin import PhotonicClassifier

        return PhotonicClassifier(
            n_classes,
            int(model_cfg["n_photons"]),
            float(model_cfg["encoding_scale"]),
            model_cfg["superposition"],
            model_cfg["photonic_readout"],
            model_cfg["computation_space"],
            dtype,
        )
    if model_cfg["name"] == "classical":
        return ClassicalClassifier(n_classes, int(model_cfg["hidden"]), dtype)
    raise ValueError(f"Unknown model.name {model_cfg['name']!r} for an MNIST task")


def _log_probabilities(
    model: nn.Module, pixels: torch.Tensor, device: torch.device
) -> np.ndarray:
    chunks = [
        model(pixels[start : start + EVALUATION_BATCH].to(device)).cpu()
        for start in range(0, len(pixels), EVALUATION_BATCH)
    ]
    return torch.cat(chunks).numpy()


def _split_metrics(
    model: nn.Module, split: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> dict[str, float]:
    pixels, labels = split
    log_probs = _log_probabilities(model, pixels, device)
    truth = labels.numpy()
    predicted = log_probs.argmax(axis=1)
    probs = np.exp(log_probs)
    if probs.shape[1] == 2:
        auc = skm.roc_auc_score(truth, probs[:, 1])
    else:
        auc = skm.roc_auc_score(truth, probs, multi_class="ovr", average="macro")
    return {
        "loss": float(functional.nll_loss(torch.from_numpy(log_probs), labels)),
        "accuracy": float((predicted == truth).mean()),
        "f1": float(skm.f1_score(truth, predicted, average="macro")),
        "auc": float(auc),
    }


def build_mnist_task(
    cfg: dict[str, Any], dtype: torch.dtype, device: torch.device
) -> Task:
    """Build the data, model, loss and evaluation of an MNIST experiment.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (``dataset``, ``model``, ``federation``).
    dtype : torch.dtype
        Real dtype for pixels and parameters.
    device : torch.device
        Device used for evaluation batches.

    Returns
    -------
    Task
        Ready to be passed to :func:`lib.federated.run_training`. ``evaluate`` reports
        ``test_*`` metrics and, when a validation split exists, ``validation_*``.
    """
    splits = build_mnist_splits(cfg, dtype)
    n_classes = len(cfg["dataset"]["classes"])
    model = _build_model(cfg, n_classes, dtype).to(device)
    train_pixels, train_labels = splits["train"]
    client_data = [
        (train_pixels[indices], train_labels[indices])
        for indices in splits["client_indices"]
    ]

    def loss(model: nn.Module, batch: TensorBatch) -> torch.Tensor:
        pixels, labels = batch
        return functional.nll_loss(model(pixels), labels)

    def evaluate(model: nn.Module) -> dict[str, float]:
        result = {
            f"test_{name}": value
            for name, value in _split_metrics(model, splits["test"], device).items()
        }
        if "validation" in splits:
            result.update(
                {
                    f"validation_{name}": value
                    for name, value in _split_metrics(
                        model, splits["validation"], device
                    ).items()
                }
            )
        return result

    def final_report(model: nn.Module) -> dict[str, Any]:
        pixels, labels = splits["test"]
        truth = labels.numpy()
        with torch.no_grad():
            predicted = _log_probabilities(model, pixels, device).argmax(axis=1)
        precision, recall, f1, _ = skm.precision_recall_fscore_support(
            truth, predicted, labels=list(range(n_classes)), zero_division=0
        )
        confusion = skm.confusion_matrix(
            truth, predicted, labels=list(range(n_classes)), normalize="true"
        )
        return {
            "train_metrics": _split_metrics(model, splits["train"], device),
            "classes": splits["summary"]["classes"],
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "normalized_confusion_matrix": confusion.tolist(),
        }

    return Task(
        model=model,
        client_data=client_data,
        loss=loss,
        evaluate=evaluate,
        primary_metric="test_accuracy",
        final_report=final_report,
        summary=splits["summary"],
    )
