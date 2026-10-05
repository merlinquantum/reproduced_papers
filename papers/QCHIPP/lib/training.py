"""Fixed-budget full-batch training and batched held-out evaluation."""

from __future__ import annotations

import logging
import time

import torch

from .gate import corrected_objective, source_objective
from .models import PHOTONIC_MODELS, build_trainable_model, config_dtype

LOGGER = logging.getLogger(__name__)

SOURCE_LABEL_MODES = ("signed", "zero_one")


def _zero_one_source_objective(scores, targets01):
    # Release-default labels: negatives (y=0) contribute exactly zero loss.
    return (-targets01 * torch.log2(scores.clamp_min(1e-10))).mean()


def _spsa_step(model, objective, bits, targets, epoch, training_cfg):
    """One SPSA update with the release's gain schedule (a/k^0.602, c/k^0.101)."""
    gain = float(training_cfg["spsa_a"]) / epoch**0.602
    perturbation = float(training_cfg["spsa_c"]) / epoch**0.101
    # Two objective evaluations per update; no QML gradient surrogate.
    with torch.no_grad():
        origin = model.weights.clone()
        delta = torch.randint(0, 2, origin.shape).to(origin) * 2 - 1
        model.weights.copy_(origin + perturbation * delta)
        plus = objective(model(bits), targets)
        model.weights.copy_(origin - perturbation * delta)
        minus = objective(model(bits), targets)
        model.weights.copy_(origin - gain * (plus - minus) / (2 * perturbation) * delta)
        return objective(model(bits), targets)


def train_model(cfg: dict, bits: torch.Tensor, targets: torch.Tensor):
    """Train for a fixed budget and keep the last checkpoint.

    No checkpoint is selected on test metrics. ``gate_source`` uses SPSA on the
    source objective; every other model uses full-batch Adam.

    Parameters
    ----------
    cfg : dict
        Needs ``model.name``, ``seed``, ``dtype``, ``training.epochs`` and
        ``training.lr``. ``gate_source`` also needs ``model.source_labels``
        and ``training.spsa_a`` / ``training.spsa_c``.
    bits : torch.Tensor
        Full released training split, shape (B, 20).
    targets : torch.Tensor
        Binary labels, shape (B,).

    Returns
    -------
    tuple
        Model and a record with losses, parameter counts, weight change,
        optimizer and objective names.
    """
    name = cfg["model"]["name"]
    dtype = config_dtype(cfg)
    epochs = int(cfg["training"]["epochs"])
    if epochs < 1:
        raise ValueError("Training epochs must be positive")
    model = build_trainable_model(cfg)
    total = sum(p.numel() for p in model.parameters())
    if name.startswith("gate_"):
        quantum = total
    else:
        quantum = sum(
            p.numel() for n, p in model.named_parameters() if ".quantum." in n
        )
    initial = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

    if name == "gate_source":
        label_mode = cfg["model"]["source_labels"]
        if label_mode not in SOURCE_LABEL_MODES:
            raise ValueError(f"source_labels must be one of {SOURCE_LABEL_MODES}")
        objective = (
            source_objective if label_mode == "signed" else _zero_one_source_objective
        )
        optimizer_name = "SPSA"
        objective_name = f"source {label_mode} clipped scalar log2"
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(cfg["training"]["lr"])
        )
        optimizer_name = "Adam"
        if name == "gate_corrected":
            objective = corrected_objective
            objective_name = "corrected physical-probability Bernoulli BCE in bits"
        else:
            bce = torch.nn.BCEWithLogitsLoss()

            def objective(output, labels):
                return bce(output, labels.to(dtype))

            objective_name = "Bernoulli BCE in nats"

    history = []
    start = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        if name == "gate_source":
            loss = _spsa_step(model, objective, bits, targets, epoch, cfg["training"])
        else:
            optimizer.zero_grad()
            loss = objective(model(bits), targets)
            loss.backward()
            optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        history.append(float(loss.detach()))
        LOGGER.info(
            "TRAIN_EPOCH_COMPLETED | epoch=%d/%d | loss=%.8f | optimizer=%s | full_batch=%d",
            epoch,
            epochs,
            history[-1],
            optimizer_name,
            len(targets),
        )
    weight_change = torch.linalg.vector_norm(
        torch.nn.utils.parameters_to_vector(model.parameters()).detach() - initial
    ).item()
    return model, {
        "losses": history,
        "parameters": {
            "quantum": quantum,
            "classical": total - quantum,
            "total": total,
        },
        "weight_change_l2": weight_change,
        "train_seconds": time.perf_counter() - start,
        "updates": epochs,
        "optimizer": optimizer_name,
        "objective": objective_name,
        "batch": "full released training split",
    }


def predict_probabilities(
    model, cfg: dict, bits: torch.Tensor, shots: int = 0
) -> torch.Tensor:
    """Return positive-class probabilities in eval mode without gradients.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    cfg : dict
        Needs ``model.name`` and ``evaluation.batch_size``.
    bits : torch.Tensor
        Inputs, shape (B, 20).
    shots : int
        Shots per optical cell; 0 is analytic. Only valid for MerLin models.
        Default value is 0.

    Returns
    -------
    torch.Tensor
        Probabilities of shape (B,), on CPU.
    """
    model.eval()
    name = cfg["model"]["name"]
    if shots and name not in PHOTONIC_MODELS:
        raise ValueError("Finite optical shots apply only to MerLin models")
    batch_size = int(cfg["evaluation"]["batch_size"])
    results = []
    with torch.no_grad():
        for batch in bits.split(batch_size):
            if name.startswith("gate_"):
                results.append(model.probabilities(batch).clamp(0, 1))
            elif name in PHOTONIC_MODELS:
                results.append(torch.sigmoid(model(batch, shots=shots)))
            else:
                results.append(torch.sigmoid(model(batch)))
    return torch.cat(results).detach().cpu()
