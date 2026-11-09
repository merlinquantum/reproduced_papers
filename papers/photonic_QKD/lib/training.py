"""Teacher training, PQKD alternating optimisation (SPSA + Adam) and evaluation.

Student loop (Supp. Algorithm S1), per epoch:
1. photonic phase: ``spsa.updates_per_epoch`` SPSA steps on the conditioning
   parameters with the student weights fixed (Supp. Algorithm S2);
2. draw one fresh conditioning vector ``z`` (``S`` shots), optionally smooth it
   with an EMA (Supp. Eq. S-EMA), and load it into every compressed layer;
3. one epoch of Adam on the KD loss with ``z`` fixed.

The SPSA objective ``J`` is the KD loss on a fixed set of validation batches
(the paper allows "negative validation accuracy or equivalently the validation
distillation loss"). It is evaluated with an instantaneous feature at
``theta +- c * Delta``, not the EMA, so that ``J`` responds to ``theta``.

MerLin-native sources (Phase 4 adaptation, ``source.differentiable``) skip
SPSA: their exact feature is recomputed on every batch and its parameters are
trained by Adam together with the student through autograd.

Student training writes ``checkpoint.pt`` after every epoch so a stopped run
can resume from its last completed epoch.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch.utils.data import TensorDataset

from .conditioning import ConditioningSource
from .data import TensorBatchLoader
from .models import ThreeBlockCNN

logger = logging.getLogger(__name__)


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    hard_label_weight: float,
) -> torch.Tensor:
    """Paper Eq. (2): ``lambda*CE(y, p_S) + (1-lambda)*tau^2*KL(p_T^tau || p_S^tau)``."""
    hard = functional.cross_entropy(student_logits, labels)
    soft = functional.kl_div(
        functional.log_softmax(student_logits / temperature, dim=1),
        functional.log_softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
        log_target=True,
    )
    return hard_label_weight * hard + (1.0 - hard_label_weight) * temperature**2 * soft


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: TensorBatchLoader, device: torch.device
) -> dict[str, float]:
    """Top-1 accuracy (%) and mean cross-entropy with the model in eval mode."""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        loss_sum += functional.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return {"accuracy": 100.0 * correct / total, "cross_entropy": loss_sum / total}


@torch.no_grad()
def teacher_logits_for(
    teacher: torch.nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Teacher logits for every sample (eval mode, so dropout is off and they are fixed)."""
    teacher.eval()
    images = dataset.tensors[0]
    outputs = [
        teacher(images[start : start + batch_size].to(device)).cpu()
        for start in range(0, images.shape[0], batch_size)
    ]
    return torch.cat(outputs)


@dataclass
class TeacherResult:
    """Teacher training outcome."""

    history: list[dict[str, float]]
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    best_val_accuracy: float


def train_teacher(
    model: ThreeBlockCNN,
    train_loader: TensorBatchLoader,
    val_loader: TensorBatchLoader,
    training_cfg: dict[str, Any],
    device: torch.device,
) -> TeacherResult:
    """Train the teacher with cross-entropy and Adam (Supp. Note 8).

    The checkpoint with the best validation accuracy is kept, as in the paper
    ("best-performing checkpoint is selected using validation accuracy").
    """
    epochs = int(training_cfg["epochs"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["lr"]))
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_val_accuracy = 0, -1.0
    logger.info(
        "TRAINING_STARTED | model=teacher | epochs=%d | optimizer=adam | lr=%g",
        epochs,
        training_cfg["lr"],
    )
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
        validation = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_cross_entropy": loss_sum / total,
            "train_accuracy": 100.0 * correct / total,
            "val_accuracy": validation["accuracy"],
            "val_cross_entropy": validation["cross_entropy"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        if validation["accuracy"] > best_val_accuracy:
            best_val_accuracy = validation["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        logger.info(
            "TRAIN_EPOCH_COMPLETED | model=teacher | epoch=%d/%d | train_ce=%.5f | train_acc=%.3f | val_ce=%.5f | val_acc=%.3f | seconds=%.1f",
            epoch,
            epochs,
            record["train_cross_entropy"],
            record["train_accuracy"],
            record["val_cross_entropy"],
            record["val_accuracy"],
            record["seconds"],
        )
    return TeacherResult(history, best_state, best_epoch, best_val_accuracy)


class SPSA:
    """Simultaneous perturbation stochastic approximation (Supp. Algorithm S2).

    Parameters
    ----------
    step_size : float
        ``a``.
    perturbation : float
        ``c``.
    bound : float
        Elementwise clip ``theta_max`` applied to perturbed and updated points.
    generator : torch.Generator
        Draws the Rademacher directions.
    """

    def __init__(
        self,
        step_size: float,
        perturbation: float,
        bound: float,
        generator: torch.Generator,
    ) -> None:
        self.step_size = step_size
        self.perturbation = perturbation
        self.bound = bound
        self.generator = generator

    def step(
        self, vector: torch.Tensor, objective
    ) -> tuple[torch.Tensor, float, float]:
        """One SPSA update; returns the new vector and ``(J+, J-)``."""
        direction = (
            torch.randint(0, 2, vector.shape, generator=self.generator).to(vector) * 2
            - 1
        )
        plus = (vector + self.perturbation * direction).clamp(-self.bound, self.bound)
        minus = (vector - self.perturbation * direction).clamp(-self.bound, self.bound)
        objective_plus = objective(plus)
        objective_minus = objective(minus)
        gradient = (
            (objective_plus - objective_minus) / (2 * self.perturbation) * direction
        )
        updated = (vector - self.step_size * gradient).clamp(-self.bound, self.bound)
        return updated, objective_plus, objective_minus


@dataclass
class StudentResult:
    """Student training outcome."""

    history: list[dict[str, float]]
    feature_used: torch.Tensor | None
    feature_trace_raw: list[torch.Tensor]
    feature_trace_used: list[torch.Tensor]


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically write a training checkpoint (temporary file, then rename)."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`."""
    return torch.load(path, map_location="cpu", weights_only=False)


def train_student(
    model: ThreeBlockCNN,
    source: ConditioningSource | None,
    train_loader: TensorBatchLoader,
    val_loader: TensorBatchLoader,
    val_proxy_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],
    device: torch.device,
    generator: torch.Generator,
    checkpoint_path: Path | None = None,
    resume_state: dict[str, Any] | None = None,
) -> StudentResult:
    """Train a student by KD; alternate SPSA updates when the source is learnable.

    Parameters
    ----------
    model : ThreeBlockCNN
        Student. Feature-mixed layers receive ``z`` from ``source``.
    source : ConditioningSource or None
        Conditioning source; ``None`` for students without feature mixing
        (dense KD-only and dictionary-only baselines).
    train_loader : TensorBatchLoader
        Yields ``(images, labels, teacher_logits)``.
    val_loader : TensorBatchLoader
        Yields ``(images, labels)`` for per-epoch validation.
    val_proxy_batches : list of tuple
        Fixed ``(images, labels, teacher_logits)`` batches for the SPSA
        objective ``J``.
    cfg : dict
        Resolved configuration (``training``, ``spsa``, ``conditioning``).
    device : torch.device
        Device for the student.
    generator : torch.Generator
        Drives shot sampling, random features and SPSA directions.
    checkpoint_path : pathlib.Path, optional
        Where to write the per-epoch checkpoint. Default value is None (no
        checkpointing).
    resume_state : dict, optional
        A checkpoint from :func:`load_checkpoint`; training continues after its
        last completed epoch. Default value is None.
    """
    training_cfg = cfg["training"]
    spsa_cfg = cfg["spsa"]
    ema_cfg = cfg["conditioning"]["ema"]
    epochs = int(training_cfg["epochs"])
    temperature = float(training_cfg["temperature"])
    hard_weight = float(training_cfg["hard_label_weight"])
    uses_feature = model.uses_feature
    if uses_feature and source is None:
        raise ValueError("Student has feature-mixed layers but no conditioning source")
    differentiable = uses_feature and source is not None and source.differentiable
    ema_enabled = bool(ema_cfg["enabled"]) and uses_feature
    if differentiable and ema_enabled:
        raise ValueError(
            "EMA is undefined for a differentiable exact feature; set conditioning.ema.enabled=false"
        )
    ema_beta = float(ema_cfg["beta"])
    learnable = uses_feature and source is not None and source.learnable
    updates_per_epoch = int(spsa_cfg["updates_per_epoch"]) if learnable else 0
    spsa = SPSA(
        float(spsa_cfg["a"]),
        float(spsa_cfg["c"]),
        float(spsa_cfg["theta_max"]),
        generator,
    )
    parameter_groups = [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": float(training_cfg["lr"]),
        }
    ]
    if differentiable:
        parameter_groups.append(
            {
                "params": list(source.parameters()),
                "lr": float(cfg["conditioning"]["native_lr"]),
            }
        )
    optimizer = torch.optim.Adam(parameter_groups)

    @torch.no_grad()
    def proxy_objective(feature: torch.Tensor | None) -> float:
        if feature is not None:
            model.set_feature(feature.to(device))
        model.eval()
        total, count = 0.0, 0
        for images, labels, teacher_logits in val_proxy_batches:
            logits = model(images.to(device))
            loss = kd_loss(
                logits,
                teacher_logits.to(device),
                labels.to(device),
                temperature,
                hard_weight,
            )
            total += loss.item() * labels.numel()
            count += labels.numel()
        return total / count

    def objective_at(vector: torch.Tensor) -> float:
        saved = source.vector()
        source.set_vector(vector)
        value = proxy_objective(source.feature(generator))
        source.set_vector(saved)
        return value

    history: list[dict[str, float]] = []
    trace_raw: list[torch.Tensor] = []
    trace_used: list[torch.Tensor] = []
    feature_used: torch.Tensor | None = None
    first_epoch = 1
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        if source is not None:
            source.load_checkpoint_state(resume_state["source"])
        feature_used = resume_state["feature_used"]
        history = resume_state["history"]
        trace_raw = resume_state["trace_raw"]
        trace_used = resume_state["trace_used"]
        generator.set_state(resume_state["rng"]["sampling"])
        train_loader.generator.set_state(resume_state["rng"]["batch_order"])
        torch.set_rng_state(resume_state["rng"]["torch"])
        if feature_used is not None:
            model.set_feature(feature_used.to(device))
        first_epoch = int(resume_state["epoch"]) + 1
    logger.info(
        "TRAINING_STARTED | model=student | epochs=%d | start_epoch=%d | optimizer=adam | lr=%g | conditioning=%s | spsa_updates_per_epoch=%d | ema=%s | differentiable_feature=%s",
        epochs,
        first_epoch,
        training_cfg["lr"],
        source.name if (source is not None and uses_feature) else "none",
        updates_per_epoch,
        f"beta={ema_beta}" if ema_enabled else "off",
        differentiable,
    )
    if resume_state is not None:
        logger.info(
            "TRAIN_RESUME_COMPLETED | restored_epochs=%d | next_epoch=%d",
            first_epoch - 1,
            first_epoch,
        )
    for epoch in range(first_epoch, epochs + 1):
        started = time.perf_counter()
        spsa_plus, spsa_minus = [], []
        for update in range(1, updates_per_epoch + 1):
            vector, objective_plus, objective_minus = spsa.step(
                source.vector(), objective_at
            )
            source.set_vector(vector)
            spsa_plus.append(objective_plus)
            spsa_minus.append(objective_minus)
            logger.debug(
                "SPSA_STEP | epoch=%d | update=%d | J_plus=%.6f | J_minus=%.6f",
                epoch,
                update,
                objective_plus,
                objective_minus,
            )

        if uses_feature and not differentiable:
            feature_raw = source.feature(generator)
            if ema_enabled and feature_used is not None:
                feature_used = ema_beta * feature_used + (1.0 - ema_beta) * feature_raw
            else:
                feature_used = feature_raw.clone()
            model.set_feature(feature_used.to(device))
            trace_raw.append(feature_raw.clone())
            trace_used.append(feature_used.clone())

        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels, teacher_logits in train_loader:
            images, labels, teacher_logits = (
                images.to(device),
                labels.to(device),
                teacher_logits.to(device),
            )
            if differentiable:
                # Fresh exact feature per batch so gradients reach theta.
                logits = model(
                    images, feature=source.differentiable_feature().to(device)
                )
            else:
                logits = model(images)
            loss = kd_loss(logits, teacher_logits, labels, temperature, hard_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()

        if differentiable:
            with torch.no_grad():
                feature_used = source.differentiable_feature().detach().float()
            model.set_feature(feature_used.to(device))
            trace_raw.append(feature_used.clone())
            trace_used.append(feature_used.clone())

        validation = evaluate(model, val_loader, device)
        proxy = proxy_objective(None)
        vector = (
            source.vector()
            if (source is not None and (learnable or differentiable))
            else None
        )
        record = {
            "epoch": epoch,
            "train_kd_loss": loss_sum / total,
            "train_accuracy": 100.0 * correct / total,
            "val_accuracy": validation["accuracy"],
            "val_cross_entropy": validation["cross_entropy"],
            "proxy_objective": proxy,
            "spsa_mean_objective": (sum(spsa_plus) + sum(spsa_minus))
            / (2 * len(spsa_plus))
            if spsa_plus
            else None,
            "parameter_norm": float(vector.norm()) if vector is not None else None,
            "feature_raw_std": float(trace_raw[-1].std()) if trace_raw else None,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        logger.info(
            "TRAIN_EPOCH_COMPLETED | model=student | epoch=%d/%d | train_kd=%.5f | train_acc=%.3f | val_ce=%.5f | val_acc=%.3f | J=%.5f | param_norm=%s | seconds=%.1f",
            epoch,
            epochs,
            record["train_kd_loss"],
            record["train_accuracy"],
            record["val_cross_entropy"],
            record["val_accuracy"],
            proxy,
            f"{record['parameter_norm']:.4f}"
            if record["parameter_norm"] is not None
            else "n/a",
            record["seconds"],
        )
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "source": source.checkpoint_state() if source is not None else {},
                    "feature_used": feature_used,
                    "history": history,
                    "trace_raw": trace_raw,
                    "trace_used": trace_used,
                    "rng": {
                        "sampling": generator.get_state(),
                        "batch_order": train_loader.generator.get_state(),
                        "torch": torch.get_rng_state(),
                    },
                },
            )
            logger.info(
                "CHECKPOINT_WRITTEN | path=%s | epoch=%d", checkpoint_path.name, epoch
            )
    return StudentResult(history, feature_used, trace_raw, trace_used)
