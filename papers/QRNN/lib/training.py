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
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    device: torch.device,
    target_scale: float | None = None,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> tuple[float, float | None]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    rel_sq_sum = 0.0
    rel_count = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        mask = batch_y != 0
        if mask.any():
            t_mean = torch.tensor(
                target_mean if target_mean is not None else 0.0,
                device=batch_y.device,
                dtype=batch_y.dtype,
            )
            t_std = torch.tensor(
                target_std if target_std is not None else 1.0,
                device=batch_y.device,
                dtype=batch_y.dtype,
            )
            raw_t = batch_y[mask] * t_std + t_mean
            raw_p = preds[mask] * t_std + t_mean
            scale = target_scale if target_scale is not None else 0.0
            denom = torch.clamp(raw_t.abs(), min=max(1e-6, scale))
            rel = ((raw_t - raw_p) / denom).detach()
            rel_sq_sum += torch.sum(rel**2).item()
            rel_count += rel.numel()

    loss_value = total_loss / max(total_samples, 1)
    if rel_count == 0:
        return loss_value, None
    rmse_rel = (rel_sq_sum / rel_count) ** 0.5
    accuracy = (1 - rmse_rel) * 100
    return loss_value, float(accuracy)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
    *,
    photonic_optimizer_step: str = "per_sample",
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    # Local import to avoid making photonic support a hard dependency for readers.
    try:
        from .photonic_qrnn import PhotonicQRNNRegressor  # type: ignore
    except Exception:  # pragma: no cover
        PhotonicQRNNRegressor = None  # type: ignore[assignment]

    photonic_optimizer_step = str(photonic_optimizer_step).strip().lower()
    if photonic_optimizer_step not in {"per_sample", "per_batch"}:
        raise ValueError(
            "photonic_optimizer_step must be 'per_sample' or 'per_batch', "
            f"got {photonic_optimizer_step!r}"
        )

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_size = int(batch_x.size(0))

        is_photonic = (
            PhotonicQRNNRegressor is not None
            and isinstance(model, PhotonicQRNNRegressor)
            and batch_size > 1
        )

        if is_photonic and photonic_optimizer_step == "per_sample":
            # NOTE: With large batch sizes, stepping once per batch dramatically
            # reduces the number of optimizer updates per epoch (e.g., ~67 vs ~3
            # steps on the tiny sin dataset). Since the photonic forward is
            # sequential anyway, we default to per-sample stepping for a fairer
            # comparison with batch_size=1.
            loss_sum = 0.0
            for i in range(batch_size):
                optimizer.zero_grad()
                preds_i = model(batch_x[i : i + 1])
                loss_i = criterion(preds_i, batch_y[i : i + 1])
                loss_i.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                loss_sum += float(loss_i.item())
            loss_value = loss_sum / batch_size
        else:
            optimizer.zero_grad()
            if is_photonic:
                # Avoid holding the computation graph for all samples simultaneously.
                # Accumulate gradients over per-sample losses scaled to match the
                # default mean reduction, then apply a single optimizer step.
                loss_sum = 0.0
                for i in range(batch_size):
                    preds_i = model(batch_x[i : i + 1])
                    loss_i = criterion(preds_i, batch_y[i : i + 1])
                    (loss_i / batch_size).backward()
                    loss_sum += float(loss_i.item())
                loss_value = loss_sum / batch_size
            else:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                loss_value = float(loss.item())

            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss_value * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
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

    optimizer_name = str(training_cfg.get("optimizer", "adam")).strip().lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        momentum = float(training_cfg.get("momentum", 0.9))
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}' (expected: adam, adamw, sgd)"
        )

    loss_name = str(training_cfg.get("loss", "mse")).strip().lower()
    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name in {"huber", "smoothl1"}:
        delta = float(training_cfg.get("huber_delta", 1.0))
        criterion = nn.HuberLoss(delta=delta)
    else:
        raise ValueError(f"Unsupported loss '{loss_name}' (expected: mse, huber)")

    scheduler = None
    scheduler_cfg = training_cfg.get("scheduler")
    if scheduler_cfg:
        if isinstance(scheduler_cfg, str):
            scheduler_name = scheduler_cfg.strip().lower()
            scheduler_params: dict = {}
        else:
            scheduler_name = str(scheduler_cfg.get("name", "")).strip().lower()
            scheduler_params = dict(scheduler_cfg.get("params", {}) or {})

        if scheduler_name in {"plateau", "reduceonplateau", "reduce_lr_on_plateau"}:
            factor = float(scheduler_params.get("factor", 0.5))
            patience = int(scheduler_params.get("patience", 5))
            min_lr = float(scheduler_params.get("min_lr", 1e-6))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )
        elif scheduler_name in {"cosine", "cosineannealing"}:
            t_max = int(scheduler_params.get("t_max", max(1, epochs)))
            eta_min = float(scheduler_params.get("min_lr", 0.0))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=eta_min
            )
        else:
            raise ValueError(
                "Unsupported scheduler (expected: plateau, cosine). "
                f"Got: {scheduler_name!r}"
            )

    metadata = cfg.get("metadata", {}) or {}
    target_scale = metadata.get("target_abs_mean")
    target_mean = metadata.get("target_mean")
    target_std = metadata.get("target_std")

    history: list[dict[str, float]] = []
    early_cfg = training_cfg.get("early_stopping") or {}
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 10))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    best_val = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        photonic_step = str(training_cfg.get("photonic_optimizer_step", "per_sample"))
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip,
            photonic_optimizer_step=photonic_step,
        )
        val_loss, val_acc = evaluate_metrics(
            model, val_loader, criterion, device, target_scale, target_mean, target_std
        )
        test_loss = test_acc = None
        if test_loader is not None and len(test_loader.dataset) > 0:  # type: ignore[arg-type]
            test_loss, test_acc = evaluate_metrics(
                model,
                test_loader,
                criterion,
                device,
                target_scale,
                target_mean,
                target_std,
            )

        LOGGER.info(
            "Epoch %d/%d - train_loss=%.4f - val_loss=%.4f - val_acc=%s - test_acc=%s",
            epoch,
            epochs,
            train_loss,
            val_loss,
            f"{val_acc:.2f}%" if val_acc is not None else "n/a",
            f"{test_acc:.2f}%" if test_acc is not None else "n/a",
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss + early_min_delta < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        if early_enabled and (epoch - best_epoch) >= early_patience:
            LOGGER.info(
                "Early stopping at epoch %d (best_val=%.4f at epoch %d)",
                epoch,
                best_val,
                best_epoch,
            )
            break
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }
        )

    if best_state is not None and early_enabled:
        model.load_state_dict(best_state)

    last = history[-1]
    metrics = {
        "final_train_loss": last["train_loss"],
        "final_val_loss": last["val_loss"],
        "final_val_accuracy": last["val_accuracy"],
        "final_test_loss": last["test_loss"],
        "final_test_accuracy": last["test_accuracy"],
        "best_val_loss": best_val if best_state is not None else last["val_loss"],
        "best_epoch": best_epoch if best_state is not None else last["epoch"],
        "history": history,
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved metrics to %s", metrics_path)
    return metrics


__all__ = ["fit", "train_one_epoch", "evaluate_metrics"]
