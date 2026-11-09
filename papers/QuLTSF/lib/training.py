"""Training and evaluation loops, following upstream ``exp/exp_main.py``.

Upstream behaviour reproduced here:
- Adam on MSE, one optimizer step per training batch;
- after every epoch: validation MSE (mean of per-batch MSEs), early stopping
  on it (patience, strict improvement), then the ``lradj`` learning-rate rule
  of ``utils/tools.py::adjust_learning_rate``;
- the best-validation weights are restored before testing;
- metrics are computed on standardised values, with ``MS`` scoring only the
  last (target) channel.

Deviation: upstream also evaluates the test set after every epoch and only
prints it; this loop does not touch the test set during training.
"""

from __future__ import annotations

import copy
import logging
import math
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

LR_SCHEDULES = ("type1", "type3", "constant")


def scheduled_learning_rate(schedule: str, base_lr: float, epoch: int) -> float:
    """Return the learning rate set after finishing ``epoch`` (1-based).

    Mirrors ``adjust_learning_rate(optimizer, scheduler, epoch + 1, args)``
    in upstream ``utils/tools.py``; the returned rate is used for the next epoch.

    Parameters
    ----------
    schedule : str
        ``"type3"`` (upstream default: keep ``base_lr`` while ``epoch < 3``,
        then ``base_lr * 0.9 ** (epoch - 3)``), ``"type1"`` (halve every
        epoch, LTSF-Linear's default) or ``"constant"``.
    base_lr : float
        Initial learning rate.
    epoch : int
        Number of the epoch just finished, starting at 1.

    Returns
    -------
    float
        Learning rate for the next epoch.
    """
    if schedule == "type3":
        return base_lr if epoch < 3 else base_lr * 0.9 ** (epoch - 3)
    if schedule == "type1":
        return base_lr * 0.5 ** (epoch - 1)
    if schedule == "constant":
        return base_lr
    raise ValueError(f"training.lr_schedule must be one of {LR_SCHEDULES}")


class EarlyStopping:
    """Upstream ``EarlyStopping`` with ``delta=0``, keeping the best weights.

    Parameters
    ----------
    patience : int
        Epochs without strict improvement before stopping.
    """

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = math.inf
        self.best_epoch = 0
        self.best_state: dict | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float, model: torch.nn.Module, epoch: int) -> bool:
        """Record one validation loss; return True if it improved."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def _scored(prediction: torch.Tensor, target: torch.Tensor, features_mode: str):
    if features_mode == "MS":
        return prediction[..., -1:], target[..., -1:]
    return prediction, target


def train_one_epoch(
    model, loader, optimizer, device, features_mode: str, max_batches: int | None
) -> dict:
    """Run one training epoch and return its mean batch MSE and batch count."""
    model.train()
    criterion = torch.nn.MSELoss()
    losses = []
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        prediction, target = _scored(model(inputs), targets, features_mode)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (batch_index + 1) % 100 == 0:
            logger.debug(
                "TRAIN_BATCH | batch=%d | loss=%.6f", batch_index + 1, losses[-1]
            )
    if not losses:
        raise ValueError("The training loader produced no batches.")
    return {"loss": float(np.mean(losses)), "batches": len(losses)}


@torch.no_grad()
def validation_loss(model, loader, device, features_mode: str) -> float:
    """Return the mean of per-batch validation MSEs, as upstream ``vali``."""
    model.eval()
    criterion = torch.nn.MSELoss()
    losses = []
    for inputs, targets in loader:
        prediction, target = _scored(
            model(inputs.to(device)), targets.to(device), features_mode
        )
        losses.append(criterion(prediction, target).item())
    if not losses:
        raise ValueError("The validation loader produced no batches.")
    return float(np.mean(losses))


@torch.no_grad()
def per_window_errors(
    model, loader, device, features_mode: str
) -> dict[str, np.ndarray]:
    """Return each window's mean squared and mean absolute error.

    Every window has the same number of scored elements (``T x channels``),
    so the mean over windows equals upstream's mean over all elements.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``"squared"`` and ``"absolute"``, each of shape ``[n_windows]``
        (float64), in loader order.
    """
    model.eval()
    squared, absolute = [], []
    for inputs, targets in loader:
        prediction, target = _scored(
            model(inputs.to(device)), targets.to(device), features_mode
        )
        error = (prediction - target).double()
        squared.append(error.pow(2).mean(dim=(1, 2)).cpu())
        absolute.append(error.abs().mean(dim=(1, 2)).cpu())
    return {
        "squared": torch.cat(squared).numpy(),
        "absolute": torch.cat(absolute).numpy(),
    }


def summarise_errors(errors: dict[str, np.ndarray], batch_size: int) -> dict:
    """Return MSE/MAE over all windows and over the upstream drop-last subset.

    The upstream test loader (``shuffle=False, drop_last=True``) scores only
    the first ``floor(n / batch_size) * batch_size`` windows.
    """
    n_windows = int(errors["squared"].shape[0])
    kept = (n_windows // batch_size) * batch_size
    return {
        "mse": float(errors["squared"].mean()),
        "mae": float(errors["absolute"].mean()),
        "n_windows": n_windows,
        "mse_drop_last": float(errors["squared"][:kept].mean()),
        "mae_drop_last": float(errors["absolute"][:kept].mean()),
        "n_windows_drop_last": kept,
    }


def fit(model, train_loader, val_loader, cfg: dict, device: torch.device) -> dict:
    """Train with early stopping and restore the best-validation weights.

    Parameters
    ----------
    model : torch.nn.Module
        Model already on ``device``.
    train_loader, val_loader : torch.utils.data.DataLoader
        Training and validation batches.
    cfg : dict
        Resolved configuration with a ``training`` section.
    device : torch.device
        Device of the model.

    Returns
    -------
    dict
        Per-epoch history, best epoch and loss, stop reason and wall-clock.
    """
    training_cfg = cfg["training"]
    for key in ("epochs", "lr", "lr_schedule", "patience"):
        if key not in training_cfg:
            raise KeyError(f"Missing required config key training.{key}")
    epochs = int(training_cfg["epochs"])
    base_lr = float(training_cfg["lr"])
    schedule = str(training_cfg["lr_schedule"])
    max_batches = training_cfg.get("max_batches_per_epoch")
    features_mode = str(cfg["dataset"]["features_mode"])

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    stopper = EarlyStopping(int(training_cfg["patience"]))
    history = []
    stop_reason = "max_epochs"
    started = time.perf_counter()
    logger.info(
        "TRAINING_STARTED | epochs=%d | optimizer=adam | lr=%g | lr_schedule=%s | "
        "patience=%d | train_batches=%d | max_batches_per_epoch=%s",
        epochs,
        base_lr,
        schedule,
        stopper.patience,
        len(train_loader),
        max_batches,
    )
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        train = train_one_epoch(
            model, train_loader, optimizer, device, features_mode, max_batches
        )
        val_loss = validation_loss(model, val_loader, device, features_mode)
        improved = stopper.step(val_loss, model, epoch)
        epoch_seconds = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train["loss"],
                "val_loss": val_loss,
                "improved": improved,
                "seconds": epoch_seconds,
            }
        )
        logger.info(
            "TRAIN_EPOCH_COMPLETED | epoch=%d/%d | loss=%.6f | val_loss=%.6f | lr=%.6g | "
            "improved=%s | best_epoch=%d | seconds=%.1f",
            epoch,
            epochs,
            train["loss"],
            val_loss,
            current_lr,
            improved,
            stopper.best_epoch,
            epoch_seconds,
        )
        if stopper.should_stop:
            stop_reason = "early_stopping"
            logger.info(
                "EARLY_STOPPING | epoch=%d | patience=%d | best_epoch=%d",
                epoch,
                stopper.patience,
                stopper.best_epoch,
            )
            break
        for group in optimizer.param_groups:
            group["lr"] = scheduled_learning_rate(schedule, base_lr, epoch)

    model.load_state_dict(stopper.best_state)
    wall_clock = time.perf_counter() - started
    logger.info(
        "TRAINING_COMPLETED | epochs_run=%d | best_epoch=%d | best_val_loss=%.6f | "
        "stop_reason=%s | seconds=%.1f",
        len(history),
        stopper.best_epoch,
        stopper.best_loss,
        stop_reason,
        wall_clock,
    )
    return {
        "history": history,
        "epochs_run": len(history),
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_loss,
        "stop_reason": stop_reason,
        "seconds": wall_clock,
    }


__all__ = [
    "EarlyStopping",
    "LR_SCHEDULES",
    "fit",
    "per_window_errors",
    "scheduled_learning_rate",
    "summarise_errors",
    "train_one_epoch",
    "validation_loss",
]
