"""Training loop and hybrid loss for QEGM rare-event reproduction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import GMMDataset, iter_batches

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    losses: list
    val_losses: list
    train_time_s: float


def hybrid_loss(
    x: torch.Tensor,
    out: dict,
    *,
    tail_threshold: float,
    lambda_rec: float,
    lambda_tail: float,
    lambda_kld: float,
) -> tuple[torch.Tensor, dict]:
    """Hybrid reconstruction + tail-aware + KLD loss (paper Eqs. 5 and 17)."""

    x_hat = out["x_hat"]
    mu = out["mu"]
    log_var = out["log_var"]

    rec = F.mse_loss(x_hat, x, reduction="mean")

    tail_mask = (x.abs() > tail_threshold).float()
    n_tail = tail_mask.sum()
    if n_tail.item() > 0:
        tail_loss = (tail_mask * (x_hat - x).pow(2)).sum() / n_tail
    else:
        tail_loss = torch.zeros((), device=x.device)

    kld = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())

    total = lambda_rec * rec + lambda_tail * tail_loss + lambda_kld * kld
    parts = {
        "loss": float(total.detach()),
        "rec": float(rec.detach()),
        "tail": float(tail_loss.detach()),
        "kld": float(kld.detach()),
    }
    return total, parts


def train_one(
    model: torch.nn.Module,
    dataset: GMMDataset,
    cfg: dict,
    seed: int,
) -> TrainResult:
    """Train ``model`` for one seed and return the loss histories."""

    train_cfg = cfg["training"]
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    lr = float(train_cfg["lr"])
    lambda_rec = float(train_cfg["lambda_rec"])
    lambda_tail = float(train_cfg["lambda_tail"])
    lambda_kld = float(train_cfg["lambda_kld"])
    tail_threshold = float(dataset.tail_threshold)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[dict] = []
    val_losses: list[dict] = []

    torch.manual_seed(seed)
    np.random.seed(seed)

    start = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_parts: list[dict] = []
        for batch in iter_batches(
            dataset.train, batch_size, shuffle=True, seed=seed * 1000 + epoch
        ):
            optimizer.zero_grad()
            out = model(batch)
            loss, parts = hybrid_loss(
                batch,
                out,
                tail_threshold=tail_threshold,
                lambda_rec=lambda_rec,
                lambda_tail=lambda_tail,
                lambda_kld=lambda_kld,
            )
            loss.backward()
            optimizer.step()
            epoch_parts.append(parts)

        mean = {k: float(np.mean([p[k] for p in epoch_parts])) for k in epoch_parts[0]}
        losses.append({"epoch": epoch, **mean})

        model.eval()
        with torch.no_grad():
            out = model(dataset.val)
            _, val_parts = hybrid_loss(
                dataset.val,
                out,
                tail_threshold=tail_threshold,
                lambda_rec=lambda_rec,
                lambda_tail=lambda_tail,
                lambda_kld=lambda_kld,
            )
        val_losses.append({"epoch": epoch, **val_parts})
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(
                "epoch %d/%d  train_rec=%.4f  train_tail=%.4f  val_rec=%.4f",
                epoch + 1,
                epochs,
                mean["rec"],
                mean["tail"],
                val_parts["rec"],
            )

    return TrainResult(
        losses=losses, val_losses=val_losses, train_time_s=time.time() - start
    )


@torch.no_grad()
def generate(model: torch.nn.Module, n: int, device: str = "cpu") -> np.ndarray:
    """Generate ``n`` samples and return them as a flat numpy array."""

    model.eval()
    samples = model.sample(n, device=device)
    return samples.cpu().numpy()
