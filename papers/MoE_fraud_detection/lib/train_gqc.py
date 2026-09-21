"""Joint reconstruction + classification training loop for :class:`GQCModel`.

Per minibatch:

- ``L_R`` = MSE reconstruction loss, computed ONLY over the non-fraud rows
  in the minibatch (``y == 0``). If a minibatch has zero non-fraud rows the
  reconstruction term is skipped for that batch (divide-by-zero guard).
- ``L_C`` = BCELoss over ALL rows in the minibatch between ``p_hat`` and
  ``y``.
- Total loss = ``lambda_ * L_R + (1 - lambda_) * L_C``.

Optimizer: Adam. All hyperparameters (``lambda_recon``, ``lr``, epochs,
batch size) come from ``cfg``; see ``configs/defaults.json`` for defaults and
LOG.md for the epoch-count assumption (the paper does not specify one).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn

from .gqc_model import GQCModel

logger = logging.getLogger(__name__)


def train_gqc_model(
    model: GQCModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict[str, Any],
) -> list[float]:
    """Train ``model`` in place on the balanced training pool.

    Returns the per-epoch mean total-loss history (useful for a quick
    loss-curve sanity check).
    """
    model_cfg = cfg.get("model", {})
    lambda_ = float(model_cfg.get("lambda_recon", 0.5))
    training_cfg = cfg.get("training", {})
    lr = float(training_cfg.get("lr", 1e-3))
    epochs = int(training_cfg.get("epochs", 30))
    batch_size = int(cfg.get("dataset", {}).get("batch_size", 32))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    n = X_t.shape[0]

    model.train()
    epoch_losses: list[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        batch_losses: list[float] = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_t[idx], y_t[idx]

            optimizer.zero_grad()
            p_hat, x_recon = model(xb)

            nonfraud_mask = yb == 0
            if nonfraud_mask.any():
                l_r = mse(x_recon[nonfraud_mask], xb[nonfraud_mask])
            else:
                l_r = torch.zeros((), dtype=xb.dtype)

            l_c = bce(p_hat, yb)
            loss = lambda_ * l_r + (1.0 - lambda_) * l_c
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        epoch_losses.append(mean_loss)
        logger.debug("GQC epoch %d/%d: mean loss=%.5f", epoch + 1, epochs, mean_loss)

    model.eval()
    return epoch_losses


__all__ = ["train_gqc_model"]
