"""Shared-runtime entrypoint for the BVE photonic dual-rail QNN reproduction.

Reproduces Experiment 1 of the paper (main.tex) with MerLin: a photonic
dual-rail QNN trained to regress the Barotropic Vorticity Equation stream
function psi(t, x, y, z) against a reference SEM solution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .data import load_dataset
from .metrics import compute_figures_of_merit
from .model import build_model

logger = logging.getLogger(__name__)


def _resolve_checkpoint_path(cfg: dict[str, Any]) -> Path | None:
    checkpoint_name = cfg.get("model", {}).get("checkpoint")
    if not checkpoint_name:
        return None
    path = Path("models") / checkpoint_name
    return path if path.exists() else None


def _load_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: Path
) -> list[float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss_history = list(checkpoint["loss_history"])
    logger.info(
        "Resumed from checkpoint %s (step=%s, last_loss=%.6e)",
        checkpoint_path,
        checkpoint.get("step"),
        loss_history[-1] if loss_history else float("nan"),
    )
    return loss_history


def _train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    loss_history: list[float],
    total_steps: int,
) -> list[float]:
    loss_fn = nn.MSELoss()
    data_iterator = iter(dataloader)
    model.train()

    start_step = len(loss_history)
    for global_step in range(start_step + 1, total_steps + 1):
        try:
            x_batch, y_batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            x_batch, y_batch = next(data_iterator)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.detach()))
        if global_step % 100 == 0 or global_step == total_steps:
            logger.info("step %04d | loss = %.6e", global_step, loss_history[-1])

    return loss_history


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(cfg)
    features_tensor = dataset["features_tensor"]
    targets_tensor = dataset["targets_tensor"]
    dataloader = dataset["dataloader"]
    psi_qcl_training = dataset["psi_qcl_training"]
    training_hours = dataset["training_hours"]

    model = build_model(cfg, targets_tensor)

    training_cfg = cfg.get("training", {})
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training_cfg.get("lr", 1e-2))
    )

    loss_history: list[float] = []
    checkpoint_path = _resolve_checkpoint_path(cfg)
    if checkpoint_path is not None:
        loss_history = _load_checkpoint(model, optimizer, checkpoint_path)
    else:
        logger.info("No checkpoint found, training from scratch")

    total_steps = int(training_cfg.get("total_steps", 5000))
    if len(loss_history) < total_steps:
        loss_history = _train(model, optimizer, dataloader, loss_history, total_steps)
    else:
        logger.info(
            "Checkpoint already at/after target step (%d >= %d), skipping training",
            len(loss_history),
            total_steps,
        )

    final_checkpoint = {
        "architecture": "merlin_dualrail_photonic",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
        "step": len(loss_history),
        "depth": cfg.get("model", {}).get("params", {}).get("depth", 32),
        "n_qubits": cfg.get("model", {}).get("params", {}).get("n_qubits", 6),
    }
    torch.save(final_checkpoint, run_dir / "checkpoint.pt")

    model.eval()
    with torch.no_grad():
        mse = torch.mean((model(features_tensor) - targets_tensor) ** 2)
        psi_pred_flat = model(features_tensor).detach().cpu().numpy()

    psi_pred_training = psi_pred_flat.reshape(psi_qcl_training.shape)
    figures_of_merit = compute_figures_of_merit(psi_pred_training, psi_qcl_training)

    logger.info("Final training MSE: %.6e", float(mse))
    logger.info("Median MRE percent: %.3f", figures_of_merit["median_mre_percent"])
    logger.info("Median PPMCC: %.3f", figures_of_merit["median_ppmcc"])

    np.savez(
        run_dir / "exp1_merlin_results.npz",
        psi_pred_training=psi_pred_training,
        psi_qcl_training=psi_qcl_training,
        training_hours=training_hours,
        mre_per_time=figures_of_merit["mre_per_time"],
        ppmcc_per_grid_point=figures_of_merit["ppmcc_per_grid_point"],
        median_mre_percent=figures_of_merit["median_mre_percent"],
        median_ppmcc=figures_of_merit["median_ppmcc"],
    )

    metrics = {
        "final_step": len(loss_history),
        "final_training_mse": float(mse),
        "median_mre_percent": figures_of_merit["median_mre_percent"],
        "median_ppmcc": figures_of_merit["median_ppmcc"],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (run_dir / "done.txt").write_text("ok\n", encoding="utf-8")

    return metrics


__all__ = ["train_and_evaluate"]
