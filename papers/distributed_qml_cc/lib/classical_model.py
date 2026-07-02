"""Fair iso-parameter MLP baseline for the DQML classification task.

The DQML schemes of the paper differ in parameter count: at L=9 we have
roughly 50 (non), 100 (NC), 110 (CC), 120 (QC) trainable parameters.
This module trains a fully classical 2-hidden-layer MLP whose hidden
width is chosen so its parameter count brackets the quantum models'
range. The resulting accuracy provides a "free of geometric inductive
bias" reference: if a tiny classical network can solve the synthetic
task already, the quantum-classical gap reduces to one of inductive
bias rather than expressivity.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import SyntheticDatasetConfig, build_synthetic_dataset

LOGGER = logging.getLogger(__name__)


class TinyMLP(nn.Module):
    """Two-hidden-layer MLP with tanh activations and scalar output."""

    def __init__(self, n_in: int = 8, hidden: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x).squeeze(-1)


def num_mlp_parameters(n_in: int, hidden: int) -> int:
    """Count parameters of the two-hidden-layer MLP."""
    return (n_in * hidden + hidden) + (hidden * hidden + hidden) + (hidden + 1)


def _accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return ((pred >= 0).float() * 2 - 1 == y).float().mean().item()


def train_classical_baseline(cfg: dict, run_dir: Path, dtype: torch.dtype) -> dict[str, Any]:
    dataset_cfg = SyntheticDatasetConfig(**cfg.get("dataset", {}).get("params", {}))
    model_cfg = cfg.get("model", {}).get("params", {})
    hidden = int(model_cfg.get("hidden", 6))
    training_cfg = cfg.get("training", {})
    n_iterations = int(training_cfg.get("n_iterations", 1000))
    lr = float(training_cfg.get("lr", 0.05))
    batch_size = int(training_cfg.get("batch_size", 512))
    eval_every = int(training_cfg.get("eval_every", 50))
    seeds = list(cfg.get("seeds") or [int(cfg.get("seed", 42))])

    x_tr, y_tr, x_va, y_va, info = build_synthetic_dataset(dataset_cfg)
    x_tr_t = torch.as_tensor(x_tr, dtype=dtype)
    y_tr_t = torch.as_tensor(y_tr, dtype=dtype)
    x_va_t = torch.as_tensor(x_va, dtype=dtype)
    y_va_t = torch.as_tensor(y_va, dtype=dtype)

    n_params = num_mlp_parameters(dataset_cfg.n_dim, hidden)
    summary: dict[str, Any] = {
        "pipeline": "classical",
        "model": "tiny_mlp",
        "hidden": hidden,
        "n_params": n_params,
        "n_iterations": n_iterations,
        "lr": lr,
        "batch_size": batch_size,
        "dataset_info": {
            "n_train": int(x_tr.shape[0]),
            "n_val": int(x_va.shape[0]),
            "pearson_max_abs": info["pearson_max_abs"],
            "seed": dataset_cfg.seed,
        },
        "seeds": seeds,
        "runs": [],
    }
    for seed in seeds:
        torch.manual_seed(int(seed))
        model = TinyMLP(n_in=dataset_cfg.n_dim, hidden=hidden).to(dtype=dtype)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        run = {"seed": int(seed), "n_params": n_params, "history": {
            "iteration": [], "train_loss": [], "train_acc": [], "val_acc": []}}
        t0 = time.time()
        for it in range(1, n_iterations + 1):
            gen = torch.Generator().manual_seed(seed + it)
            idx = torch.randint(0, x_tr_t.shape[0], (batch_size,), generator=gen)
            xb, yb = x_tr_t[idx], y_tr_t[idx]
            pred = model(xb)
            loss = ((yb - pred) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            if it % eval_every == 0 or it == n_iterations:
                model.eval()
                with torch.no_grad():
                    train_acc = _accuracy(model(x_tr_t), y_tr_t)
                    val_acc = _accuracy(model(x_va_t), y_va_t)
                model.train()
                run["history"]["iteration"].append(it)
                run["history"]["train_loss"].append(float(loss.item()))
                run["history"]["train_acc"].append(train_acc)
                run["history"]["val_acc"].append(val_acc)
        run["wall_clock_seconds"] = time.time() - t0
        run["final_val_acc"] = run["history"]["val_acc"][-1]
        run["final_train_acc"] = run["history"]["train_acc"][-1]
        run["final_loss"] = run["history"]["train_loss"][-1]
        summary["runs"].append(run)
        LOGGER.info(
            "  MLP hidden=%d seed=%d val_acc=%.4f time=%.1fs",
            hidden, seed, run["final_val_acc"], run["wall_clock_seconds"],
        )

    val_accs = np.array([r["final_val_acc"] for r in summary["runs"]])
    summary["mean_val_acc"] = float(val_accs.mean())
    summary["std_val_acc"] = float(val_accs.std(ddof=0))
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary
