"""Generic training loops shared by the QPINN and the classical PINN baseline."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import torch


def cosine_annealing_with_warm_restarts(initial_lr: float, t: int, period: int,
                                        min_lr: float = 1e-5) -> float:
    """Single-cycle cosine annealing with warm restart at every ``period`` step.

    Matches the schedule used in the paper for the heat-equation training run.
    """
    t_in_cycle = t % period
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * t_in_cycle / period))


def train_poisson(model, x_collocation: torch.Tensor, x_bc: tuple[torch.Tensor, torch.Tensor],
                  *, loss_fn: Callable, lr: float, epochs: int,
                  lambdas: dict[str, float], history_path: Path | None = None,
                  log_every: int = 50, lr_schedule: dict | None = None) -> dict:
    """Train any model that conforms to the (u, ux, trace) signature on the Poisson task."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict] = []
    best = {"epoch": -1, "loss": float("inf"), "state": None}
    x_bc_left, x_bc_right = x_bc
    start = time.time()
    for epoch in range(epochs):
        if lr_schedule is not None:
            new_lr = cosine_annealing_with_warm_restarts(
                lr_schedule["initial_lr"], epoch, lr_schedule["period"],
                min_lr=lr_schedule.get("min_lr", 1e-5),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
        optimizer.zero_grad()
        loss, parts = loss_fn(model, x_collocation, x_bc_left, x_bc_right, lambdas)
        loss.backward()
        optimizer.step()
        history.append({
            "epoch": epoch,
            **{k: float(v.item()) for k, v in parts.items()},
        })
        if loss.item() < best["loss"]:
            best = {"epoch": epoch, "loss": loss.item(),
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"  [epoch {epoch:5d}] total={parts['total'].item():.4e} "
                  f"pde={parts['pde'].item():.4e} bc={parts['bc'].item():.4e} "
                  f"consist={parts['consistency'].item():.4e} trace={parts['trace'].item():.4e}")
    elapsed = time.time() - start
    if history_path is not None:
        with open(history_path, "w") as fh:
            json.dump(history, fh)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"history": history, "best": {"epoch": best["epoch"], "loss": best["loss"]},
            "wall_time_sec": elapsed}


def train_heat(model, xt_collocation: torch.Tensor, xt_ic: torch.Tensor, T_ic: torch.Tensor,
               xt_bc: tuple[torch.Tensor, torch.Tensor], *, loss_fn: Callable,
               lr: float, pretrain_epochs: int, epochs: int,
               lambdas: dict[str, float], alpha: float,
               history_path: Path | None = None, log_every: int = 50,
               lr_schedule: dict | None = None) -> dict:
    """Train a model on the heat equation with an optional IC pre-training stage."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict] = []
    xt_bc_left, xt_bc_right = xt_bc
    best = {"epoch": -1, "loss": float("inf"), "state": None}
    start = time.time()
    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()
        loss, parts = loss_fn(model, xt_collocation, xt_ic, T_ic, xt_bc_left, xt_bc_right,
                              alpha, lambdas, pre_train_only_ic=True)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "phase": "pretrain",
                        **{k: float(v.item()) for k, v in parts.items()}})
        if epoch % log_every == 0:
            print(f"  [pre  {epoch:5d}] ic={parts['ic'].item():.4e}")
    for epoch in range(epochs):
        if lr_schedule is not None:
            new_lr = cosine_annealing_with_warm_restarts(
                lr_schedule["initial_lr"], epoch, lr_schedule["period"],
                min_lr=lr_schedule.get("min_lr", 1e-5),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
        optimizer.zero_grad()
        loss, parts = loss_fn(model, xt_collocation, xt_ic, T_ic, xt_bc_left, xt_bc_right,
                              alpha, lambdas, pre_train_only_ic=False)
        loss.backward()
        optimizer.step()
        history.append({"epoch": pretrain_epochs + epoch, "phase": "full",
                        **{k: float(v.item()) for k, v in parts.items()}})
        if loss.item() < best["loss"]:
            best = {"epoch": epoch, "loss": loss.item(),
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"  [full {epoch:5d}] total={parts['total'].item():.4e} "
                  f"pde={parts['pde'].item():.4e} ic={parts['ic'].item():.4e} "
                  f"bc={parts['bc'].item():.4e}")
    elapsed = time.time() - start
    if history_path is not None:
        with open(history_path, "w") as fh:
            json.dump(history, fh)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"history": history, "best": {"epoch": best["epoch"], "loss": best["loss"]},
            "wall_time_sec": elapsed}


__all__ = ["train_poisson", "train_heat", "cosine_annealing_with_warm_restarts"]
