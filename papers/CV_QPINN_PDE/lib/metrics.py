"""Evaluation metrics used in the paper Tables IV and the Poisson section."""

from __future__ import annotations

import torch


def rmse(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((pred - ref) ** 2).mean())


def mae(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return (pred - ref).abs().mean()


def l_infinity(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return (pred - ref).abs().max()


def nmse(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """NMSE = ||pred - ref||^2 / ||ref||^2."""
    num = ((pred - ref) ** 2).mean()
    den = (ref ** 2).mean()
    return num / den


def summarise(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    return {
        "rmse": float(rmse(pred, ref).item()),
        "mae": float(mae(pred, ref).item()),
        "l_inf": float(l_infinity(pred, ref).item()),
        "nmse": float(nmse(pred, ref).item()),
    }


__all__ = ["rmse", "mae", "l_infinity", "nmse", "summarise"]
