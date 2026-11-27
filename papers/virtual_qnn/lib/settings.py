"""Helpers for reading required values from the resolved configuration."""

from __future__ import annotations

import torch

from runtime_lib.dtypes import dtype_torch

from .quantum import complex_dtype_for


def real_dtype(cfg: dict) -> torch.dtype:
    """Return the real torch dtype selected by the top-level ``dtype`` key.

    Raises
    ------
    ValueError
        If ``dtype`` is missing or does not resolve to float32/float64.
    """
    resolved = dtype_torch(cfg.get("dtype"))
    if resolved not in (torch.float32, torch.float64):
        raise ValueError(
            f"Top-level dtype must be float32 or float64 for density-matrix "
            f"simulation, got {cfg.get('dtype')!r}"
        )
    return resolved


def complex_dtype(cfg: dict) -> torch.dtype:
    """Return the complex simulation dtype matching :func:`real_dtype`."""
    return complex_dtype_for(real_dtype(cfg))


def device(cfg: dict) -> torch.device:
    """Return the torch device from the required top-level ``device`` key."""
    return torch.device(cfg["device"])
