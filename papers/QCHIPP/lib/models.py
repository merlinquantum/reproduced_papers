"""Model registry shared by training and checkpoint inference."""

from __future__ import annotations

import torch
from torch import nn

from runtime_lib.dtypes import dtype_torch

from .classical import ClassicalQCNN
from .gate import GateQCNN
from .photonic import PhotonicQCNN, ReferencePhotonicQCNN

PHOTONIC_MODELS = ("merlin", "merlin_reference")
CLASSICAL_MODELS = ("classical", "classical_reference")
GATE_MODELS = ("gate_source", "gate_corrected")
TRAINABLE_MODELS = PHOTONIC_MODELS + CLASSICAL_MODELS + GATE_MODELS


def config_dtype(cfg: dict) -> torch.dtype:
    """Return the torch dtype of a raw or runtime-normalised config."""
    return dtype_torch(cfg["dtype"])


def build_trainable_model(cfg: dict) -> nn.Module:
    """Construct the configured trainable model before any training step.

    Parameters
    ----------
    cfg : dict
        Needs ``model.name``, ``seed`` and ``dtype``; classical models also
        need ``model.hidden``.

    Returns
    -------
    torch.nn.Module
        Freshly initialised model.

    Raises
    ------
    ValueError
        For an unknown model name, or a non-float32 dtype on the photonic
        models (the MerLin adaptation is verified in float32 only).
    """
    name = cfg["model"]["name"]
    dtype = config_dtype(cfg)
    if name in PHOTONIC_MODELS:
        if dtype != torch.float32:
            raise ValueError("The MerLin adaptation is verified in float32 only")
        return ReferencePhotonicQCNN() if name == "merlin_reference" else PhotonicQCNN()
    if name in CLASSICAL_MODELS:
        return ClassicalQCNN(hidden=cfg["model"]["hidden"]).to(dtype=dtype)
    if name in GATE_MODELS:
        return GateQCNN(seed=int(cfg["seed"]), dtype=dtype)
    raise ValueError(f"Unsupported trainable model: {name}")
