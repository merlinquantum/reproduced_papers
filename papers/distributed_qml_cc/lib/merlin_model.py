"""Single-chip photonic baseline (non-DQML) using angle encoding.

The baseline mirrors the FirstQuantumLayers MerLin tutorial:
* trainable MZI mesh on all modes,
* angle encoding ``add_angle_encoding`` on the feature-carrying modes,
* trainable rotations,
* a few more MZI meshes,
* threshold-detector PROBABILITIES readout,
* :class:`merlin.LexGrouping` into two class probabilities, and
* a fixed ``[+1, -1]`` classifier head producing a real-valued score
  trained against ``y in {-1, +1}`` with least-squares loss.

Inputs are z-score normalised on the training set before being passed
to the photonic layer, as in the tutorial. Default geometry is
``n_modes = 8, n_photons = 3`` so the 8 attributes map one-per-mode
via angle encoding.

The distributed (NC/CC) and quantum-communication (QC) variants live in
:mod:`lib.merlin_distributed`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import merlin as ml
import numpy as np
import torch
from torch import nn

from .data import SyntheticDatasetConfig, build_synthetic_dataset

LOGGER = logging.getLogger(__name__)


def _evenly_spread_state(n_modes: int, n_photons: int) -> list[int]:
    """Place ``n_photons`` photons at integer positions spread across all modes.

    Clustering photons at one end of the chip leaves modes outside their
    "light cone" unaffected by the trainable MZI phases — those phases
    cannot contribute to the readout. Spreading the photons evenly
    (e.g. positions ``0, 3, 6`` for ``n=3, m=8``) keeps every mode in
    the photon path so every trainable parameter is active.
    """
    if n_photons < 1 or n_photons > n_modes:
        raise ValueError("require 1 <= n_photons <= n_modes")
    state = [0] * n_modes
    for i in range(n_photons):
        # Round-half-down placement; guaranteed distinct because n_photons <= n_modes.
        pos = (i * n_modes) // n_photons
        while state[pos] == 1:
            pos += 1
        state[pos] = 1
    return state


# Backwards-compatible alias.
_left_filled_state = _evenly_spread_state


class PhotonicSingleChip(nn.Module):
    """Single-chip photonic classifier (angle encoding, tutorial recipe).

    Architecture (in order):

    1. trainable MZI entangling layer,
    2. ``add_angle_encoding(modes=[0..n_features-1], scale=angle_scale)``,
    3. trainable single-mode rotations,
    4. ``n_entangling_layers`` more trainable MZI meshes.

    Readout is PROBABILITIES over the UNBUNCHED subspace, grouped into
    two class buckets by ``LexGrouping`` and combined with a fixed
    ``[+1, -1]`` head.
    """

    def __init__(
        self,
        n_modes: int = 8,
        n_photons: int = 3,
        n_features: int = 8,
        angle_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if n_photons < 1 or n_photons >= n_modes:
            raise ValueError("require 1 <= n_photons < n_modes")
        if n_features > n_modes:
            raise ValueError("angle encoding cannot fit more features than modes")
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.n_features = n_features

        # A single trainable MZI entangling layer is photonically universal;
        # no need to stack additional meshes.
        builder = ml.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer(trainable=True)
        builder.add_angle_encoding(modes=list(range(n_features)), scale=angle_scale)
        builder.add_rotations(trainable=True)
        builder.add_entangling_layer(trainable=True)

        self.qlayer = ml.QuantumLayer(
            input_size=n_features,
            builder=builder,
            input_state=_left_filled_state(n_modes, n_photons),
            n_photons=n_photons,
            measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
            computation_space=ml.ComputationSpace.UNBUNCHED,
        )
        self.head = ml.LexGrouping(self.qlayer.output_size, 2)
        self.register_buffer("classifier_weights", torch.tensor([1.0, -1.0]))

    @property
    def output_size(self) -> int:
        return self.qlayer.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {x.shape[-1]}")
        probs = self.qlayer(x)
        classes = self.head(probs)
        return classes @ self.classifier_weights


# Backwards-compatible alias.
PhotonicDQML = PhotonicSingleChip


def _accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return ((pred >= 0).float() * 2 - 1 == y).float().mean().item()


def _normalise(x_train: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp_min(1e-6)
    return tuple((t - mean) / std for t in (x_train, *others))


def train_merlin_dqml(cfg: dict, run_dir: Path, dtype: torch.dtype) -> dict[str, Any]:
    dataset_cfg = SyntheticDatasetConfig(**cfg.get("dataset", {}).get("params", {}))
    model_cfg = cfg.get("model", {}).get("params", {})
    n_modes = int(model_cfg.get("n_modes", 8))
    n_photons = int(model_cfg.get("n_photons", 3))
    n_features = int(model_cfg.get("n_features", 8))
    angle_scale = float(model_cfg.get("angle_scale", 1.0))
    normalize_inputs = bool(model_cfg.get("normalize_inputs", False))

    training_cfg = cfg.get("training", {})
    n_iterations = int(training_cfg.get("n_iterations", 800))
    lr = float(training_cfg.get("lr", 0.05))
    batch_size = int(training_cfg.get("batch_size", 256))
    eval_every = int(training_cfg.get("eval_every", 50))
    seeds = list(cfg.get("seeds") or [int(cfg.get("seed", 42))])

    x_tr, y_tr, x_va, y_va, info = build_synthetic_dataset(dataset_cfg)
    x_tr_t = torch.as_tensor(x_tr, dtype=dtype)
    y_tr_t = torch.as_tensor(y_tr, dtype=dtype)
    x_va_t = torch.as_tensor(x_va, dtype=dtype)
    y_va_t = torch.as_tensor(y_va, dtype=dtype)
    if normalize_inputs:
        x_tr_t, x_va_t = _normalise(x_tr_t, x_va_t)

    summary: dict[str, Any] = {
        "pipeline": "merlin",
        "model": "PhotonicSingleChip",
        "n_modes": n_modes,
        "n_photons": n_photons,
        "n_features": n_features,
        "encoding": "angle",
        "angle_scale": angle_scale,
        "normalize_inputs": normalize_inputs,
        "computation_space": "UNBUNCHED",
        "detector_model": "threshold",
        "measurement_strategy": "PROBABILITIES",
        "postselection": "none",
        "backend": "MerLin CPU simulator (analytic, shots=0)",
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
        model = PhotonicSingleChip(
            n_modes=n_modes,
            n_photons=n_photons,
            n_features=n_features,
            angle_scale=angle_scale,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        run = {
            "seed": int(seed),
            "n_params": int(n_params),
            "output_size": int(model.output_size),
            "history": {"iteration": [], "train_loss": [], "train_acc": [], "val_acc": []},
        }
        best_val = 0.0
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
                    tr_acc = _accuracy(model(x_tr_t), y_tr_t)
                    va_acc = _accuracy(model(x_va_t), y_va_t)
                model.train()
                best_val = max(best_val, va_acc)
                run["history"]["iteration"].append(it)
                run["history"]["train_loss"].append(float(loss.item()))
                run["history"]["train_acc"].append(tr_acc)
                run["history"]["val_acc"].append(va_acc)
        run["wall_clock_seconds"] = time.time() - t0
        run["final_val_acc"] = run["history"]["val_acc"][-1]
        run["best_val_acc"] = best_val
        run["final_train_acc"] = run["history"]["train_acc"][-1]
        run["final_loss"] = run["history"]["train_loss"][-1]
        summary["runs"].append(run)
        LOGGER.info(
            "  MerLin n_modes=%d n_photons=%d seed=%d val_acc=%.4f best=%.4f "
            "params=%d time=%.1fs",
            n_modes, n_photons, seed, run["final_val_acc"], best_val,
            n_params, run["wall_clock_seconds"],
        )

    val_accs = np.array([r["final_val_acc"] for r in summary["runs"]])
    best_accs = np.array([r["best_val_acc"] for r in summary["runs"]])
    summary["mean_val_acc"] = float(val_accs.mean())
    summary["std_val_acc"] = float(val_accs.std(ddof=0))
    summary["mean_best_val_acc"] = float(best_accs.mean())
    summary["std_best_val_acc"] = float(best_accs.std(ddof=0))
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary
