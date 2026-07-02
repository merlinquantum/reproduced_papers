"""Distributed and quantum-communication photonic DQML schemes (angle encoding only).

Three schemes, each photonically faithful to the paper:

* ``"nc"`` — two photonic chips of ``n_modes_per_chip`` modes /
  ``n_photons_per_chip`` photons. Each chip angle-encodes
  ``n_features_per_chip`` of the input attributes on its first modes.
  The chip output (``C(m, n)`` UNBUNCHED probabilities) is grouped via
  :class:`merlin.LexGrouping` into a 2-bin distribution: this is the
  photonic analogue of the paper's QCNN pooling tree, which reduces
  4 qubits per QPU to a single readout bit.
  No inter-chip communication.
* ``"cc"`` — same two chips. Chip 0 contributes a soft bit
  ``p[b0] = LexGrouping(chip0_out)``. Two trainable feedforward phases
  ``(phi_0, phi_1)`` are angle-encoded onto an extra mode of chip 1,
  and the chip-1 output is computed for each setting; the soft bit
  mixes the two chip-1 outputs.
* ``"qc"`` — one chip of ``2 * n_modes_per_chip`` modes and
  ``2 * n_photons_per_chip`` photons (full quantum communication
  between the two halves). Angle encoding loads all 8 attributes on
  the first 8 modes of the doubled chip. Readout via
  :class:`merlin.LexGrouping` into 2 bins.

All chips use the FirstQuantumLayers-tutorial recipe with one
trainable MZI entangling layer + angle encoding + trainable rotations
+ one more trainable MZI mesh (a single mesh is photonically
universal; the extra mesh is the standard tutorial pattern).
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
    """Place photons evenly across all modes (avoids "cone light" dead phases)."""
    if n_photons < 1 or n_photons > n_modes:
        raise ValueError("require 1 <= n_photons <= n_modes")
    state = [0] * n_modes
    for i in range(n_photons):
        pos = (i * n_modes) // n_photons
        while state[pos] == 1:
            pos += 1
        state[pos] = 1
    return state


def _build_chip(n_modes: int, n_photons: int, n_features: int,
                angle_scale: float = 1.0) -> ml.QuantumLayer:
    """Construct one MerLin chip using the tutorial recipe."""
    builder = ml.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer(trainable=True)
    builder.add_angle_encoding(modes=list(range(n_features)), scale=angle_scale)
    builder.add_rotations(trainable=True)
    builder.add_entangling_layer(trainable=True)
    state = _evenly_spread_state(n_modes, n_photons)
    return ml.QuantumLayer(
        input_size=n_features,
        builder=builder,
        input_state=state,
        n_photons=n_photons,
        measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
        computation_space=ml.ComputationSpace.UNBUNCHED,
    )


class LearnedBitHead(nn.Module):
    """Photonic analogue of a trainable QCNN 4-to-1 pooling reduction.

    Takes the ``output_size``-dim probability distribution from a chip and
    learns a softmax 2-bin reduction (``P[b=0]``, ``P[b=1]``). The
    pre-softmax logits are ``W p + b`` where ``W`` and ``b`` are
    trainable. Compared to a fixed ``LexGrouping``, this lets the head
    adapt the partition to the data; compared to a full linear head over
    the outer product of two chips, it preserves the structure of the
    paper's interpret function (one bit per chip).
    """

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.head = nn.Linear(output_size, 2, bias=True)
        nn.init.normal_(self.head.weight, std=0.05)
        nn.init.zeros_(self.head.bias)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.head(probs), dim=-1)


def _grouping_into_two(output_size: int) -> LearnedBitHead:
    return LearnedBitHead(output_size)


class MerLinDistributedDQML(nn.Module):
    """Photonic NC / CC / QC DQML model."""

    def __init__(
        self,
        scheme: str = "cc",
        n_modes_per_chip: int = 8,
        n_photons_per_chip: int = 3,
        n_features_per_chip: int = 4,
    ) -> None:
        super().__init__()
        scheme = scheme.lower()
        if scheme not in {"nc", "cc", "qc"}:
            raise ValueError(f"unknown scheme '{scheme}'")
        self.scheme = scheme
        self.n_modes_per_chip = n_modes_per_chip
        self.n_photons_per_chip = n_photons_per_chip
        self.n_features_per_chip = n_features_per_chip

        if scheme == "qc":
            # Single chip of doubled size, angle encoding all 8 attributes.
            self.chip = _build_chip(
                n_modes=2 * n_modes_per_chip,
                n_photons=2 * n_photons_per_chip,
                n_features=2 * n_features_per_chip,
            )
            self.group = _grouping_into_two(self.chip.output_size)
            # Fixed +1/-1 head (parity-like) to match single-chip baseline.
            self.register_buffer("class_weights", torch.tensor([1.0, -1.0]))
        else:
            self.chip0 = _build_chip(
                n_modes=n_modes_per_chip,
                n_photons=n_photons_per_chip,
                n_features=n_features_per_chip,
            )
            self.group0 = _grouping_into_two(self.chip0.output_size)
            chip1_modes = n_modes_per_chip + (1 if scheme == "cc" else 0)
            chip1_features = n_features_per_chip + (1 if scheme == "cc" else 0)
            self.chip1 = _build_chip(
                n_modes=chip1_modes,
                n_photons=n_photons_per_chip,
                n_features=chip1_features,
            )
            self.group1 = _grouping_into_two(self.chip1.output_size)
            # Interpret-function weights (Eq. 3 in the paper).
            self.interpret_weights = nn.Parameter(torch.tensor([1.0, -1.0, -1.0, 1.0]))
            if scheme == "cc":
                # Two trainable feedforward phases (one per chip-0 outcome bit).
                self.feedforward_phases = nn.Parameter(torch.zeros(2))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_features = 2 * self.n_features_per_chip
        if x.shape[-1] != expected_features:
            raise ValueError(
                f"expected {expected_features} features, got {x.shape[-1]}"
            )

        if self.scheme == "qc":
            probs = self.chip(x)
            grouped = self.group(probs)  # (B, 2)
            return grouped @ self.class_weights

        x0 = x[..., : self.n_features_per_chip]
        x1 = x[..., self.n_features_per_chip :]
        p0_full = self.chip0(x0)
        p0 = self.group0(p0_full)  # (B, 2)

        if self.scheme == "cc":
            ff0 = self.feedforward_phases[0].expand(x1.shape[0], 1)
            ff1 = self.feedforward_phases[1].expand(x1.shape[0], 1)
            p1_g0 = self.group1(self.chip1(torch.cat([x1, ff0], dim=-1)))  # (B, 2)
            p1_g1 = self.group1(self.chip1(torch.cat([x1, ff1], dim=-1)))  # (B, 2)
            # Conditional mixing: p_b1[b1] = p0[0]*p1_g0[b1] + p0[1]*p1_g1[b1]
            # Joint: P[b0, b1] = p0[b0] * (b0==0 ? p1_g0 : p1_g1)[b1]
            P00 = p0[:, 0] * p1_g0[:, 0]
            P01 = p0[:, 0] * p1_g0[:, 1]
            P10 = p0[:, 1] * p1_g1[:, 0]
            P11 = p0[:, 1] * p1_g1[:, 1]
        else:  # nc
            p1_full = self.chip1(x1)
            p1 = self.group1(p1_full)  # (B, 2)
            P00 = p0[:, 0] * p1[:, 0]
            P01 = p0[:, 0] * p1[:, 1]
            P10 = p0[:, 1] * p1[:, 0]
            P11 = p0[:, 1] * p1[:, 1]

        joint = torch.stack([P00, P01, P10, P11], dim=-1)
        return joint @ self.interpret_weights


def _accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return ((pred >= 0).float() * 2 - 1 == y).float().mean().item()


def _normalise(x_train: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp_min(1e-6)
    return tuple((t - mean) / std for t in (x_train, *others))


def train_merlin_distributed(cfg: dict, run_dir: Path, dtype: torch.dtype) -> dict[str, Any]:
    dataset_cfg = SyntheticDatasetConfig(**cfg.get("dataset", {}).get("params", {}))
    model_cfg = cfg.get("model", {}).get("params", {})
    scheme = str(model_cfg.get("scheme", "cc")).lower()
    n_modes = int(model_cfg.get("n_modes_per_chip", 8))
    n_photons = int(model_cfg.get("n_photons_per_chip", 3))
    n_features_per_chip = int(model_cfg.get("n_features_per_chip", 4))
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
        "pipeline": "merlin_distributed",
        "model": "MerLinDistributedDQML",
        "scheme": scheme,
        "n_modes_per_chip": n_modes,
        "n_photons_per_chip": n_photons,
        "n_features_per_chip": n_features_per_chip,
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
        model = MerLinDistributedDQML(
            scheme=scheme,
            n_modes_per_chip=n_modes,
            n_photons_per_chip=n_photons,
            n_features_per_chip=n_features_per_chip,
        )
        n_params = model.num_parameters()
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        run = {
            "seed": int(seed),
            "n_params": int(n_params),
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
            "  MerLin-Dist scheme=%s seed=%d val_acc=%.4f best=%.4f params=%d time=%.1fs",
            scheme, seed, run["final_val_acc"], best_val, n_params,
            run["wall_clock_seconds"],
        )

    val_accs = np.array([r["final_val_acc"] for r in summary["runs"]])
    best_accs = np.array([r["best_val_acc"] for r in summary["runs"]])
    summary["mean_val_acc"] = float(val_accs.mean())
    summary["std_val_acc"] = float(val_accs.std(ddof=0))
    summary["mean_best_val_acc"] = float(best_accs.mean())
    summary["std_best_val_acc"] = float(best_accs.std(ddof=0))
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary
