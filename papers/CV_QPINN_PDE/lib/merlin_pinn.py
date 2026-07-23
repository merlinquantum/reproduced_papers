"""MerLin photonic linear-optics adaptation of the consistency-loss PINN.

The original paper builds the QPINN on a *continuous-variable* photonic
architecture (squeezing, displacement, Kerr non-Gaussian gate). MerLin
targets a different photonic regime: linear-optical interferometers with
discrete photon-number measurement, where squeezing and non-Gaussian
gates are not native. A faithful CV port of the paper is therefore not
meaningful in MerLin.

What we do here is reproduce the *consistency-loss* idea on a MerLin
linear-optics backbone: the network input is angle-encoded into a
trainable interferometer, two qumode-like output probabilities are
mapped to the (u, ux) pair used by the consistency loss, and training
proceeds with the same `lib.losses.poisson_total_loss` as the QPINN
experiment. This is honestly a *photonic adaptation* of the idea, not a
reproduction of the paper's CV architecture.

The MerLin model is wrapped in a small `torch.nn.Module` so that the
training loop in `lib.training` does not need any MerLin-specific glue.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import merlin as ml
import torch
import torch.nn as nn

from .data import PoissonProblem, sobol_1d
from .losses import poisson_total_loss
from .metrics import summarise
from .training import train_poisson

_LOG = logging.getLogger(__name__)


class MerLinPINN(nn.Module):
    """Photonic linear-optics PINN that returns (u, ux, trace).

    The input scalar is encoded by `add_angle_encoding` on the first
    ``input_modes`` modes (the paper-style displacement is replaced by
    an angle encoding because MerLin's circuit builder is linear-optical).
    The output probability distribution over UNBUNCHED detection outcomes
    is reduced to two scalars via two trainable linear heads — one for u,
    one for ux. The trace output is constant 1.0 (probability totals are
    intrinsic to the measurement statistics, so a separate trace-loss term
    is not needed).
    """

    def __init__(self, n_modes: int = 6, input_modes: tuple[int, ...] = (0, 1, 2),
                 entangling_layers: int = 3, n_photons: int = 3,
                 scale: float = math.pi, seed: int = 42) -> None:
        super().__init__()
        # Parameter init draws from torch's global RNG; callers seed it via
        # torch.manual_seed (see run_merlin_poisson). `seed` is kept for the
        # config snapshot only.
        del seed
        builder = ml.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer()
        builder.add_angle_encoding(modes=list(input_modes), scale=float(scale))
        for _ in range(entangling_layers):
            builder.add_entangling_layer()
        input_state = [0] * n_modes
        # Place photons every other mode (dual-rail-style spread).
        for i in range(n_photons):
            input_state[min(2 * i, n_modes - 1)] = 1
        # merlin >= 0.4: the computation space is owned by the measurement
        # strategy factory and the photon count is inferred from input_state.
        self.qlayer = ml.QuantumLayer(
            builder=builder,
            input_state=input_state,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.UNBUNCHED,
            ),
            dtype=torch.float64,
        )
        out_size = self.qlayer.output_size
        # Two trainable linear heads producing scalar outputs.
        self.head_u = nn.Linear(out_size, 1, dtype=torch.float64)
        self.head_ux = nn.Linear(out_size, 1, dtype=torch.float64)
        self.input_modes = input_modes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 1:
            x_in = x.unsqueeze(-1).expand(x.shape[0], len(self.input_modes))
        else:
            raise ValueError("MerLin Poisson model expects 1D inputs")
        probs = self.qlayer(x_in)
        u = self.head_u(probs).squeeze(-1)
        ux = self.head_ux(probs).squeeze(-1)
        trace = torch.ones_like(u)
        return u, ux, trace

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def run_merlin_poisson(cfg: dict, run_dir: Path) -> dict:
    problem = PoissonProblem()
    torch.manual_seed(cfg.get("seed", 42))
    model_cfg = cfg["model"]
    model = MerLinPINN(
        n_modes=model_cfg.get("n_modes", 6),
        input_modes=tuple(model_cfg.get("input_modes", [0, 1, 2])),
        entangling_layers=model_cfg.get("entangling_layers", 3),
        n_photons=model_cfg.get("n_photons", 3),
        scale=float(model_cfg.get("encoding_scale", math.pi)),
        seed=cfg.get("seed", 42),
    )
    _LOG.info("MerLin PINN trainable params: %d, qlayer output size: %d",
              model.n_trainable(), model.qlayer.output_size)
    n_train = cfg["training"]["collocation_points"]
    x_coll = sobol_1d(n_train, problem.x_min, problem.x_max, seed=cfg.get("seed", 42))
    # Scale collocation points to fit roughly in [-1, 1] for the angle encoding;
    # the angle-encoding scale picks up the rest.
    x_bc_left = torch.tensor([problem.x_min], dtype=torch.float64)
    x_bc_right = torch.tensor([problem.x_max], dtype=torch.float64)
    lambdas = cfg["training"]["lambdas"]
    history_path = run_dir / "history.json"
    schedule = cfg["training"].get("lr_schedule")
    result = train_poisson(
        model, x_coll, (x_bc_left, x_bc_right),
        loss_fn=poisson_total_loss,
        lr=cfg["training"]["lr"], epochs=cfg["training"]["epochs"],
        lambdas=lambdas, history_path=history_path,
        log_every=cfg["training"].get("log_every", 50),
        lr_schedule=schedule,
    )
    x_eval = torch.linspace(problem.x_min, problem.x_max, 200, dtype=torch.float64)
    with torch.no_grad():
        u_pred, _, _ = model(x_eval)
    u_ref = problem.analytic(x_eval)
    metrics = summarise(u_pred, u_ref)
    artefacts = {
        "x": x_eval.tolist(),
        "u_pred": u_pred.tolist(),
        "u_ref": u_ref.tolist(),
    }
    (run_dir / "predictions.json").write_text(json.dumps(artefacts))
    summary = {
        "experiment": "poisson_merlin",
        "n_params": model.n_trainable(),
        "qlayer_output_size": int(model.qlayer.output_size),
        "metrics": metrics,
        "best": result["best"],
        "wall_time_sec": result["wall_time_sec"],
        "merlin_hardware": {
            "computation_space": "UNBUNCHED",
            "detector_model": "threshold",
            "n_photons": model_cfg.get("n_photons", 3),
            "n_modes": model_cfg.get("n_modes", 6),
            "input_state": list(model.qlayer.input_state.tolist()
                                if isinstance(model.qlayer.input_state, torch.Tensor)
                                else model.qlayer.input_state),
            "encoding": "angle",
            "encoding_modes": list(model.input_modes),
            "encoding_scale": float(model_cfg.get("encoding_scale", math.pi)),
            "measurement_strategy": "MeasurementStrategy.probs(UNBUNCHED)",
            "postselection": "none",
            "simulator": "MerLin CPU simulator (analytic, shots=0)",
        },
        "cfg": cfg,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    torch.save(model.state_dict(), run_dir / "model.pt")
    return summary


__all__ = ["MerLinPINN", "run_merlin_poisson"]
