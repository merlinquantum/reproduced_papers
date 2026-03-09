# Classical–Interferometer PINN for the damped oscillator

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ..config import DHO_N_EPOCHS, DHO_PLOT_EVERY, DHO_LR
from ..utils import make_time_grid, make_optimizer
from .core_a2_dho import train_oscillator_pinn, u_exact
from ..run_common import run_series_inference_mode
from ..layer_merlin import make_interf_qlayer, BranchMerlin
from ..layer_classical import BranchPyTorch


# ============================================================
#  CI_PINN model: MerLin quantum + classical branch
# ============================================================


class CI_PINN(nn.Module):
    """
    Classical–Interferometer PINN:

        u(t) = u_c(t) + u_q(t)
    Each branch uses its own QuantumLayer instance → independent parameters.
    """

    def __init__(self, processor=None) -> None:
        super().__init__()

        # One MerLin quantum branch
        self.branch1 = BranchMerlin(
            make_interf_qlayer(n_photons=1),
            processor=processor,
            feature_map_kind="dho",
        )
        # One classical MLP branch
        self.branch2 = BranchPyTorch()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Forward pass: sum of MerLin and classical branches
        return self.branch1(t) + self.branch2(t)


def plot_model_prediction(u_pred, u_ex, t, save_path="HQPINN/DHO/results/dho_ci/"):
    plt.figure(figsize=(10, 6))
    plt.plot(t.cpu().numpy(), u_pred, label="Prediction PINN", lw=2)
    plt.plot(t.cpu().numpy(), u_ex, "--", label="Exact solution", lw=2)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("DHO - Classical-Interferometer PINN")
    plt.grid(True)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    png_path = os.path.join(save_path, f"dho_ci_plot_{timestamp}.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {png_path}")


def run(mode="train", backend="sim:ascella") -> None:
    """Run the Classical–Interferometer DHO PINN experiment."""
    torch.manual_seed(0)
    np.random.seed(0)
    ckpt_dir = "HQPINN/DHO/models"
    case_prefix = "dho_ci"

    if mode == "train":
        model = CI_PINN()
        train_oscillator_pinn(
            model=model,
            t_train=make_time_grid(),
            optimizer=make_optimizer(model, lr=DHO_LR),
            n_epochs=DHO_N_EPOCHS,
            plot_every=DHO_PLOT_EVERY,
            out_dir=f"HQPINN/DHO/results/{case_prefix}",
            model_label="ci",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_path = os.path.join(ckpt_dir, f"{case_prefix}_{timestamp}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Model saved to: {ckpt_path}")

    elif mode == "run":
        run_series_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=CI_PINN,
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=plot_model_prediction,
        )

    elif mode == "remote":
        run_series_inference_mode(
            mode="remote",
            backend=backend,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=CI_PINN,
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=plot_model_prediction,
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
