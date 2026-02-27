# a2_dho_ii.py
# Interferometer-Interferometer PINN for the damped oscillator using oscillator_core + merlin_quantum

import os
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for batch PDF export
matplotlib.use("Agg")

import torch
import torch.nn as nn

from ..config import DHO_N_EPOCHS, DHO_PLOT_EVERY, DHO_LR
from ..utils import (
    make_time_grid,
    make_optimizer,
)
from .core_a2_dho import train_oscillator_pinn, u_exact
from ..run_common import run_series_inference_mode
from ..layer_merlin import make_interf_qlayer, BranchMerlin


# ============================================================
#  MM_PINN model: two MerLin quantum branches
# ============================================================


class MM_PINN(nn.Module):
    """
    Interferometer-Interferometer PINN:

        u(t) = u_q1(t) + u_q2(t)

    Each branch uses its own QuantumLayer instance → independent parameters.
    """

    def __init__(self, processor=None) -> None:
        super().__init__()

        # Two distinct quantum branches with independent parameters
        self.branch1 = BranchMerlin(
            make_interf_qlayer(n_photons=1),
            processor=processor,
            feature_map_kind="dho",
        )
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=1),
            processor=processor,
            feature_map_kind="dho",
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Forward pass: sum of the two interferometer branches
        return self.branch1(t) + self.branch2(t)


def plot_model_prediction(u_pred, u_ex, t, save_path="HQPINN/DHO/results/"):
    plt.figure(figsize=(10, 6))
    plt.plot(t.cpu().numpy(), u_pred, label="Prediction PINN", lw=2)
    plt.plot(t.cpu().numpy(), u_ex, "--", label="Exact solution", lw=2)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("DHO – Interferometer–Interferometer PINN")
    plt.grid(True)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"dho_ii_plot_{timestamp}.png")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {save_path}")


def run(mode="train", backend="sim:ascella") -> None:
    """
    mode = "train" : train the model from scratch and save the checkpoint
    mode = "run"   : load the latest checkpoint and run inference (not implemented here, but can be added)
    mode = "remote" : load and run in remote
    """
    torch.manual_seed(0)
    np.random.seed(0)

    ckpt_dir = "HQPINN/DHO/models/"
    case_prefix = "dho_ii"

    # ======================
    #  MODE TRAIN
    # ======================
    if mode == "train":
        print("=== TRAINING MODE ===")

        model = MM_PINN()

        train_oscillator_pinn(
            model=model,
            t_train=make_time_grid(),
            optimizer=make_optimizer(model, lr=DHO_LR),
            n_epochs=DHO_N_EPOCHS,
            plot_every=DHO_PLOT_EVERY,
            out_dir="HQPINN/DHO/results",
            model_label="Interferometer-Interferometer",
        )

        # === Save model ===
        os.makedirs(ckpt_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_path = os.path.join(ckpt_dir, f"{case_prefix}_{timestamp}.pt")
        torch.save(model.state_dict(), ckpt_path)

        print(f"Model saved to: {ckpt_path}")

    # ======================
    #  MODE RUN
    # ======================
    elif mode == "run":
        print("=== RUN MODE ===")
        run_series_inference_mode(
            mode="run",
            backend=backend,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=MM_PINN,
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=plot_model_prediction,
        )

    # ======================
    #  MODE RUN REMOTE
    # ======================
    elif mode == "remote":
        run_series_inference_mode(
            mode="remote",
            backend=backend,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=MM_PINN,
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=plot_model_prediction,
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
