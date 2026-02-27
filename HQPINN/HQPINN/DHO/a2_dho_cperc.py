# a2_dho_cperc.py
# Classical–Perceval PINN with a quantum branch using MerLin QuantumLayer and a classical MLP branch

import numpy as np
import torch
import torch.nn as nn

from ..config import DHO_N_EPOCHS, DHO_PLOT_EVERY, DHO_LR
from ..utils import make_time_grid, make_optimizer
from .core_a2_dho import train_oscillator_pinn
from ..layer_merlin import make_perceval_qlayer, BranchMerlin
from ..layer_classical import BranchPyTorch


# ============================================================
#  Hybrid CM_PINN model
# ============================================================


class CM_PINN(nn.Module):
    """
    Hybrid Classical–Perceval PINN:

        u(t) = u_m(t) + u_c(t)

    where u_m(t) is the MerLin quantum branch and u_c(t) is the classical MLP.
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch_q = BranchMerlin(make_perceval_qlayer(), feature_map_kind="dho")
        self.branch_c = BranchPyTorch()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Forward pass: sum of MerLin and classical branches
        return self.branch_q(t) + self.branch_c(t)


def run() -> None:
    """Run the Classical–Perceval DHO PINN experiment."""
    torch.manual_seed(0)
    np.random.seed(0)

    model = CM_PINN()

    train_oscillator_pinn(
        model=model,
        t_train=make_time_grid(),
        optimizer=make_optimizer(model, lr=DHO_LR),
        n_epochs=DHO_N_EPOCHS,
        plot_every=DHO_PLOT_EVERY,
        out_dir="HQPINN/DHO/results",
        model_label="Classical-Perceval",
    )
