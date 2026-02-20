# a2_dho_perc_perc.py
# MerLin–MerLin PINN with two parallel quantum branches using MerLin QuantumLayer

import numpy as np
import torch
import torch.nn as nn

from ..config import DHO_N_EPOCHS, DHO_PLOT_EVERY, DHO_LR
from ..utils import make_time_grid, make_optimizer
from .core_a2_dho import train_oscillator_pinn
from ..layer_merlin import make_perceval_qlayer, BranchMerlin


# ============================================================
#  MM_PINN model: two Perceval quantum branches
# ============================================================


class MM_PINN(nn.Module):
    """
    Perceval–Perceval PINN:

        u(t) = u_q1(t) + u_q2(t)

    Each branch uses its own QuantumLayer instance → independent parameters.
    """

    def __init__(self) -> None:
        super().__init__()

        # Two distinct quantum branches with independent parameters
        self.branch1 = BranchMerlin(make_perceval_qlayer())
        self.branch2 = BranchMerlin(make_perceval_qlayer())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Forward pass: sum of the two Perceval branches
        return self.branch1(t) + self.branch2(t)


def run() -> None:
    """Run the Perceval–Perceval DHO PINN experiment."""
    torch.manual_seed(0)
    np.random.seed(0)

    model = MM_PINN()

    train_oscillator_pinn(
        model=model,
        t_train=make_time_grid(),
        optimizer=make_optimizer(model, lr=DHO_LR),
        n_epochs=DHO_N_EPOCHS,
        plot_every=DHO_PLOT_EVERY,
        out_dir="HQPINN/DHO/results",
        model_label="Perceval-Perceval",
    )
