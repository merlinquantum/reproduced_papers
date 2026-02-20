# Classical–Interferometer PINN for the damped oscillator

import numpy as np
import torch
import torch.nn as nn

from ..config import DHO_N_EPOCHS, DHO_PLOT_EVERY, DHO_LR
from ..utils import make_time_grid, make_optimizer
from ..core import train_oscillator_pinn
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

    def __init__(self) -> None:
        super().__init__()

        # One MerLin quantum branch
        self.branch1 = BranchMerlin(make_interf_qlayer())
        # One classical MLP branch
        self.branch2 = BranchPyTorch()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Forward pass: sum of MerLin and classical branches
        return self.branch1(t) + self.branch2(t)


def run() -> None:
    """Run the Classical–Interferometer DHO PINN experiment."""
    torch.manual_seed(0)
    np.random.seed(0)

    model = CI_PINN()
    train_oscillator_pinn(
        model=model,
        t_train=make_time_grid(),
        optimizer=make_optimizer(model, lr=DHO_LR),
        n_epochs=DHO_N_EPOCHS,
        plot_every=DHO_PLOT_EVERY,
        out_dir="HQPINN/DHO/results",
        model_label="Classical-Interferometer",
    )
