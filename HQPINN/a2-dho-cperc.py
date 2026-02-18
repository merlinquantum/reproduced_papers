# a2-dho-cperc.py
# Classical–Perceval PINN with a quantum branch using MerLin QuantumLayer and a classical MLP branch

import numpy as np
import torch
import torch.nn as nn

from config import N_EPOCHS, PLOT_EVERY, LR
from utils import make_time_grid, make_optimizer
from core import train_oscillator_pinn
from layer_merlin import make_perceval_qlayer, BranchMerlin
from layer_classical import BranchPyTorch


# ============================================================
#  Hybrid CM_PINN model
# ============================================================


class CM_PINN(nn.Module):
    """
    Hybrid Classical–MerLin PINN:

        u(t) = u_m(t) + u_c(t)

    where u_m(t) is the MerLin quantum branch and u_c(t) is the classical MLP.
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch_q = BranchMerlin(make_perceval_qlayer())
        self.branch_c = BranchPyTorch()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.branch_q(t) + self.branch_c(t)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    model = CM_PINN()

    train_oscillator_pinn(
        model=model,
        t_train=make_time_grid(),
        optimizer=make_optimizer(model, lr=LR),
        n_epochs=N_EPOCHS,
        plot_every=PLOT_EVERY,
        out_dir="HQPINN/results",
        model_label="classical-perceval",
    )


if __name__ == "__main__":
    main()
