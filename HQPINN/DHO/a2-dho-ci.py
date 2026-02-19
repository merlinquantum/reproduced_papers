# MerLin–MerLin PINN for the damped oscillator using oscillator_core + merlin_quantum

import numpy as np
import torch
import torch.nn as nn

from config import DHO_N_EPOCHS, DHO_PLOT_EVERY, DHO_LR
from utils import make_time_grid, make_optimizer
from core import train_oscillator_pinn
from layer_merlin import make_interf_qlayer, BranchMerlin
from layer_classical import BranchPyTorch


# ============================================================
#  MM_PINN model: two MerLin quantum branches
# ============================================================


class MM_PINN(nn.Module):
    """
    MerLin–MerLin PINN:

        u(t) = u_q1(t) + u_q2(t)

    Each branch uses its own QuantumLayer instance → independent parameters.
    """

    def __init__(self) -> None:
        super().__init__()

        # Two distinct quantum branches
        self.branch1 = BranchMerlin(make_interf_qlayer())
        self.branch2 = BranchPyTorch()  # Classical MLP branch

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.branch1(t) + self.branch2(t)


# ============================================================
#  Main: training via oscillator_core.train_oscillator_pinn
# ============================================================


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    model = MM_PINN()

    train_oscillator_pinn(
        model=model,
        t_train=make_time_grid(),
        optimizer=make_optimizer(model, lr=DHO_LR),
        n_epochs=DHO_N_EPOCHS,
        plot_every=DHO_PLOT_EVERY,
        out_dir="HQPINN/results",
        model_label="classical-interferometer",
    )


if __name__ == "__main__":
    main()
