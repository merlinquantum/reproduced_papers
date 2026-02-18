# Interferometer-Interferometer PINN for the damped oscillator using oscillator_core + merlin_quantum

import numpy as np
import torch
import torch.nn as nn

from config import N_EPOCHS, PLOT_EVERY, LR
from utils import make_time_grid, make_optimizer
from core import train_oscillator_pinn
from layer_merlin import make_interf_qlayer, BranchMerlin


# ============================================================
#  MM_PINN model: two MerLin quantum branches
# ============================================================


class MM_PINN(nn.Module):
    """
    Interferometer-Interferometer PINN:

        u(t) = u_q1(t) + u_q2(t)

    Each branch uses its own QuantumLayer instance → independent parameters.
    """

    def __init__(self) -> None:
        super().__init__()

        # Two distinct quantum branches
        self.branch1 = BranchMerlin(make_interf_qlayer())
        self.branch2 = BranchMerlin(make_interf_qlayer())

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
        optimizer=make_optimizer(model, lr=LR),
        n_epochs=N_EPOCHS,
        plot_every=PLOT_EVERY,
        out_dir="HQPINN/results",
        model_label="interferometer-interferometer",
    )


if __name__ == "__main__":
    main()
