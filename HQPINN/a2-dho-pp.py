# a2-dho-pp.py
# PennyLane–PennyLane PINN with two parallel quantum branches

import torch
import torch.nn as nn

from config import N_EPOCHS, PLOT_EVERY, LR
from utils import make_time_grid, make_optimizer
from core import train_oscillator_pinn
from layer_pennylane import make_quantum_block, BranchPennylane


class QQ_PINN(nn.Module):
    """
    Physics-Informed model: sum of two independent quantum branches.

        u(t) = u_q1(t) + u_q2(t)
    """

    def __init__(self) -> None:
        super().__init__()
        qblock = make_quantum_block()

        # Two distinct branches => two independent parameter sets
        self.branch1 = BranchPennylane(qblock)
        self.branch2 = BranchPennylane(qblock)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.branch1(t) + self.branch2(t)


def main():
    model = QQ_PINN()

    train_oscillator_pinn(
        model=model,
        t_train=make_time_grid(),
        optimizer=make_optimizer(model, lr=LR),
        n_epochs=N_EPOCHS,
        plot_every=PLOT_EVERY,
        out_dir="HQPINN/results",
        model_label="pennylane-pennylane",
    )


if __name__ == "__main__":
    main()
