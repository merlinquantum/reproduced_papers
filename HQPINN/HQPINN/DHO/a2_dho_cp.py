# a2_dho_cp.py
# Classical–PennyLane PINN with a quantum branch and a classical MLP branch

import numpy as np
import torch
import torch.nn as nn

from ..config import DHO_N_EPOCHS, DHO_PLOT_EVERY, DHO_LR
from ..utils import make_time_grid, make_optimizer
from ..core import train_oscillator_pinn
from ..layer_pennylane import make_quantum_block, BranchPennylane
from ..layer_classical import BranchPyTorch


class CQ_PINN(nn.Module):
    """
    Hybrid Classical–Quantum PINN:

        u(t) = u_q(t) + u_c(t)
    """

    def __init__(self) -> None:
        super().__init__()
        qblock = make_quantum_block()
        self.branch_q = BranchPennylane(qblock)
        self.branch_c = BranchPyTorch()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Forward pass: sum of quantum and classical branches
        return self.branch_q(t) + self.branch_c(t)


def run() -> None:
    """Run the Classical–PennyLane DHO PINN experiment."""
    torch.manual_seed(0)
    np.random.seed(0)

    model = CQ_PINN()

    train_oscillator_pinn(
        model=model,
        t_train=make_time_grid(),
        optimizer=make_optimizer(model, lr=DHO_LR),
        n_epochs=DHO_N_EPOCHS,
        plot_every=DHO_PLOT_EVERY,
        out_dir="HQPINN/DHO/results",
        model_label="Classical-PennyLane",
    )
