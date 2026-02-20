# a2_dho_cc.py
# Classical–Classical PINN with two parallel MLP branches

import torch
import torch.nn as nn

# Relative imports because this file is inside HQPINN/HQPINN/DHO/
from ..config import DHO_LR, DHO_N_EPOCHS, DHO_PLOT_EVERY, DTYPE
from ..utils import make_time_grid, make_optimizer
from ..core import train_oscillator_pinn
from ..layer_classical import BranchPyTorch


class CC_PINN(nn.Module):
    """
    Classical–Classical PINN with two parallel MLP branches:
        u(t) = u_c1(t) + u_c2(t)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch1 = BranchPyTorch()
        self.branch2 = BranchPyTorch()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Forward pass: sum of the two classical branches
        return self.branch1(t) + self.branch2(t)


def run():
    """Run the Classical–Classical DHO PINN experiment."""
    torch.manual_seed(0)

    model = CC_PINN()

    train_oscillator_pinn(
        model=model,
        t_train=make_time_grid(),
        optimizer=make_optimizer(model, DHO_LR),
        n_epochs=DHO_N_EPOCHS,
        plot_every=DHO_PLOT_EVERY,
        out_dir="HQPINN/DHO/results",
        model_label="classical-classical",
    )
