# a2-dho-pp.py
# PennyLane–PennyLane PINN with two parallel quantum branches

import torch
import torch.nn as nn

from ..config import DHO_N_EPOCHS, DHO_PLOT_EVERY, DHO_LR, N_LAYERS
from ..utils import make_time_grid, make_optimizer
from .core_a2_dho import train_oscillator_pinn
from ..layer_pennylane import make_quantum_block, dho_feature_map, BranchPennylane


class PP_PINN(nn.Module):
    """
    Physics-Informed model: sum of two independent quantum branches.

        u(t) = u_q1(t) + u_q2(t)
    """

    def __init__(self) -> None:
        super().__init__()
        qblock1 = make_quantum_block()
        qblock2 = make_quantum_block()

        # Two distinct branches => two independent parameter sets
        self.branch1 = BranchPennylane(
            qblock1,
            feature_map=dho_feature_map,
            output_as_column=True,
            n_layers=N_LAYERS,
        )
        self.branch2 = BranchPennylane(
            qblock2,
            feature_map=dho_feature_map,
            output_as_column=True,
            n_layers=N_LAYERS,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.branch1(t) + self.branch2(t)


def run():
    model = PP_PINN()

    train_oscillator_pinn(
        model=model,
        t_train=make_time_grid(),
        optimizer=make_optimizer(model, lr=DHO_LR),
        n_epochs=DHO_N_EPOCHS,
        plot_every=DHO_PLOT_EVERY,
        out_dir="HQPINN/DHO/results",
        model_label="PennyLane-PennyLane",
    )
