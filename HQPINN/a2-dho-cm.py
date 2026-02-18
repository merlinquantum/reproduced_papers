# run_cm_oscillator.py
# Classical–MerLin PINN for the damped oscillator using oscillator_core

import numpy as np
import torch
import torch.nn as nn

from core import train_oscillator_pinn
from layer_classical import MLP
from layer_merlin import make_merlin_qlayer, MerlinQuantumBranch

# ============================================================
#  Hyperparameters and quantum architecture
# ============================================================

# Diverge with lr = 0.05
lr = 0.002
n_epochs = 1801
plot_every = 100

n_qubits = 3
dtype = torch.float32


# ============================================================
#  Hybrid CM_PINN model
# ============================================================


class CM_PINN(nn.Module):
    """
    Hybrid Classical–MerLin PINN:

        u(t) = u_q(t) + u_c(t)

    where u_q(t) is the MerLin quantum branch and u_c(t) is the classical MLP.
    """

    def __init__(self, n_qubits: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        qlayer = make_merlin_qlayer(n_qubits, dtype=dtype)
        self.branch_q = MerlinQuantumBranch(qlayer, n_qubits)
        self.branch_c = MLP(dtype=dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.branch_q(t) + self.branch_c(t)


# ============================================================
#  Main: training via oscillator_core.train_oscillator_pinn
# ============================================================


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    t_train = np.linspace(0.0, 1.0, 200)
    t_train_torch = torch.tensor(t_train, dtype=dtype).reshape(-1, 1)

    model = CM_PINN(n_qubits=n_qubits, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_oscillator_pinn(
        model=model,
        t_train=t_train_torch,
        optimizer=optimizer,
        n_epochs=n_epochs,
        plot_every=plot_every,
        out_dir="HQPINN/results",
        model_label="classical-merlin",
    )


if __name__ == "__main__":
    main()
