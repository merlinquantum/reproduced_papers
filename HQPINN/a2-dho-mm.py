# run_mm_oscillator.py
# MerLin–MerLin PINN for the damped oscillator using oscillator_core + merlin_quantum

import numpy as np
import torch
import torch.nn as nn

from core import train_oscillator_pinn
from layer_merlin import make_merlin_qlayer, MerlinQuantumBranch

# ============================================================
#  Hyperparameters and quantum architecture
# ============================================================

lr = 0.002
n_epochs = 2000
plot_every = 100

n_qubits = 3
dtype = torch.float32


# ============================================================
#  MM_PINN model: two MerLin quantum branches
# ============================================================


class MM_PINN(nn.Module):
    """
    MerLin–MerLin PINN:

        u(t) = u_q1(t) + u_q2(t)

    Each branch uses its own QuantumLayer instance → independent parameters.
    """

    def __init__(self, n_qubits: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()

        # Two independent QuantumLayers (very important)
        qlayer1 = make_merlin_qlayer(n_qubits, dtype=dtype)
        qlayer2 = make_merlin_qlayer(n_qubits, dtype=dtype)

        # Two distinct quantum branches
        self.branch1 = MerlinQuantumBranch(qlayer1, n_qubits)
        self.branch2 = MerlinQuantumBranch(qlayer2, n_qubits)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.branch1(t) + self.branch2(t)


# ============================================================
#  Main: training via oscillator_core.train_oscillator_pinn
# ============================================================


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    t_train = np.linspace(0.0, 1.0, 200)
    t_train_torch = torch.tensor(t_train, dtype=dtype).reshape(-1, 1)

    model = MM_PINN(n_qubits=n_qubits, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_oscillator_pinn(
        model=model,
        t_train=t_train_torch,
        optimizer=optimizer,
        n_epochs=n_epochs,
        plot_every=plot_every,
        out_dir="HQPINN/results",
        model_label="merlin-merlin",
    )


if __name__ == "__main__":
    main()
