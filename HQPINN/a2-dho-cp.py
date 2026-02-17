# run_cq_oscillator.py
# Classical–Quantum PINN using PennyLane + oscillator_core + shared quantum branch

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp

from core import train_oscillator_pinn
from layer_pennylane import make_device, make_quantum_block, QuantumBranch
from layer_classical import MLP  # <- reuse shared MLP

# ============================================================
#  Hyperparameters and quantum architecture
# ============================================================

lr = 0.002
n_epochs = 2000
plot_every = 100

n_qubits = 3
n_layers = 3
dtype = torch.float64


# ============================================================
#  Training data: time samples t ∈ [0, 1]
# ============================================================

t_train = pnp.linspace(0.0, 1.0, 200) # type: ignore
t_train_torch = torch.tensor(t_train, dtype=dtype).reshape(-1, 1)


# ============================================================
#  Quantum device and random seeds
# ============================================================

dev = make_device(n_qubits)

torch.manual_seed(0)
np.random.seed(0)


# ============================================================
#  Hybrid CQ_PINN
# ============================================================


class CQ_PINN(nn.Module):
    """
    Hybrid Classical–Quantum PINN:

        u(t) = u_q(t) + u_c(t)
    """

    def __init__(self) -> None:
        super().__init__()
        qblock = make_quantum_block(dev, n_qubits=n_qubits, n_layers=n_layers)
        self.branch_q = QuantumBranch(qblock, n_qubits, n_layers, dtype=dtype)
        self.branch_c = MLP(dtype=dtype)  # <- shared MLP

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.branch_q(t) + self.branch_c(t)


# ============================================================
#  Main: training via oscillator_core.train_oscillator_pinn
# ============================================================


def main():
    model = CQ_PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_oscillator_pinn(
        model=model,
        t_train=t_train_torch,
        optimizer=optimizer,
        n_epochs=n_epochs,
        plot_every=plot_every,
        out_dir="HQPINN/results",
        model_label="classical-quantum",
    )


if __name__ == "__main__":
    main()
