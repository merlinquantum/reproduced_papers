# run_qq_oscillator.py
# Quantum–Quantum PINN using PennyLane + oscillator_core + shared quantum branch

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp

from core import train_oscillator_pinn
from layer_pennylane import make_device, make_quantum_block, QuantumBranch

# ============================================================
#  Hyperparameters and quantum architecture
# ============================================================

lr = 0.05
n_epochs = 2800
plot_every = 100

n_qubits = 3
n_layers = 3
dtype = torch.float32


# ============================================================
#  Training data: time samples t ∈ [0, 1]
# ============================================================

t_train = pnp.linspace(0.0, 1.0, 200)
t_train_torch = torch.tensor(t_train, dtype=dtype).reshape(-1, 1)


# ============================================================
#  Quantum device and random seeds
# ============================================================

dev = make_device(n_qubits)

torch.manual_seed(0)
np.random.seed(0)


# ============================================================
#  QQ-PINN model (two quantum branches)
# ============================================================


class QQ_PINN(nn.Module):
    """
    Physics-Informed model: sum of two independent quantum branches.

        u(t) = u_q1(t) + u_q2(t)
    """

    def __init__(self) -> None:
        super().__init__()
        qblock = make_quantum_block(dev, n_qubits=n_qubits, n_layers=n_layers)

        # Two distinct branches => two independent parameter sets
        self.branch1 = QuantumBranch(qblock, n_qubits, n_layers, dtype=dtype)
        self.branch2 = QuantumBranch(qblock, n_qubits, n_layers, dtype=dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.branch1(t) + self.branch2(t)


# ============================================================
#  Main: training via oscillator_core.train_oscillator_pinn
# ============================================================


def main():
    model = QQ_PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_oscillator_pinn(
        model=model,
        t_train=t_train_torch,
        optimizer=optimizer,
        n_epochs=n_epochs,
        plot_every=plot_every,
        out_dir="HQPINN/results",
        model_label="quantum-quantum",
    )


if __name__ == "__main__":
    main()
