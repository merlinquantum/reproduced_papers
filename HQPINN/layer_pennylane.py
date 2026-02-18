# layer_pennylane.py

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from config import N_QUBITS, N_LAYERS, DTYPE


def make_device() -> qml.Device:  # type: ignore
    """Create a PennyLane default.qubit device."""
    return qml.device("default.qubit", wires=N_QUBITS)


# ============================================================
#  Quantum circuit building blocks
# ============================================================


def ansatz_layer(theta: torch.Tensor) -> None:
    """
    Single ansatz layer with local RZ–RX–RZ rotations and ring CNOT entanglers.

    Parameters
    ----------
    theta : (n_qubits, 3) tensor-like
        For each qubit i:
          theta[i, 0] : RZ angle
          theta[i, 1] : RX angle
          theta[i, 2] : RZ angle
    """
    for i in range(N_QUBITS):
        qml.RZ(theta[i, 0], wires=i)  # type: ignore
        qml.RX(theta[i, 1], wires=i)  # type: ignore
        qml.RZ(theta[i, 2], wires=i)  # type: ignore

    # Entangling ring
    for i in range(N_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % N_QUBITS])


def feature_layer(phi: torch.Tensor) -> None:
    """
    Feature map: angle encoding via RY rotations.

    Parameters
    ----------
    phi : (n_qubits,) tensor-like
        For qubit i, apply RY(phi[i]).
    """
    for i in range(N_QUBITS):
        qml.RY(phi[i], wires=i)  # type: ignore


# ============================================================
#  QNode factory
# ============================================================


def make_quantum_block() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

    @qml.qnode(make_device(), interface="torch")
    def quantum_block(phi: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        """
        Multi-layer quantum block used as a branch core.

        Parameters
        ----------
        phi :Feature vector.
        thetas : Trainable parameters for each ansatz layer.

        Returns
        -------
        Expectation value ⟨Z₀⟩.
        """
        for layer in range(N_LAYERS):
            ansatz_layer(thetas[layer])
            if layer < N_LAYERS - 1:
                feature_layer(phi)

        return qml.expval(qml.PauliZ(0))  # type: ignore

    return quantum_block  # type: ignore


# ============================================================
#  Quantum branch 
# ============================================================


class BranchPennylane(nn.Module):
    """
    Quantum branch: maps scalar time t to u_q(t) via a QNode.

    Pipeline:
        t -> feature map φ(t) -> quantum_block(φ, θ) -> scalar output
    """

    def __init__(
        self,
        quantum_block: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.quantum_block = quantum_block

        # Trainable ansatz parameters: (n_layers, n_qubits, 3)
        self.theta = nn.Parameter(
            torch.randn(N_LAYERS, N_QUBITS, 3, dtype=DTYPE) * init_scale
        )

    def _feature_map(self, t: torch.Tensor) -> torch.Tensor:
        """
        Feature map φ(t) = [π t, 2π t, 3π t].
        """
        if t.dim() == 2:
            t_flat = t.squeeze(-1)
        else:
            t_flat = t

        scale = np.pi
        phi = torch.stack(
            [scale * t_flat, 2.0 * scale * t_flat, 3.0 * scale * t_flat],
            dim=1,
        )
        return phi

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        phi = self._feature_map(t)

        outputs = []
        for i in range(phi.size(0)):
            out_i = self.quantum_block(phi[i], self.theta)
            outputs.append(out_i)

        return torch.stack(outputs).unsqueeze(-1)
