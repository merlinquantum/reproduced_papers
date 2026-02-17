# layer_pennylane.py

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# ============================================================
#  Device factory
# ============================================================


def make_device(n_qubits: int) -> qml.Device: # type: ignore
    """Create a PennyLane default.qubit device."""
    return qml.device("default.qubit", wires=n_qubits)


# ============================================================
#  Quantum circuit building blocks
# ============================================================


def ansatz_layer(theta: torch.Tensor, n_qubits: int) -> None:
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
    for i in range(n_qubits):
        qml.RZ(theta[i, 0], wires=i) # type: ignore
        qml.RX(theta[i, 1], wires=i) # type: ignore
        qml.RZ(theta[i, 2], wires=i) # type: ignore

    # Entangling ring
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])


def feature_layer(phi: torch.Tensor, n_qubits: int) -> None:
    """
    Feature map: angle encoding via RY rotations.

    Parameters
    ----------
    phi : (n_qubits,) tensor-like
        For qubit i, apply RY(phi[i]).
    """
    for i in range(n_qubits):
        qml.RY(phi[i], wires=i) # type: ignore


# ============================================================
#  QNode factory
# ============================================================


def make_quantum_block(
    dev: qml.Device, # type: ignore
    n_qubits: int,
    n_layers: int,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Factory that creates a PennyLane QNode:

        quantum_block(phi, thetas) -> <Z_0>

    QNode is stateless; trainable parameters are passed as 'thetas'.
    """

    @qml.qnode(dev, interface="torch")
    def quantum_block(phi: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        """
        Multi-layer quantum block used as a branch core.

        Parameters
        ----------
        phi : (n_qubits,) torch.Tensor
            Feature vector.

        thetas : (n_layers, n_qubits, 3) torch.Tensor
            Trainable parameters for each ansatz layer.

        Returns
        -------
        torch.Tensor (scalar)
            Expectation value ⟨Z₀⟩.
        """
        for layer in range(n_layers):
            ansatz_layer(thetas[layer], n_qubits=n_qubits)
            if layer < n_layers - 1:
                feature_layer(phi, n_qubits=n_qubits)

        return qml.expval(qml.PauliZ(0)) # type: ignore

    return quantum_block  # type: ignore


# ============================================================
#  Quantum branch (shared between QQ and CQ)
# ============================================================


class QuantumBranch(nn.Module):
    """
    Quantum branch: maps scalar time t to u_q(t) via a QNode.

    Pipeline:
        t -> feature map φ(t) -> quantum_block(φ, θ) -> scalar output
    """

    def __init__(
        self,
        quantum_block: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_qubits: int,
        n_layers: int,
        dtype: torch.dtype = torch.float32,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.quantum_block = quantum_block
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Trainable ansatz parameters: (n_layers, n_qubits, 3)
        self.theta = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3, dtype=dtype) * init_scale
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
