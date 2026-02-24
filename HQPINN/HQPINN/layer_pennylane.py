# layer_pennylane.py

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from .config import N_QUBITS, N_LAYERS, DTYPE


import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.",
)


def make_device_lightning() -> qml.Device:  # type: ignore
    return qml.device("lightning.qubit", wires=N_QUBITS, shots=None, batch_obs=True)  # type: ignore


def make_device_default() -> qml.Device:  # type: ignore
    return qml.device("default.qubit", wires=N_QUBITS, shots=None)


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


def _make_quantum_block_with_measurement(
    measure_fn: Callable[[], torch.Tensor] | Callable[[], list],
    n_layers: int = N_LAYERS,
    device: str = "default",
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Internal helper to build a QNode with a given measurement rule.

    The circuit structure (ansatz_layer + feature_layer) is shared between
    scalar-output and multi-output quantum blocks.
    """

    if device == "lightning":
        dev = make_device_lightning()
        diff_method = "adjoint"  # first-order gradients only
    elif device == "default":
        dev = make_device_default()
        diff_method = "backprop"  # supports higher-order derivatives
    else:
        raise ValueError(f"Unknown device '{device}'. Use 'default' or 'lightning'.")

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def quantum_block(phi: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        # Apply ansatz + feature map layers
        for layer in range(n_layers):
            ansatz_layer(thetas[layer])
            if layer < n_layers - 1:
                feature_layer(phi)

        # Measurement is delegated to measure_fn (e.g. single Z or list of Z's)
        return measure_fn()  # type: ignore

    return quantum_block  # type: ignore


def measure_single():
    # Single expectation value on qubit 0
    return qml.expval(qml.PauliZ(0))  # type: ignore


def measure_all():
    # One expectation value per qubit -> list of length N_QUBITS
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]  # type: ignore


def make_quantum_block() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create a single-output QNode returning <Z_0>."""
    return _make_quantum_block_with_measurement(measure_single, device="default")  # type: ignore


def make_quantum_block_multiout(
    n_layers: int,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create a multi-output QNode returning one expectation per qubit."""
    return _make_quantum_block_with_measurement(measure_all, n_layers, device="lightning")  # type: ignore


# ============================================================
#  Quantum branch
# ============================================================


class BranchPennylane(nn.Module):
    """
    Generic quantum branch.

    Inputs:
        - quantum_block: QNode(phi, theta) -> scalar or vector
        - feature_map: callable mapping raw input x -> phi(x)
        - output_as_column: if True and the QNode returns scalar, output shape is [N, 1]

    This class can emulate:
        - scalar branch (DHO): t -> u_q(t) in R
        - vector branch (SEE): (x,t) -> R^D
    """

    def __init__(
        self,
        quantum_block,
        feature_map,
        n_layers: int,
        output_as_column: bool = False,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.quantum_block = quantum_block
        self.feature_map = feature_map
        self.output_as_column = output_as_column
        self.n_layers = n_layers

        # Trainable ansatz parameters: (n_layers, N_QUBITS, 3)
        self.theta = nn.Parameter(
            torch.randn(n_layers, N_QUBITS, 3, dtype=DTYPE) * init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map input to features phi(x)
        phi = self.feature_map(x)

        outputs = []
        for i in range(phi.size(0)):
            # quantum_block can return scalar or vector
            out_i = self.quantum_block(phi[i], self.theta)

            # Handle multi-output QNodes returning a list/tuple of tensors
            if isinstance(out_i, (list, tuple)):
                # Convert list of scalar tensors -> 1D tensor
                out_i = torch.stack(out_i, dim=0)

            outputs.append(out_i)

        out = torch.stack(outputs, dim=0)

        # If scalar output [N] and we want a column vector [N,1]
        if self.output_as_column and out.dim() == 1:
            out = out.unsqueeze(-1)

        # return out
        return out.to(DTYPE)


def dho_feature_map(t: torch.Tensor) -> torch.Tensor:
    """
    Feature map for DHO: phi(t) in R^3.
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


def see_feature_map(xt: torch.Tensor) -> torch.Tensor:
    """
    Feature map for the Smooth Euler Equation (SEE).
    xt: [N, 2] with columns (x, t).

    We encode both space and time inputs (x, t) into three features:
        πx        : spatial component,
        πt        : temporal component,
        π(x − t)  : travelling-wave component matching the exact solution ρ(x,t) ∝ sin(π(x−t)).
    """
    if xt.dim() != 2 or xt.size(1) != 2:
        raise ValueError(f"Expected input of shape [N, 2] for (x,t), got {xt.shape}")

    x = xt[:, 0]
    t = xt[:, 1]

    scale = np.pi
    phi = torch.stack(
        [
            scale * x,
            scale * t,
            scale * (x - t),
        ],
        dim=1,
    )
    return phi
