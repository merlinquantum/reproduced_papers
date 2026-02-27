# layer_pennylane.py
"""
PennyLane-based quantum branch for HQPINN.

This module implements the gate-model quantum branch used in the paper's
hybrid PINN setting: an alternating ansatz/feature-map block followed by
observable readout. It is used both as:
- a standalone quantum path in quantum-only variants,
- a component in hybrid models alongside classical branches.
"""

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
    Build the core variational quantum block used across experiments.

    At paper level, this corresponds to the reusable quantum branch pattern:
    data encoding + trainable ansatz + task-specific measurement.
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
    return qml.expval(qml.PauliZ(0))  # type: ignore


def measure_all():
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
    High-level PennyLane quantum branch wrapper.

    It connects three conceptual steps used in HQPINN:
    1) map PDE/ODE inputs to quantum features,
    2) evaluate the variational quantum block,
    3) return branch outputs that can be fused with other branches.
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
            out_i = self.quantum_block(phi[i], self.theta)

            if isinstance(out_i, (list, tuple)):
                out_i = torch.stack(out_i, dim=0)

            outputs.append(out_i)

        out = torch.stack(outputs, dim=0)

        if self.output_as_column and out.dim() == 1:
            out = out.unsqueeze(-1)

        return out.to(DTYPE)


def dho_feature_map(t: torch.Tensor) -> torch.Tensor:
    """
    Feature map used for the DHO setting in the paper.
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
    Feature map used for Euler-type experiments (SEE/DEE) in the paper.

    It maps spatio-temporal inputs (x, t) to a compact feature vector before
    the quantum branch evaluation.
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
