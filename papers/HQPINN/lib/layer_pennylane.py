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

from .config import DEFAULT_N_OUTPUTS, N_LAYERS, DTYPE, DEE_X0, DEE_U


import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.",
)


def make_device_lightning(n_qubits: int = DEFAULT_N_OUTPUTS) -> qml.Device:  # type: ignore
    return qml.device("lightning.qubit", wires=n_qubits, shots=None, batch_obs=True)  # type: ignore


def make_device_default(n_qubits: int = DEFAULT_N_OUTPUTS) -> qml.Device:  # type: ignore
    return qml.device("default.qubit", wires=n_qubits, shots=None)


# ============================================================
#  Quantum circuit building blocks
# ============================================================


def ansatz_layer(theta: torch.Tensor, n_qubits: int = DEFAULT_N_OUTPUTS) -> None:
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
        qml.RZ(theta[i, 0], wires=i)  # type: ignore
        qml.RX(theta[i, 1], wires=i)  # type: ignore
        qml.RZ(theta[i, 2], wires=i)  # type: ignore

    # Entangling ring
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])


def feature_layer(phi: torch.Tensor, n_qubits: int = DEFAULT_N_OUTPUTS) -> None:
    """
    Feature map: angle encoding via RY rotations.

    Parameters
    ----------
    phi : (n_qubits,) tensor-like
        For qubit i, apply RY(phi[i]).
    """
    for i in range(n_qubits):
        qml.RY(phi[i], wires=i)  # type: ignore


# ============================================================
#  QNode factory
# ============================================================


def _make_quantum_block_with_measurement(
    measure_fn: Callable[[int], torch.Tensor] | Callable[[int], list],
    n_layers: int = N_LAYERS,
    device: str = "default",
    n_qubits: int = DEFAULT_N_OUTPUTS,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Build the core variational quantum block used across experiments.

    At paper level, this corresponds to the reusable quantum branch pattern:
    data encoding + trainable ansatz + task-specific measurement.
    """

    if device == "lightning":
        dev = make_device_lightning(n_qubits=n_qubits)
        diff_method = "adjoint"  # first-order gradients only
    elif device == "default":
        dev = make_device_default(n_qubits=n_qubits)
        diff_method = "backprop"  # supports higher-order derivatives
    else:
        raise ValueError(f"Unknown device '{device}'. Use 'default' or 'lightning'.")

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def quantum_block(phi: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        # Apply ansatz + feature map layers
        for layer in range(n_layers):
            ansatz_layer(thetas[layer], n_qubits=n_qubits)
            if layer < n_layers - 1:
                feature_layer(phi, n_qubits=n_qubits)

        # Measurement is delegated to measure_fn (e.g. single Z or list of Z's)
        return measure_fn(n_qubits)  # type: ignore

    return quantum_block  # type: ignore


def measure_single(n_qubits: int):
    return qml.expval(qml.PauliZ(0))  # type: ignore


def measure_all(n_qubits: int):
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]  # type: ignore


def make_quantum_block(
    n_qubits: int = DEFAULT_N_OUTPUTS,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create a single-output QNode returning <Z_0>."""
    return _make_quantum_block_with_measurement(
        measure_single, device="default", n_qubits=n_qubits
    )  # type: ignore


def make_quantum_block_multiout(
    n_layers: int,
    n_qubits: int = DEFAULT_N_OUTPUTS,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create a multi-output QNode returning one expectation per qubit."""
    return _make_quantum_block_with_measurement(
        measure_all, n_layers, device="lightning", n_qubits=n_qubits
    )  # type: ignore


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
        n_qubits: int = DEFAULT_N_OUTPUTS,
        output_as_column: bool = False,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.quantum_block = quantum_block
        self.feature_map = feature_map
        self.output_as_column = output_as_column
        self.n_layers = n_layers
        self.n_qubits = n_qubits

        # Trainable ansatz parameters: (n_layers, n_qubits, 3)
        self.theta = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3, dtype=DTYPE) * init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map input to features phi(x)
        phi = self.feature_map(x)
        if phi.dim() != 2 or phi.size(1) != self.n_qubits:
            raise ValueError(
                f"Feature map must return shape [N, {self.n_qubits}], got {tuple(phi.shape)}"
            )

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


def dho_feature_map(
    t: torch.Tensor,
    n_qubits: int = DEFAULT_N_OUTPUTS,
) -> torch.Tensor:
    """
    Feature map used for the DHO setting in the paper.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")

    if t.dim() == 2:
        t_flat = t.squeeze(-1)
    else:
        t_flat = t

    scale = np.pi
    # Generalizes the original [1, 2, 3] harmonics to any qubit count.
    phi = torch.stack(
        [k * scale * t_flat for k in range(1, n_qubits + 1)],
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


def dee_feature_map(xt: torch.Tensor) -> torch.Tensor:
    """
    Feature map used for DEE experiments.

    It uses the shock-relative coordinate x - (x0 + u*t), aligned with the DEE
    setup where the front position is x_f(t) = x0 + u*t.
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
            scale * (x - (DEE_X0 + DEE_U * t)),
        ],
        dim=1,
    )
    return phi


# def taf_feature_map(xy: torch.Tensor) -> torch.Tensor:
#     """
#     Feature map used for TAF experiments.

#     Returns 4 encoded features to match a 4-qubit PennyLane branch.
#     """
#     if xy.dim() != 2 or xy.size(1) != 2:
#         raise ValueError(f"Expected input of shape [N, 2] for (x,y), got {xy.shape}")

#     x = xy[:, 0]
#     y = xy[:, 1]

#     scale = np.pi
#     phi = torch.stack(
#         [
#             scale * x,
#             scale * y,
#             scale * (x - y),
#             scale * (x + y),
#         ],
#         dim=1,
#     )
#     return phi


def taf_feature_map(xy: torch.Tensor) -> torch.Tensor:
    """TAF: encodage angulaire centré, normalisé, avec couplage modéré."""
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f"Expected input of shape [N, 2] for (x,y), got {tuple(xy.shape)}"
        )

    x = xy[:, 0]
    y = xy[:, 1]

    # Domaine recentré et ramené à une échelle ~[-1, 1].
    xh = (x - 1.25) / 2.25
    yh = y / 2.25

    # x et y restent lisibles ; x±y ajoutent un couplage sans sur-osciller.
    phi = torch.stack(
        [
            torch.pi * xh,
            torch.pi * yh,
            0.5 * torch.pi * (xh - yh),
            0.5 * torch.pi * (xh + yh),
        ],
        dim=1,
    )
    return phi
