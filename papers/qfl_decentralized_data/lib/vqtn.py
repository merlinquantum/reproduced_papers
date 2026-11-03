"""Variational quantum tensor network (VQTN) classifier, Huang et al. 2022, Eq. (2).

The paper only sketches the circuit, so the structure below is an interpretation
(LOG.md, "Reproducibility Assessment"):

* kernel encoding: 16 pixel values of a 4x4 image go to 8 qubits, two pixels per
  qubit, as ``RZ(pi x_odd) RY(pi x_even) |0>``;
* a tree tensor network of seven nodes. Each node applies a trainable ``RY`` to
  both of its qubits and a trainable ``RXX`` entangler, then passes one qubit up
  to the next layer;
* read-out: marginal outcome probabilities of the root qubit (binary tasks) or of
  the two qubits of the root node (4-class task).

Three switches exist only for the gap diagnosis (LOG.md, "Gap diagnosis"); their
defaults are the circuit above: ``encoding_scale`` (angle = scale * pi * x),
``encoding_gate`` (``rz`` or ``rx`` for the second pixel of a pair) and
``readout`` (``root_marginal``, or ``z_head``: a classical linear layer on the eight
single-qubit ``<Z>``, a diagnostic that is not the paper's read-out).

Every model in this package returns log-probabilities, so one negative
log-likelihood loss serves quantum and classical models.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from . import statevector as sv

N_PIXELS = 16
N_QUBITS = 8
# (wire_a, wire_b) per tree node; layer 1 pairs neighbours, upper layers pair survivors.
TREE_NODES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (4, 6), (0, 4)]
ROOT_WIRES = {2: [0], 4: [0, 4]}
PARAMS_PER_NODE = 3  # RY on wire_a, RY on wire_b, RXX
LOG_PROBABILITY_FLOOR = 1e-8


class VQTNClassifier(nn.Module):
    """Tree-tensor-network variational circuit returning class log-probabilities.

    Parameters
    ----------
    n_classes : int
        Number of classes; 2 or 4.
    init_std : float
        Standard deviation of the normal initialisation of the circuit angles.
    dtype : torch.dtype
        Real dtype of the parameters; the simulation uses the matching complex dtype.
    layers : int
        Number of times the seven-node tree block is applied in sequence (21 trainable
        angles per layer). The paper does not give the depth; 1 is the shallowest
        reading and deeper values are a calibration axis. Default value is 1.
    encoding_scale : float
        Multiplies ``pi`` in the encoding angles. Default value is 1.0.
    encoding_gate : str
        ``rz`` or ``rx``: rotation applied for the second pixel of each pair.
        Default value is ``rz``.
    readout : str
        ``root_marginal`` or ``z_head`` (diagnostic). Default value is
        ``root_marginal``.
    """

    def __init__(
        self,
        n_classes: int,
        init_std: float,
        dtype: torch.dtype = torch.float32,
        layers: int = 1,
        encoding_scale: float = 1.0,
        encoding_gate: str = "rz",
        readout: str = "root_marginal",
    ) -> None:
        super().__init__()
        if encoding_gate not in ("rz", "rx"):
            raise ValueError(
                f"encoding_gate must be 'rz' or 'rx', got {encoding_gate!r}"
            )
        if readout not in ("root_marginal", "z_head"):
            raise ValueError(
                f"readout must be 'root_marginal' or 'z_head', got {readout!r}"
            )
        if n_classes not in ROOT_WIRES:
            raise ValueError(f"VQTN supports 2 or 4 classes, got {n_classes}")
        if layers < 1:
            raise ValueError(f"layers must be at least 1, got {layers}")
        self.n_classes = n_classes
        self.layers = layers
        self.encoding_scale = encoding_scale
        self.encoding_gate = encoding_gate
        self.readout = readout
        if readout == "z_head":
            self.head = nn.Linear(N_QUBITS, n_classes, dtype=dtype)
        self.angles = nn.Parameter(
            init_std
            * torch.randn(layers * len(TREE_NODES), PARAMS_PER_NODE, dtype=dtype)
        )

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Kernel-encode ``(batch, 16)`` pixels in [0, 1] into an 8-qubit state."""
        state = sv.zero_state(pixels.shape[0], N_QUBITS, pixels.dtype, pixels.device)
        for wire in range(N_QUBITS):
            polar = self.encoding_scale * math.pi * pixels[:, 2 * wire]
            second = self.encoding_scale * math.pi * pixels[:, 2 * wire + 1]
            second_gate = sv.rz_matrix if self.encoding_gate == "rz" else sv.rx_matrix
            gate = second_gate(second) @ sv.ry_matrix(polar)
            state = sv.apply_one_qubit(state, gate, wire)
        return state

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return class log-probabilities, shape ``(batch, n_classes)``."""
        state = self.encode(pixels)
        for node, (wire_a, wire_b) in enumerate(TREE_NODES * self.layers):
            angle_a, angle_b, angle_xx = self.angles[node]
            state = sv.apply_one_qubit(state, sv.ry_matrix(angle_a), wire_a)
            state = sv.apply_one_qubit(state, sv.ry_matrix(angle_b), wire_b)
            state = sv.apply_two_qubit(state, sv.rxx_matrix(angle_xx), wire_a, wire_b)
        if self.readout == "z_head":
            excited = torch.stack(
                [sv.marginal_probabilities(state, [w])[:, 1] for w in range(N_QUBITS)],
                dim=1,
            )
            return torch.log_softmax(self.head(1.0 - 2.0 * excited), dim=1)
        probabilities = sv.marginal_probabilities(state, ROOT_WIRES[self.n_classes])
        return torch.log(probabilities[:, : self.n_classes] + LOG_PROBABILITY_FLOOR)
