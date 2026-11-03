"""Gate-model variational quantum circuit (VQC) for the GQC secondary expert.

Circuit (PennyLane, ``default.qubit``):

1. Angle encoding — ``qml.RY(x_i, wires=i)`` for each of the ``n_qubits``
   latent features, applied once (single data-upload).
2. ``n_layers`` repeats of a trainable block: ``RY(theta)`` then
   ``RZ(theta)`` on every qubit, followed by a nearest-neighbour CNOT chain
   (``CNOT(wires=[i, i + 1])`` for ``i`` in ``0 .. n_qubits - 2``).
3. Measurement: ``qml.expval(qml.PauliZ(0))`` — a single scalar in
   ``[-1, 1]``.

Wrapped with ``qml.qnn.TorchLayer`` so it behaves as a plain ``nn.Module``.

PennyLane batching note (see LOG.md "PennyLane API pitfalls"): TorchLayer
passes the *whole* batched input tensor straight through to the QNode as the
``inputs`` argument (no per-sample Python loop). PennyLane's own parameter
broadcasting then requires the batch dimension to be axis 0 of each gate
*parameter*, not axis 0 of ``inputs`` itself. Concretely, for a batched
``inputs`` of shape ``(batch, n_qubits)`` the per-qubit encoding angle must
be selected as ``inputs[:, i]`` (shape ``(batch,)``), NOT ``inputs[i]``
(which would index into the batch dimension). ``inputs[i]`` only works for
an unbatched single sample of shape ``(n_qubits,)``. This module always
feeds the circuit a 2D ``(batch, n_qubits)`` tensor (reshaping a single
sample to ``(1, n_qubits)`` if needed) so the ``inputs[:, i]`` convention is
always valid; the output shape is then ``(batch,)``.
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn


def _build_qnode(n_qubits: int, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        # inputs: (batch, n_qubits) -- see module docstring for why axis 1
        # (not axis 0) selects the per-qubit angle.
        for i in range(n_qubits):
            qml.RY(inputs[:, i], wires=i)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    return circuit


class VQCLayer(nn.Module):
    """``nn.Module`` wrapper around the gate-model VQC.

    Always reshapes its input to 2D ``(batch, n_qubits)`` before calling the
    underlying ``TorchLayer`` (see module docstring), then returns a
    ``(batch,)`` tensor of PauliZ(0) expectation values in ``[-1, 1]``.
    """

    def __init__(self, n_qubits: int = 6, n_layers: int = 6) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        circuit = _build_qnode(n_qubits, n_layers)
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.torch_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.torch_layer(x)


__all__ = ["VQCLayer"]
