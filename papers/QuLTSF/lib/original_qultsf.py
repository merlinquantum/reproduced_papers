"""Original QuLTSF hybrid model port.

This module ports the core hybrid model from:
`chariharasuthan/QuLTSF` (`models/QuLTSF.py`).

The upstream Weather experiment script uses:
- `features=M`
- `enc_in=21`
- `seq_len=336`
- `pred_len in {96, 192, 336, 720}`
- `num_qubits=10`
- `QML_device=default.qubit`

We keep the architecture behavior intact:
- one classical input projection per channel from `seq_len -> 2**num_qubits`
- one shared quantum hidden layer returning `num_qubits` expectation values
- one classical output projection per channel from `num_qubits -> pred_len`

Inputs are expected as `[batch, seq_len, channels]` and outputs are
`[batch, pred_len, channels]`, matching the upstream implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class OriginalQuLTSFConfig:
    seq_len: int
    pred_len: int
    num_qubits: int = 10
    qml_device: str = "default.qubit"
    num_layers: int = 3
    dtype: torch.dtype | None = None


class OriginalQuLTSFModel(nn.Module):
    """Port of the upstream `Model` / `Hybrid_QML_Model` classes."""

    def __init__(self, config: OriginalQuLTSFConfig) -> None:
        super().__init__()
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.num_qubits = int(config.num_qubits)
        self.qml_device = str(config.qml_device)
        self.num_layers = int(config.num_layers)
        self.dtype = config.dtype or torch.float32

        try:
            import pennylane as qml
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "qultsf_original requires PennyLane. Install it with "
                "`pip install pennylane` or use another QuLTSF model."
            ) from exc

        self._qml = qml
        self.dev = qml.device(self.qml_device, wires=self.num_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def quantum_function(inputs, weights):
            qml.AmplitudeEmbedding(
                features=inputs,
                wires=range(self.num_qubits),
                normalize=True,
            )
            qml.StronglyEntanglingLayers(
                weights=weights,
                wires=range(self.num_qubits),
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        q_weights_shape = {"weights": (self.num_layers, self.num_qubits, 3)}
        self.input_classical_layer = nn.Linear(
            self.seq_len,
            2**self.num_qubits,
            dtype=self.dtype,
        )
        self.hidden_quantum_layer = qml.qnn.TorchLayer(
            quantum_function,
            q_weights_shape,
        )
        self.output_classical_layer = nn.Linear(
            self.num_qubits,
            self.pred_len,
            dtype=self.dtype,
        )
        self.hidden_quantum_layer = self.hidden_quantum_layer.to(dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upstream convention:
        # x: [batch, input length, channel]
        # internal: [batch, channel, input length]
        batch_input = x.to(dtype=self.dtype).permute(0, 2, 1)
        y = batch_input.reshape(
            batch_input.shape[0] * batch_input.shape[1],
            batch_input.shape[2],
        )
        y = self.input_classical_layer(y)
        y = self.hidden_quantum_layer(y)
        y = self.output_classical_layer(y)
        batch_output = y.reshape(
            batch_input.shape[0],
            batch_input.shape[1],
            self.pred_len,
        )
        return batch_output.permute(0, 2, 1)


__all__ = ["OriginalQuLTSFConfig", "OriginalQuLTSFModel"]
