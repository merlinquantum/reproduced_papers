"""DQML model classes wrapping the QCNN circuit blocks.

A single :class:`DQMLModel` covers all four schemes from the paper:

* ``"non"``  - single 4-qubit QPU; embedding applied twice (4 attributes
  each) since the dataset has 8 attributes.
* ``"nc"``   - two 4-qubit QPUs, no inter-QPU operations.
* ``"cc"``   - two 4-qubit QPUs with cross-QPU classical communication
  modelled as a controlled-RX gate per sub-layer (deferred measurement).
* ``"qc"``   - two 4-qubit QPUs with cross-QPU quantum communication
  modelled as a controlled-RX + controlled-RZ gate pair per sub-layer.

The forward pass returns the interpret-function output

    fint = w_0 P[00] + w_1 P[01] + w_2 P[10] + w_3 P[11]

for two-QPU schemes (and ``w_0 P[0] + w_1 P[1]`` for ``"non"``), where the
weights ``w`` are trainable alongside the gate parameters. This matches
Eq. (3) of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from . import circuit as qc
from . import simulator as sim


@dataclass(frozen=True)
class DQMLConfig:
    scheme: str = "cc"  # one of {"non", "nc", "cc", "qc"}
    n_layers: int = 9
    qubits_per_qpu: int = 4
    n_attributes: int = 8


class DQMLModel(nn.Module):
    """Distributed QCNN model implementing all four schemes."""

    def __init__(self, cfg: DQMLConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        scheme = cfg.scheme.lower()
        if scheme not in {"non", "nc", "cc", "qc"}:
            raise ValueError(f"unknown scheme '{scheme}'")
        if cfg.qubits_per_qpu != 4:
            raise ValueError("only qubits_per_qpu=4 is supported")
        if cfg.n_attributes != 8:
            raise ValueError("only n_attributes=8 is supported")
        self.cfg = cfg
        self.scheme = scheme
        self.dtype = dtype
        # Use double precision complex if the model dtype is double.
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        self.layout = qc.ParamLayout.for_scheme(scheme, cfg.n_layers, cfg.qubits_per_qpu)

        # Layout flags.
        self.n_qpus = 1 if scheme == "non" else 2
        self.n_total_qubits = self.n_qpus * cfg.qubits_per_qpu

        # Trainable parameters.
        # Initial values: uniform in [0, 2*pi) for circuit params; uniform [-1, 1] for w.
        self.conv_params = nn.Parameter(
            torch.empty(self.n_qpus, cfg.n_layers, cfg.qubits_per_qpu, dtype=dtype).uniform_(0.0, 2 * torch.pi)
        )
        self.pool_params = nn.Parameter(
            torch.empty(self.n_qpus, self.layout.pool_per_qpu, dtype=dtype).uniform_(0.0, 2 * torch.pi)
        )
        if self.layout.cross_qpu > 0:
            self.cross_params = nn.Parameter(
                torch.empty(cfg.n_layers, self.layout.cross_qpu // cfg.n_layers, dtype=dtype).uniform_(0.0, 2 * torch.pi)
            )
        else:
            self.register_parameter("cross_params", None)
        self.interpret_weights = nn.Parameter(
            torch.empty(self.layout.interpret, dtype=dtype).uniform_(-1.0, 1.0)
        )

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    def _qpu_qubits(self, qpu_idx: int) -> list[int]:
        return list(range(qpu_idx * self.cfg.qubits_per_qpu,
                          (qpu_idx + 1) * self.cfg.qubits_per_qpu))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the interpret-function output for a batch of inputs.

        ``x`` has shape ``(batch, 8)``.
        """
        batch = x.shape[0]
        device = x.device
        # Initialise |0...0>.
        state = sim.init_state(self.n_total_qubits, batch, dtype=self.cdtype, device=device)

        # Embedding -------------------------------------------------------
        qpb = self.cfg.qubits_per_qpu
        if self.scheme == "non":
            # First half on qubits 0..3, then second half repeats the embedding.
            state = qc.embed_attributes(state, x[:, :qpb], self._qpu_qubits(0), self.cdtype)
            state = qc.embed_attributes(state, x[:, qpb:], self._qpu_qubits(0), self.cdtype)
        else:
            state = qc.embed_attributes(state, x[:, :qpb], self._qpu_qubits(0), self.cdtype)
            state = qc.embed_attributes(state, x[:, qpb:], self._qpu_qubits(1), self.cdtype)

        # Convolutional sub-layers + cross-QPU edges ----------------------
        boundary_a = self._qpu_qubits(0)[-1] if self.n_qpus > 1 else None
        boundary_b = self._qpu_qubits(1)[0] if self.n_qpus > 1 else None
        cross_per_layer = self.layout.cross_qpu // self.cfg.n_layers if self.layout.cross_qpu else 0

        for layer_idx in range(self.cfg.n_layers):
            parity = layer_idx % 2
            for qpu_idx in range(self.n_qpus):
                params = self.conv_params[qpu_idx, layer_idx]
                state = qc.conv_sublayer(state, params, self._qpu_qubits(qpu_idx), parity, self.cdtype)
            if cross_per_layer > 0:
                state = qc.cross_qpu_edge(
                    state,
                    self.cross_params[layer_idx],
                    boundary_a,
                    boundary_b,
                    self.scheme,
                    self.cdtype,
                )

        # Pooling layers --------------------------------------------------
        readout_qubits: list[int] = []
        for qpu_idx in range(self.n_qpus):
            active = self._qpu_qubits(qpu_idx)
            pool = self.pool_params[qpu_idx]
            pool_cursor = 0
            while len(active) > 1:
                n_blocks = len(active) // 2
                seg = pool[pool_cursor:pool_cursor + 4 * n_blocks]
                state, active = qc.pooling_layer(state, seg, active, self.cdtype)
                pool_cursor += 4 * n_blocks
            readout_qubits.extend(active)

        # Readout marginals + interpret function -------------------------
        probs = sim.marginal_probabilities(state, readout_qubits)
        if self.n_qpus == 1:
            # shape (batch, 2)
            flat = probs.reshape(batch, 2)
        else:
            # shape (batch, 2, 2) -> flatten in (q0, q1) lex order
            flat = probs.reshape(batch, 4)
        fint = flat @ self.interpret_weights
        return fint


__all__ = ["DQMLConfig", "DQMLModel"]
