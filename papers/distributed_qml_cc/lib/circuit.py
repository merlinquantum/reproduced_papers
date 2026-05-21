"""QCNN circuit blocks for the four DQML schemes of arXiv:2408.16327.

The implementation reproduces:

* an 8-attribute Havlicek-style embedding (H, RZ, ZZ couplings) repeated
  appropriately for single- and dual-QPU schemes,
* brick-wall convolutional sub-layers with one parameter per qubit per
  sub-layer,
* the QCNN pooling block as a deferred-measurement two-qubit gate, and
* the four DQML communication schemes (``non``, ``nc``, ``cc``, ``qc``).

Cross-QPU edges (only present for ``cc`` and ``qc``) are placed between the
two boundary qubits ``(qubits_per_qpu - 1)`` of QPU 0 and ``0`` of QPU 1.
For ``cc`` we apply a single controlled rotation per sub-layer; for ``qc``
we apply two controlled rotations (broader two-qubit subgroup) per
sub-layer. This mirrors the circuit-capacity hierarchy
``NC < CC < QC`` quantified in Fig. 3c of the paper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from . import simulator as sim


# ---------------------------------------------------------------------------
# Embedding (Havlicek ZZ feature map with cyclic couplings).
# ---------------------------------------------------------------------------
def embed_attributes(state: torch.Tensor, x: torch.Tensor, qubits: Sequence[int],
                     dtype: torch.dtype) -> torch.Tensor:
    """Apply the H / RZ(x_i) / cyclic ZZ(x_i, x_{i+1}) embedding.

    Parameters
    ----------
    state : torch.Tensor
        State tensor of shape ``(batch, 2, ..., 2)``.
    x : torch.Tensor
        Attributes for these qubits of shape ``(batch, len(qubits))``.
    qubits : Sequence[int]
        Qubit indices receiving the attributes ``x[:, k]`` for
        ``k = 0, ..., len(qubits) - 1``.
    """
    n = len(qubits)
    for q in qubits:
        state = sim.apply_h(state, q, dtype)
    for k, q in enumerate(qubits):
        state = sim.apply_rz(state, x[:, k], q, dtype)
    # Cyclic ZZ couplings: (0,1), (1,2), ..., (n-1, 0).
    for k in range(n):
        q1 = qubits[k]
        q2 = qubits[(k + 1) % n]
        theta = x[:, k] * x[:, (k + 1) % n]
        state = sim.apply_zz_phase(state, theta, q1, q2, dtype)
    return state


# ---------------------------------------------------------------------------
# Convolutional sub-layer (one parameter per qubit, brick-wall entanglers).
# ---------------------------------------------------------------------------
def conv_sublayer(state: torch.Tensor, params: torch.Tensor, qubits: Sequence[int],
                  parity: int, dtype: torch.dtype) -> torch.Tensor:
    """Apply one brick-wall convolutional sub-layer.

    ``params`` has shape ``(len(qubits),)`` and provides one rotation angle
    per qubit. ``parity`` selects the brick-wall pairing: ``0`` couples
    ``(0,1),(2,3),...`` and ``1`` couples ``(1,2),(3,4),...,(n-1,0)``.
    """
    for k, q in enumerate(qubits):
        state = sim.apply_rx(state, params[k], q, dtype)
    n = len(qubits)
    if n < 2:
        return state
    if parity == 0:
        pairs = [(qubits[i], qubits[i + 1]) for i in range(0, n - 1, 2)]
    else:
        pairs = [(qubits[i], qubits[i + 1]) for i in range(1, n - 1, 2)]
        # Cyclic wrap-around pair for the odd sub-layer when n is even.
        if n >= 4 and n % 2 == 0:
            pairs.append((qubits[-1], qubits[0]))
    for a, b in pairs:
        state = sim.apply_cnot(state, a, b, dtype)
    return state


# ---------------------------------------------------------------------------
# Cross-QPU "communication" edge (used by CC and QC).
# ---------------------------------------------------------------------------
def cross_qpu_edge(state: torch.Tensor, params: torch.Tensor, qpu_a_boundary: int,
                   qpu_b_boundary: int, scheme: str, dtype: torch.dtype) -> torch.Tensor:
    """Apply the cross-QPU communication red block.

    * ``scheme='cc'``: one controlled-RX gate, controlled by the QPU-A
      boundary qubit and acting on the QPU-B boundary qubit. ``params``
      must have shape ``(1,)``.
    * ``scheme='qc'``: a controlled-RX and a controlled-RZ on the same
      pair; ``params`` must have shape ``(2,)``.
    """
    if scheme == "cc":
        state = sim.apply_crx(state, params[0], qpu_a_boundary, qpu_b_boundary, dtype)
    elif scheme == "qc":
        state = sim.apply_crx(state, params[0], qpu_a_boundary, qpu_b_boundary, dtype)
        state = sim.apply_crz(state, params[1], qpu_a_boundary, qpu_b_boundary, dtype)
    else:
        raise ValueError(f"cross_qpu_edge called with non-comm scheme '{scheme}'")
    return state


# ---------------------------------------------------------------------------
# Pooling layer: tree of pooling blocks reducing qubits in half each stage.
# ---------------------------------------------------------------------------
def pooling_layer(state: torch.Tensor, params: torch.Tensor, active_qubits: Sequence[int],
                  dtype: torch.dtype) -> tuple[torch.Tensor, list[int]]:
    """Apply one pooling layer on ``active_qubits`` and return the new active list.

    ``active_qubits`` must have even length; pairs ``(active[2k], active[2k+1])``
    use one pooling block (4 parameters), the block's "control" qubit
    (``active[2k+1]``) is then discarded from the active set. Returns the
    updated state and the new list of active qubits.
    """
    if len(active_qubits) % 2 != 0:
        raise ValueError("pooling_layer requires an even number of active qubits")
    n_blocks = len(active_qubits) // 2
    if params.numel() != 4 * n_blocks:
        raise ValueError(f"pooling_layer expects {4 * n_blocks} params, got {params.numel()}")
    new_active: list[int] = []
    for k in range(n_blocks):
        target = active_qubits[2 * k]
        control = active_qubits[2 * k + 1]
        block_params = params[4 * k:4 * (k + 1)]
        state = sim.apply_pooling_block(state, block_params, control, target, dtype)
        new_active.append(target)
    return state, new_active


# ---------------------------------------------------------------------------
# Parameter book-keeping for the full DQML model.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamLayout:
    """Number of trainable parameters in each block of the DQML circuit."""
    conv_per_qpu: int
    cross_qpu: int
    pool_per_qpu: int
    interpret: int
    total: int

    @classmethod
    def for_scheme(cls, scheme: str, n_layers: int, qubits_per_qpu: int = 4) -> ParamLayout:
        scheme = scheme.lower()
        if scheme not in {"non", "nc", "cc", "qc"}:
            raise ValueError(f"unknown scheme '{scheme}'")
        # Pooling reduces qubits_per_qpu -> 1 via log2 halvings.
        n_pool_blocks_per_qpu = qubits_per_qpu - 1  # 4 -> 2 -> 1 uses 3 blocks
        pool_per_qpu = 4 * n_pool_blocks_per_qpu
        conv_per_qpu = qubits_per_qpu * n_layers
        if scheme == "non":
            cross = 0
            n_qpus = 1
            interpret = 2  # w_0, w_1 for one bit of readout
        else:
            n_qpus = 2
            interpret = 4
            if scheme == "nc":
                cross = 0
            elif scheme == "cc":
                cross = 1 * n_layers
            else:  # qc
                cross = 2 * n_layers
        total = n_qpus * conv_per_qpu + cross + n_qpus * pool_per_qpu + interpret
        return cls(
            conv_per_qpu=conv_per_qpu,
            cross_qpu=cross,
            pool_per_qpu=pool_per_qpu,
            interpret=interpret,
            total=total,
        )


def num_parameters(scheme: str, n_layers: int, qubits_per_qpu: int = 4) -> int:
    return ParamLayout.for_scheme(scheme, n_layers, qubits_per_qpu).total


__all__ = [
    "embed_attributes",
    "conv_sublayer",
    "cross_qpu_edge",
    "pooling_layer",
    "ParamLayout",
    "num_parameters",
]
