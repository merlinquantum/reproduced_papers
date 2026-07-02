"""Minimal batched pure-state simulator for small qubit circuits.

The simulator keeps the wavefunction as a batched tensor of shape
``(batch, 2, 2, ..., 2)`` with one tensor axis per qubit. Single- and
two-qubit gates are applied via ``torch.einsum`` over the relevant axes,
which is fast for up to ~10 qubits and is fully autograd-compatible.

The intended use is to construct the QCNN-style circuits of the
distributed quantum machine learning paper without an external quantum
library. Mid-circuit measurements with classical feedforward (the
pooling block) are implemented through deferred measurement: the
classical-control gate is replaced by its equivalent two-qubit unitary
and the "measured" qubit is simply not read out at the end.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _ascomplex(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if t.dtype != dtype:
        t = t.to(dtype)
    return t


def init_state(n_qubits: int, batch_size: int, dtype: torch.dtype = torch.complex64,
               device: torch.device | str | None = None) -> torch.Tensor:
    """Return the all-zero state ``|0...0>`` broadcast over a batch."""
    shape = (batch_size,) + (2,) * n_qubits
    state = torch.zeros(shape, dtype=dtype, device=device)
    # element [batch, 0, 0, ..., 0] -> 1
    idx = (slice(None),) + (0,) * n_qubits
    state[idx] = 1.0
    return state


# ---------------------------------------------------------------------------
# Single-qubit gate matrices (returned as 2x2 complex tensors).
# ---------------------------------------------------------------------------
def _h_matrix(dtype: torch.dtype, device) -> torch.Tensor:
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=device)
    return h / math.sqrt(2.0)


def rx_matrix(theta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """RX(theta) = exp(-i theta/2 X). Supports a scalar tensor or a batch."""
    c = torch.cos(theta / 2.0).to(dtype)
    s = torch.sin(theta / 2.0).to(dtype)
    minus_i_s = -1j * s
    # Broadcast: theta can be scalar or shaped (...). Reshape to (..., 2, 2).
    top = torch.stack([c, minus_i_s], dim=-1)
    bot = torch.stack([minus_i_s, c], dim=-1)
    return torch.stack([top, bot], dim=-2)


def rz_matrix(theta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """RZ(theta) = exp(-i theta/2 Z). Diagonal."""
    half = theta / 2.0
    e_minus = torch.exp(-1j * half).to(dtype)
    e_plus = torch.exp(1j * half).to(dtype)
    zero = torch.zeros_like(e_minus)
    top = torch.stack([e_minus, zero], dim=-1)
    bot = torch.stack([zero, e_plus], dim=-1)
    return torch.stack([top, bot], dim=-2)


# ---------------------------------------------------------------------------
# Gate application helpers.
# ---------------------------------------------------------------------------
def apply_single_qubit_gate(state: torch.Tensor, gate: torch.Tensor, qubit: int) -> torch.Tensor:
    """Apply a single-qubit gate to the given qubit axis.

    ``state`` has shape ``(batch, 2, 2, ..., 2)``. ``gate`` is either an
    unbatched ``(2, 2)`` matrix (broadcast across the batch) or a batched
    ``(batch, 2, 2)`` matrix (one gate per sample).
    """
    axis = qubit + 1
    state = state.movedim(axis, 1)
    if gate.ndim == 2:
        state = torch.einsum("ij,bj...->bi...", gate, state)
    elif gate.ndim == 3:
        state = torch.einsum("bij,bj...->bi...", gate, state)
    else:
        raise ValueError(f"gate must have 2 or 3 dimensions, got {gate.ndim}")
    state = state.movedim(1, axis)
    return state


def apply_two_qubit_gate(state: torch.Tensor, gate: torch.Tensor, q1: int, q2: int) -> torch.Tensor:
    """Apply a two-qubit gate to the qubit pair (q1, q2).

    The basis ordering is row-major ``|q1 q2>`` in
    ``{|00>, |01>, |10>, |11>}``. ``gate`` is either an unbatched
    ``(4, 4)`` matrix (or its tensor reshape ``(2, 2, 2, 2)``) or a batched
    version with a leading batch dimension.
    """
    if q1 == q2:
        raise ValueError("Two-qubit gate requires distinct qubit indices")
    a1, a2 = q1 + 1, q2 + 1
    state = state.movedim((a1, a2), (1, 2))
    if gate.ndim == 2:
        gate4 = gate.reshape(2, 2, 2, 2)
        state = torch.einsum("ijkl,bkl...->bij...", gate4, state)
    elif gate.ndim == 3 and gate.shape[-2:] == (4, 4):
        gate4 = gate.reshape(-1, 2, 2, 2, 2)
        state = torch.einsum("bijkl,bkl...->bij...", gate4, state)
    elif gate.ndim == 4:
        # Already in (2,2,2,2) form, unbatched.
        state = torch.einsum("ijkl,bkl...->bij...", gate, state)
    elif gate.ndim == 5:
        state = torch.einsum("bijkl,bkl...->bij...", gate, state)
    else:
        raise ValueError(f"gate has unsupported shape {tuple(gate.shape)}")
    state = state.movedim((1, 2), (a1, a2))
    return state


def cnot_gate(dtype: torch.dtype, device) -> torch.Tensor:
    """CNOT with first index as control, second as target."""
    g = torch.zeros(4, 4, dtype=dtype, device=device)
    g[0, 0] = 1.0
    g[1, 1] = 1.0
    g[2, 3] = 1.0
    g[3, 2] = 1.0
    return g


def controlled_rotation_gate(rot: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Build a controlled rotation gate from a 2x2 unitary on the target.

    Acts as ``|0><0| (x) I + |1><1| (x) rot`` in the (control, target) basis.
    """
    g = torch.zeros(4, 4, dtype=dtype, device=rot.device)
    g[0, 0] = 1.0
    g[1, 1] = 1.0
    g[2, 2] = rot[0, 0]
    g[2, 3] = rot[0, 1]
    g[3, 2] = rot[1, 0]
    g[3, 3] = rot[1, 1]
    return g


def feedforward_pooling_gate(theta0: torch.Tensor, theta1: torch.Tensor,
                             phi0: torch.Tensor, phi1: torch.Tensor,
                             dtype: torch.dtype) -> torch.Tensor:
    """Build the QCNN pooling block as a deferred-measurement 2-qubit gate.

    The classical pooling block of the paper is

        measure qubit a -> outcome o in {0,1}
        apply RZ(theta_o) then RX(phi_o) on qubit b

    Deferring the measurement yields the unitary

        |0><0|_a (x) U_0 + |1><1|_a (x) U_1

    on the (a, b) qubit pair, where ``U_o = RX(phi_o) RZ(theta_o)`` acts on
    qubit b.
    """
    u0 = rx_matrix(phi0, dtype) @ rz_matrix(theta0, dtype)
    u1 = rx_matrix(phi1, dtype) @ rz_matrix(theta1, dtype)
    g = torch.zeros(4, 4, dtype=dtype, device=u0.device)
    g[0, 0] = u0[0, 0]
    g[0, 1] = u0[0, 1]
    g[1, 0] = u0[1, 0]
    g[1, 1] = u0[1, 1]
    g[2, 2] = u1[0, 0]
    g[2, 3] = u1[0, 1]
    g[3, 2] = u1[1, 0]
    g[3, 3] = u1[1, 1]
    return g


def apply_h(state: torch.Tensor, qubit: int, dtype: torch.dtype) -> torch.Tensor:
    return apply_single_qubit_gate(state, _h_matrix(dtype, state.device), qubit)


def apply_rz(state: torch.Tensor, theta: torch.Tensor, qubit: int, dtype: torch.dtype) -> torch.Tensor:
    return apply_single_qubit_gate(state, rz_matrix(theta, dtype), qubit)


def apply_rx(state: torch.Tensor, theta: torch.Tensor, qubit: int, dtype: torch.dtype) -> torch.Tensor:
    return apply_single_qubit_gate(state, rx_matrix(theta, dtype), qubit)


def apply_zz_phase(state: torch.Tensor, theta: torch.Tensor, q1: int, q2: int,
                   dtype: torch.dtype) -> torch.Tensor:
    """Apply exp(-i theta/2 Z_q1 Z_q2). Diagonal in computational basis.

    ``theta`` may be a scalar tensor or a per-sample batched tensor of shape
    ``(batch,)``; the resulting gate is broadcast accordingly.
    """
    half = (theta / 2.0).to(dtype)
    e_minus = torch.exp(-1j * half)
    e_plus = torch.exp(1j * half)
    zero = torch.zeros_like(e_minus)
    if e_minus.ndim == 0:
        g = torch.stack(
            [
                torch.stack([e_minus, zero, zero, zero]),
                torch.stack([zero, e_plus, zero, zero]),
                torch.stack([zero, zero, e_plus, zero]),
                torch.stack([zero, zero, zero, e_minus]),
            ]
        )
    else:
        row0 = torch.stack([e_minus, zero, zero, zero], dim=-1)
        row1 = torch.stack([zero, e_plus, zero, zero], dim=-1)
        row2 = torch.stack([zero, zero, e_plus, zero], dim=-1)
        row3 = torch.stack([zero, zero, zero, e_minus], dim=-1)
        g = torch.stack([row0, row1, row2, row3], dim=-2)
    return apply_two_qubit_gate(state, g, q1, q2)


def apply_cnot(state: torch.Tensor, control: int, target: int, dtype: torch.dtype) -> torch.Tensor:
    return apply_two_qubit_gate(state, cnot_gate(dtype, state.device), control, target)


def apply_crx(state: torch.Tensor, theta: torch.Tensor, control: int, target: int,
              dtype: torch.dtype) -> torch.Tensor:
    rot = rx_matrix(theta, dtype)
    return apply_two_qubit_gate(state, controlled_rotation_gate(rot, dtype), control, target)


def apply_crz(state: torch.Tensor, theta: torch.Tensor, control: int, target: int,
              dtype: torch.dtype) -> torch.Tensor:
    rot = rz_matrix(theta, dtype)
    return apply_two_qubit_gate(state, controlled_rotation_gate(rot, dtype), control, target)


def apply_pooling_block(state: torch.Tensor, params4: torch.Tensor, control: int,
                        target: int, dtype: torch.dtype) -> torch.Tensor:
    """Apply the deferred-measurement pooling block on (control, target).

    ``params4`` has shape ``(4,)`` and is interpreted as
    ``(theta0, phi0, theta1, phi1)``: outcome-0 RZ/RX angles and outcome-1
    RZ/RX angles applied to the target qubit.
    """
    gate = feedforward_pooling_gate(params4[0], params4[2], params4[1], params4[3], dtype)
    return apply_two_qubit_gate(state, gate, control, target)


# ---------------------------------------------------------------------------
# Probabilities and partial measurement.
# ---------------------------------------------------------------------------
def marginal_probabilities(state: torch.Tensor, kept_qubits: Sequence[int]) -> torch.Tensor:
    """Return the marginal probability distribution over ``kept_qubits``.

    The remaining qubits are traced out by summing |amplitude|^2 over their
    indices. Output shape is ``(batch, 2, 2, ..., 2)`` with one axis per kept
    qubit, in the order given by ``kept_qubits``.
    """
    probs = (state.real**2 + state.imag**2)
    n_qubits = state.ndim - 1
    all_qubits = list(range(n_qubits))
    kept = list(kept_qubits)
    discarded = [q for q in all_qubits if q not in kept]
    # Sum over discarded qubit axes (axis q+1 of state).
    if discarded:
        probs = probs.sum(dim=tuple(q + 1 for q in discarded))
    # Now probs has axes (batch, *(qubits in remaining order)). The "remaining
    # order" after summing keeps the original ordering of kept axes; permute to
    # the user-requested order.
    remaining_order = [q for q in all_qubits if q in kept]
    # Build permutation that maps remaining_order -> kept.
    perm_axes = [0] + [remaining_order.index(q) + 1 for q in kept]
    probs = probs.permute(*perm_axes).contiguous()
    return probs


__all__ = [
    "init_state",
    "apply_single_qubit_gate",
    "apply_two_qubit_gate",
    "apply_h",
    "apply_rz",
    "apply_rx",
    "apply_zz_phase",
    "apply_cnot",
    "apply_crx",
    "apply_crz",
    "apply_pooling_block",
    "rx_matrix",
    "rz_matrix",
    "cnot_gate",
    "controlled_rotation_gate",
    "feedforward_pooling_gate",
    "marginal_probabilities",
]
