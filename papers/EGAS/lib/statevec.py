"""Batched, differentiable statevector simulator for the EGAS gate pool.

The paper (arXiv:2605.30866) embeds an n-qubit pure state |psi_Phi(x)> per input x and
evaluates pairwise fidelity F = |<psi_i|psi_j>|^2.  Because every circuit produces a pure
state, the full N x N fidelity matrix for a batch is just ``|Psi Psi^dagger|^2`` where the
rows of ``Psi`` are the per-sample statevectors.  We therefore (1) evaluate statevectors for
all samples in one batched pass and (2) form the Gram matrix once.

Everything is implemented in ``torch`` complex tensors so the same engine is used for the
analytic EGAS energy (``torch.no_grad``) and for the continuous bias-refinement stage, where
gradients must flow through the gate angles ``phi = r*x + b_omega(x)``.

Gate set (Appendix A): single-qubit {RX, RY, RZ, H, I}, two-qubit {CNOT, MultiRZ}.
Conventions match PennyLane (validated in tests):
    RX(t)=exp(-i t/2 X), RY(t)=exp(-i t/2 Y), RZ(t)=exp(-i t/2 Z),
    MultiRZ(t,[a,b])=exp(-i t/2 Z_a Z_b), CNOT(c,t) standard.
"""

from __future__ import annotations

import torch

CDTYPE = torch.complex128
RDTYPE = torch.float64


def _to_tensor(angle, batch: int, device) -> torch.Tensor:
    """Broadcast a scalar / (batch,) angle to shape (batch,) real tensor."""
    if not torch.is_tensor(angle):
        angle = torch.tensor(angle, dtype=RDTYPE, device=device)
    angle = angle.to(device=device, dtype=RDTYPE)
    if angle.dim() == 0:
        angle = angle.expand(batch)
    return angle


def init_state(batch: int, n_qubits: int, device="cpu") -> torch.Tensor:
    """|0...0> for each item in the batch. Shape (batch, 2**n)."""
    dim = 2**n_qubits
    state = torch.zeros(batch, dim, dtype=CDTYPE, device=device)
    state[:, 0] = 1.0
    return state


def _apply_1q(
    state: torch.Tensor, mats: torch.Tensor, wire: int, n_qubits: int
) -> torch.Tensor:
    """Apply a (batch,2,2) single-qubit operator on `wire` to (batch,2**n) state."""
    batch = state.shape[0]
    shape = (batch,) + (2,) * n_qubits
    st = state.reshape(shape)
    # move the target wire axis (offset by 1 for batch dim) to the last position
    axis = wire + 1
    st = st.movedim(axis, -1)  # (batch, ..., 2)
    out = torch.einsum("bij,b...j->b...i", mats, st)
    out = out.movedim(-1, axis)
    return out.reshape(batch, 2**n_qubits)


def _rot_mats(kind: str, angle: torch.Tensor) -> torch.Tensor:
    """Return (batch,2,2) rotation matrices for RX/RY/RZ."""
    c = torch.cos(angle / 2).to(CDTYPE)
    s = torch.sin(angle / 2).to(CDTYPE)
    i = torch.tensor(1j, dtype=CDTYPE, device=angle.device)
    z = torch.zeros_like(c)
    if kind == "RX":
        return torch.stack(
            [torch.stack([c, -i * s], -1), torch.stack([-i * s, c], -1)], -2
        )
    if kind == "RY":
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    if kind == "RZ":
        e_m = torch.exp(-i * angle / 2)
        e_p = torch.exp(i * angle / 2)
        return torch.stack([torch.stack([e_m, z], -1), torch.stack([z, e_p], -1)], -2)
    raise ValueError(kind)


_H = torch.tensor([[1, 1], [1, -1]], dtype=CDTYPE) / (2**0.5)


def apply_gate(
    state: torch.Tensor, n_qubits: int, gate: str, wires, angle=None
) -> torch.Tensor:
    """Apply one gate (possibly batched angle) to the batched state."""
    batch = state.shape[0]
    device = state.device
    if gate in ("RX", "RY", "RZ"):
        a = _to_tensor(angle, batch, device)
        return _apply_1q(state, _rot_mats(gate, a), wires[0], n_qubits)
    if gate == "H":
        mats = _H.to(device).unsqueeze(0).expand(batch, 2, 2)
        return _apply_1q(state, mats, wires[0], n_qubits)
    if gate == "I":
        return state
    if gate == "CNOT":
        return _apply_cnot(state, wires[0], wires[1], n_qubits)
    if gate == "MultiRZ":
        a = _to_tensor(angle, batch, device)
        return _apply_multirz(state, a, wires, n_qubits)
    raise ValueError(f"unknown gate {gate}")


def _apply_cnot(state, control, target, n_qubits):
    batch = state.shape[0]
    shape = (batch,) + (2,) * n_qubits
    st = state.reshape(shape).clone()
    cax = control + 1
    tax = target + 1
    # select control==1 slice, flip target
    idx_c1 = [slice(None)] * (n_qubits + 1)
    idx_c1[cax] = 1
    sub = st[tuple(idx_c1)]  # control=1 block
    sub_flipped = sub.flip(dims=[tax if tax < cax else tax - 1])
    st[tuple(idx_c1)] = sub_flipped
    return st.reshape(batch, 2**n_qubits)


def _apply_multirz(state, angle, wires, n_qubits):
    """exp(-i angle/2 * Z_{w0} Z_{w1} ...) — diagonal in computational basis."""
    device = state.device
    dim = 2**n_qubits
    idx = torch.arange(dim, device=device)
    # parity of involved wires (qubit q is bit (n-1-q) in the index)
    parity = torch.zeros(dim, dtype=RDTYPE, device=device)
    for w in wires:
        bit = (idx >> (n_qubits - 1 - w)) & 1
        parity = parity + bit.to(RDTYPE)
    z_eig = 1.0 - 2.0 * (parity % 2.0)  # +1 if even number of 1s, -1 if odd
    i = torch.tensor(1j, dtype=CDTYPE, device=device)
    phase = torch.exp(-i * angle.unsqueeze(1) / 2 * z_eig.unsqueeze(0))  # (batch, dim)
    return state * phase


def fidelity_matrix(
    states_a: torch.Tensor, states_b: torch.Tensor = None
) -> torch.Tensor:
    """|<psi_i|psi_j>|^2 Gram matrix. If states_b is None, use states_a (square)."""
    if states_b is None:
        states_b = states_a
    overlap = states_a.conj() @ states_b.t()  # (Na, Nb) complex
    return (overlap.abs() ** 2).real
