"""Pure-PyTorch statevector PQC for the HQ-CNN classifier.

The paper specifies a 10-qubit, ``L``-layer parameterised quantum circuit:

  - RY(phi_j) encoding for each qubit (``phi_j`` = encoder output),
  - per-layer block of ``RY(theta) → RZ(theta) → ring-CNOT``,
  - measurement of ``<Z_j>`` for each qubit, j = 0..n-1.

Implementing the PQC directly in PyTorch (rather than Qiskit/Pennylane) keeps
the package dependency-free for the smoke run, and lets gradients flow
through torch autograd. The simulator is straightforward dense statevector
evolution and is fast enough for small ``n`` (n <= 12 batches comfortably on
CPU). For larger ``n`` use a dedicated simulator.
"""

from __future__ import annotations

import torch
from torch import nn


def _zero_state(batch: int, n_qubits: int, device, dtype) -> torch.Tensor:
    state = torch.zeros(batch, 2**n_qubits, dtype=dtype, device=device)
    state[:, 0] = 1.0
    return state


def _apply_single_qubit(state: torch.Tensor, gate: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    """Apply a per-batch single-qubit gate.

    ``state`` has shape ``(batch, 2**n)``; ``gate`` has shape ``(batch, 2, 2)``.
    """
    batch = state.shape[0]
    # reshape state to (batch, 2^pre, 2, 2^post) where the middle 2 is the target qubit
    post = n_qubits - qubit - 1
    pre = qubit
    new_shape = (batch, 2**pre, 2, 2**post)
    s = state.view(new_shape)
    # contract on the target axis using the per-batch gate
    # s : (b, pre, 2, post), gate : (b, 2, 2) — out[k] = sum_l gate[k,l] * s[l]
    s_perm = s.permute(0, 1, 3, 2)  # (b, pre, post, 2)
    out = torch.einsum("bij,bpqj->bpqi", gate, s_perm)
    out = out.permute(0, 1, 3, 2).contiguous().view(batch, -1)
    return out


def _apply_cnot(state: torch.Tensor, control: int, target: int, n_qubits: int) -> torch.Tensor:
    """Apply CNOT(control → target) to a batched statevector."""
    dim = 2**n_qubits
    # Build the permutation of computational-basis indices and reuse across the batch.
    idx = torch.arange(dim, device=state.device)
    ctrl_bit = (idx >> (n_qubits - 1 - control)) & 1
    target_mask = 1 << (n_qubits - 1 - target)
    flipped = torch.where(ctrl_bit == 1, idx ^ target_mask, idx)
    return state.index_select(1, flipped)


def _ry(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta * 0.5)
    s = torch.sin(theta * 0.5)
    # gate shape (batch, 2, 2)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _rz(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta * 0.5)
    s = torch.sin(theta * 0.5)
    zero = torch.zeros_like(c)
    # exp(-i theta/2) on |0>, exp(+i theta/2) on |1>. We keep states *real* by
    # absorbing the global phase into a Pauli-Y-style rotation that preserves
    # the <Z> expectation. For statevector evolution we need a complex dtype;
    # see ``PQC.forward`` for the complex-cast.
    row0 = torch.stack([c - 1j * s, zero.to(c.dtype) * 0j], dim=-1)
    row1 = torch.stack([zero.to(c.dtype) * 0j, c + 1j * s], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _expand_real_to_complex(state: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(state):
        return state
    return state.to(torch.complex64)


class PQC(nn.Module):
    """Parameterised quantum circuit module.

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Default ``10`` matches the paper.
    n_layers : int
        Number of variational layers. Default ``4`` matches the paper.
    init : str
        Trainable-parameter initialisation. ``"uniform"`` draws from
        ``U(0, 2*pi)`` as in the paper, ``"normal"`` from ``N(0, 0.1)``.
    """

    def __init__(self, n_qubits: int = 10, n_layers: int = 4, init: str = "uniform") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        shape = (n_layers, n_qubits, 2)  # RY and RZ per qubit per layer
        if init == "uniform":
            params = torch.rand(shape) * (2 * torch.pi)
        elif init == "normal":
            params = torch.randn(shape) * 0.1
        else:
            raise ValueError(f"Unknown init: {init}")
        self.theta = nn.Parameter(params)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Run the PQC and return ``<Z_j>`` for each qubit.

        Parameters
        ----------
        encoded : torch.Tensor
            Shape ``(batch, n_qubits)`` — angle-encoded inputs in ``[0, pi]``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_qubits)`` real-valued ``<Z_j>`` in ``[-1, 1]``.
        """
        if encoded.shape[1] != self.n_qubits:
            raise ValueError(
                f"PQC expects {self.n_qubits} encoded angles, got {encoded.shape[1]}"
            )
        batch = encoded.shape[0]
        device = encoded.device
        state = _zero_state(batch, self.n_qubits, device, torch.complex64)

        # Angle encoding: RY(phi_j) on qubit j.
        for q in range(self.n_qubits):
            gate = _ry(encoded[:, q]).to(torch.complex64)
            state = _apply_single_qubit(state, gate, q, self.n_qubits)

        # Variational layers.
        for layer in range(self.n_layers):
            theta = self.theta[layer].to(device)
            # Single-qubit RY then RZ on every qubit.
            for q in range(self.n_qubits):
                ry_gate = _ry(theta[q, 0].expand(batch)).to(torch.complex64)
                state = _apply_single_qubit(state, ry_gate, q, self.n_qubits)
                rz_gate = _rz(theta[q, 1].expand(batch))
                state = _apply_single_qubit(state, rz_gate, q, self.n_qubits)
            # Ring CNOT entanglement.
            for q in range(self.n_qubits):
                state = _apply_cnot(state, q, (q + 1) % self.n_qubits, self.n_qubits)

        # Measure <Z_j>: probability of bit j being 0 minus prob of bit j = 1.
        probs = (state.real**2 + state.imag**2)
        idx = torch.arange(2**self.n_qubits, device=device)
        z_vals = []
        for q in range(self.n_qubits):
            bit = ((idx >> (self.n_qubits - 1 - q)) & 1).to(probs.dtype)
            # bit=0 -> +1, bit=1 -> -1
            sign = 1.0 - 2.0 * bit
            z_vals.append((probs * sign).sum(dim=1))
        return torch.stack(z_vals, dim=1)
