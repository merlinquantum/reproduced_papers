"""Pure-PyTorch state-vector simulator for the QEGM hardware-efficient ansatz.

The paper (Eq. 11) specifies a hardware-efficient ansatz of alternating
single-qubit ``Ry``/``Rz`` rotations followed by a linear CNOT chain. The
latent vector ``z`` is encoded as ``Ry(z_i)`` on each of the ``n_qubits``
data wires (Eq. 13). We follow that structure exactly. Because the paper
authors use the parameter-shift rule on hardware but plain backprop is
valid for any noiseless simulator, we use PyTorch autograd here and
document the equivalence in ``LOG.md``.

This module is kept independent of any quantum library to avoid pulling
in a heavy PennyLane / Qiskit dependency for a 4-qubit simulator.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _expand_single_qubit(op: torch.Tensor, target: int, n_qubits: int) -> torch.Tensor:
    """Embed a 2x2 ``op`` acting on ``target`` qubit into the full Hilbert space."""

    dim = 2**n_qubits
    out = torch.zeros((dim, dim), dtype=op.dtype, device=op.device)
    for i in range(dim):
        for j in range(dim):
            i_t = (i >> (n_qubits - 1 - target)) & 1
            j_t = (j >> (n_qubits - 1 - target)) & 1
            i_rest = i & ~(1 << (n_qubits - 1 - target))
            j_rest = j & ~(1 << (n_qubits - 1 - target))
            if i_rest == j_rest:
                out[i, j] = op[i_t, j_t]
    return out


def _make_cnot_matrix(control: int, target: int, n_qubits: int) -> torch.Tensor:
    """Build the controlled-NOT matrix on ``n_qubits`` for fixed control/target."""

    dim = 2**n_qubits
    rows = []
    cols = []
    vals = []
    for i in range(dim):
        c = (i >> (n_qubits - 1 - control)) & 1
        if c == 0:
            j = i
        else:
            j = i ^ (1 << (n_qubits - 1 - target))
        rows.append(j)
        cols.append(i)
        vals.append(1.0)
    out = torch.zeros((dim, dim), dtype=torch.complex64)
    for r, c, v in zip(rows, cols, vals):
        out[r, c] = v
    return out


class HardwareEfficientVQC(nn.Module):
    """Hardware-efficient ansatz matching Eq. 11 of the paper.

    The latent input ``z`` of length ``n_qubits`` is angle-encoded as
    ``Ry(z_i)`` on each data wire (Eq. 13). ``n_layers`` blocks of
    ``Ry(theta) Rz(phi)`` rotations followed by a linear CNOT chain
    follow, with all rotation angles being trainable parameters of the
    layer.

    The forward returns the per-qubit Pauli-Z expectation values, mapped
    into ``[0, 1]`` via ``(1 + <Z>) / 2`` so that they can be used
    directly as quantum-randomness modulation factors for noise injection
    (Eq. 7).
    """

    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        scale = 0.1
        self.theta = nn.Parameter(scale * torch.randn(self.n_layers, self.n_qubits))
        self.phi = nn.Parameter(scale * torch.randn(self.n_layers, self.n_qubits))
        self.register_buffer("cnots", self._build_cnot_chain())

    def _build_cnot_chain(self) -> torch.Tensor:
        chain = torch.eye(2**self.n_qubits, dtype=torch.complex64)
        for q in range(self.n_qubits - 1):
            chain = _make_cnot_matrix(q, q + 1, self.n_qubits) @ chain
        return chain

    @staticmethod
    def _ry(angles: torch.Tensor) -> torch.Tensor:
        c = torch.cos(angles / 2)
        s = torch.sin(angles / 2)
        ry = torch.stack(
            [
                torch.stack([c, -s], dim=-1),
                torch.stack([s, c], dim=-1),
            ],
            dim=-2,
        )
        return ry.to(torch.complex64)

    @staticmethod
    def _rz(angles: torch.Tensor) -> torch.Tensor:
        half = angles / 2
        e_neg = torch.exp(-1j * half.to(torch.complex64))
        e_pos = torch.exp(1j * half.to(torch.complex64))
        zero = torch.zeros_like(e_neg)
        rz = torch.stack(
            [
                torch.stack([e_neg, zero], dim=-1),
                torch.stack([zero, e_pos], dim=-1),
            ],
            dim=-2,
        )
        return rz

    def _layer_unitary(
        self, layer_idx: int, encoding_angles: torch.Tensor
    ) -> torch.Tensor:
        """Compose Ry(z_i) * Ry(theta) * Rz(phi) per qubit, then a CNOT chain.

        Returns a (dim, dim) complex matrix that acts on the full register.
        """

        n = self.n_qubits
        if layer_idx == 0:
            enc = self._ry(encoding_angles)
        else:
            enc = None
        ry = self._ry(self.theta[layer_idx])
        rz = self._rz(self.phi[layer_idx])
        per_qubit = []
        for q in range(n):
            single = rz[q] @ ry[q]
            if enc is not None:
                single = single @ enc[q]
            per_qubit.append(single)
        full = per_qubit[0]
        for q in range(1, n):
            full = torch.kron(full, per_qubit[q])
        return self.cnots @ full

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        batch = z.shape[0]
        dim = 2**self.n_qubits
        device = z.device

        outputs = []
        for b in range(batch):
            state = torch.zeros(dim, dtype=torch.complex64, device=device)
            state[0] = 1.0
            unitary = self._layer_unitary(0, z[b])
            state = unitary @ state
            for layer_idx in range(1, self.n_layers):
                state = self._layer_unitary(layer_idx, z[b]) @ state
            probs = (state.conj() * state).real
            z_expectations = []
            for q in range(self.n_qubits):
                shift = self.n_qubits - 1 - q
                contributions = torch.tensor(
                    [1.0 if ((i >> shift) & 1) == 0 else -1.0 for i in range(dim)],
                    dtype=probs.dtype,
                    device=probs.device,
                )
                z_expectations.append((probs * contributions).sum())
            outputs.append(torch.stack(z_expectations))

        z_exp = torch.stack(outputs)
        return 0.5 * (1.0 + z_exp)
