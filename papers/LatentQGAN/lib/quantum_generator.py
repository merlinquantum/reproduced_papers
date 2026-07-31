"""Quantum generator for LatentQGAN.

Two parallel implementations of the same parameterised quantum circuit
are provided:

* ``qiskit_circuit(N, L)`` builds the Qiskit ``QuantumCircuit`` used as the
  authoritative spec. It mirrors the architecture described in
  Section IV.B of the paper:
      - N qubits per sub-generator (default 4 = NG=3 data + NA=1 ancilla).
      - L parametrised layers.
      - Input layer: RY(alpha_i) on each qubit (noise encoding).
      - Each layer l: RY(theta_{l,i}) on each qubit, then CZ on consecutive pairs.
      - Post-selection on the ancilla qubit being |0>, then renormalise.

* ``QuantumGeneratorTorch`` is a tensor reimplementation that exactly
  matches the Qiskit semantics but is differentiable through PyTorch
  autograd. Five sub-generators × L layers × N rotations = T*N*L params
  (the paper reports 5*4*7 = 140 with the default config).

A unit test verifies that both produce the same output (up to numerical
precision) on a sample input.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Qiskit spec
# ---------------------------------------------------------------------------


def qiskit_circuit(N: int = 4, L: int = 7):
    """Return a Qiskit parameterised circuit for one sub-generator.

    The returned circuit has parameters named ``alpha_i`` for the input
    encoding and ``theta_l_i`` for trainable rotations. Use
    ``circuit.assign_parameters({...})`` to bind values.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    qc = QuantumCircuit(N, name="latentqgan_subgen")
    alpha = ParameterVector("alpha", N)
    theta = ParameterVector("theta", N * L)
    # Input encoding layer: RY(alpha_i) on each qubit.
    for i in range(N):
        qc.ry(alpha[i], i)
    # L parametrised layers: RY rotations then CZ between consecutive qubits.
    for layer_idx in range(L):
        for i in range(N):
            qc.ry(theta[layer_idx * N + i], i)
        for i in range(N - 1):
            qc.cz(i, i + 1)
    return qc, alpha, theta


def qiskit_forward(
    alpha: np.ndarray, theta: np.ndarray, N: int = 4, L: int = 7, NA: int = 1
) -> np.ndarray:
    """Run the Qiskit circuit and return the post-selected probability vector.

    Output is a length-2**(N-NA) probability vector that sums to 1, obtained
    by post-selecting on the ancilla qubits being |0> and renormalising.

    Convention: in Qiskit, qubit 0 is the *lowest-order* bit of the bitstring.
    We treat the ancilla qubits as the *last* NA qubits (i.e. qubits N-NA..N-1),
    matching the PyTorch implementation below.
    """
    from qiskit.quantum_info import Statevector

    qc, alpha_pv, theta_pv = qiskit_circuit(N, L)
    bind = {alpha_pv[i]: float(alpha[i]) for i in range(N)}
    bind.update({theta_pv[i]: float(theta[i]) for i in range(N * L)})
    bound = qc.assign_parameters(bind)
    sv = Statevector.from_label("0" * N).evolve(bound)
    probs = np.abs(sv.data) ** 2
    # In Qiskit Statevector, basis state index i has bitstring written with
    # qubit 0 as the least significant bit. We treat ancilla qubits as the
    # last NA qubits => ancilla=0 corresponds to indices where the top NA
    # bits of i are 0, i.e. i < 2**(N-NA).
    NG = N - NA
    keep = probs[: 2**NG]
    s = keep.sum()
    if s <= 0:
        return np.ones(2**NG) / (2**NG)
    return keep / s


# ---------------------------------------------------------------------------
# PyTorch tensor reimplementation (autograd friendly)
# ---------------------------------------------------------------------------


def _ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    """Return a (..., 2, 2) RY matrix tensor for theta in radians."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    # [[c, -s], [s, c]]
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _apply_single_qubit_gate(
    state: torch.Tensor, gate: torch.Tensor, qubit: int, N: int
) -> torch.Tensor:
    """Apply a 2x2 gate to ``qubit`` of an N-qubit state vector ``state`` of shape (2,)*N (or (batch,) + (2,)*N).

    Convention: qubit 0 is the *least* significant bit (Qiskit convention).
    Internally we reshape the state to (..., 2, 2, ..., 2) with N trailing
    dims; qubit i corresponds to dim -(i+1) (i=0 is the last dim).
    """
    # gate: (2, 2)
    # state shape: (..., 2, 2, ..., 2) with N trailing dims
    dim = -(qubit + 1)
    # move target dim to last
    state_moved = torch.movedim(state, dim, -1)
    # apply: state_moved @ gate.T  (so output[..., j] = sum_i state_moved[..., i] * gate[j, i])
    out = torch.matmul(state_moved, gate.T)
    return torch.movedim(out, -1, dim)


def _apply_cz(state: torch.Tensor, q_ctrl: int, q_targ: int, N: int) -> torch.Tensor:
    """Apply a CZ gate between qubits ``q_ctrl`` and ``q_targ`` (symmetric).

    CZ flips the sign of the |11> component on the two qubits.
    """
    # We multiply by -1 the amplitudes where both qubits are 1.
    # state shape: (..., 2, 2, ..., 2) with N trailing dims.
    # Build a sign tensor of the same trailing shape.
    sign = torch.ones((2,) * N, dtype=state.dtype, device=state.device)
    idx = [slice(None)] * N
    idx[-(q_ctrl + 1)] = 1
    idx[-(q_targ + 1)] = 1
    sign[tuple(idx)] = -1.0
    return state * sign


class QuantumGeneratorTorch(nn.Module):
    """One sub-generator as a differentiable PyTorch module.

    Parameters
    ----------
    N : int
        Total qubits per circuit (default 4).
    NA : int
        Number of ancilla qubits (default 1). Ancillas are the *last* NA qubits.
    L : int
        Number of variational layers (default 7).
    """

    def __init__(self, N: int = 4, NA: int = 1, L: int = 7):
        super().__init__()
        assert NA < N
        self.N = N
        self.NA = NA
        self.NG = N - NA
        self.L = L
        # Trainable parameters: (L, N) thetas.
        self.theta = nn.Parameter(torch.randn(L, N) * 0.1)

    def _initial_state(self, batch: int, device, dtype) -> torch.Tensor:
        state = torch.zeros((batch,) + (2,) * self.N, dtype=dtype, device=device)
        # set |0...0> amplitude to 1
        idx = (slice(None),) + (0,) * self.N
        state[idx] = 1.0
        return state

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        alpha : (batch, N) tensor of input noise angles.

        Returns
        -------
        probs : (batch, 2**NG) post-selected probability vector that sums to 1
                per batch element.
        """
        assert alpha.shape[-1] == self.N
        batch = alpha.shape[0]
        dtype = torch.complex64
        device = alpha.device
        state = self._initial_state(batch, device, dtype)
        # Input encoding: RY(alpha_i) on each qubit.
        for i in range(self.N):
            # Build per-batch 2x2 RY gates and apply via einsum-like ops.
            # Vectorise: for efficiency, build (batch, 2, 2) and apply to per-batch state.
            ry = _ry_matrix(alpha[:, i]).to(dtype)  # (batch, 2, 2)
            dim = -(i + 1)
            state_moved = torch.movedim(state, dim, -1)
            shape = state_moved.shape
            # flatten the leading dims except batch and the trailing dim
            state_flat = state_moved.reshape(batch, -1, 2)  # (batch, M, 2)
            # apply: out[b, m, j] = sum_k state_flat[b, m, k] * ry[b, j, k]
            out = torch.einsum("bmk,bjk->bmj", state_flat, ry)
            state = torch.movedim(out.reshape(shape), -1, dim)
        # Variational layers.
        for layer_idx in range(self.L):
            for i in range(self.N):
                ry_gate = _ry_matrix(self.theta[layer_idx, i]).to(dtype)  # (2, 2)
                state = _apply_single_qubit_gate(state, ry_gate, i, self.N)
            for i in range(self.N - 1):
                state = _apply_cz(state, i, i + 1, self.N)
        # Compute probabilities, then post-select on ancilla qubits (last NA) = 0.
        # state shape: (batch, 2, 2, ..., 2). Qubit i corresponds to dim -(i+1).
        # ancillas are qubits NG..N-1 i.e. dims -(NG+1) ... -N (the leading data dims).
        probs = state.real**2 + state.imag**2  # (batch, 2, ..., 2)
        # Sum the ancilla = 0 slice: for each ancilla qubit (index NG..N-1), slice 0
        # at dim -(q+1).
        sliced = probs
        for q in range(self.NG, self.N):
            sliced = sliced.index_select(-(q + 1), torch.tensor([0], device=device))
        # Now sliced has shape (batch, 2, 2, ..., 2) with NG data qubits and NA singleton dims.
        # Flatten data qubits: data qubit i corresponds to dim -(i+1) for i in 0..NG-1.
        sliced = sliced.squeeze()  # remove singleton ancilla dims
        if sliced.dim() == 1 + self.NG - 0 - 1:
            # squeeze may have collapsed too much for batch=1; ensure batch dim present.
            pass
        # Reshape to (batch, 2**NG). The order of bits: we want index j to have
        # bit i = (j >> i) & 1, matching Qiskit basis-state ordering.
        # The slicing/squeeze leaves dimensions in order (batch, q_{NG-1}, ..., q_0)
        # because qubit i is dim -(i+1). So we need to reverse the trailing dims to
        # get (batch, q_0, q_1, ..., q_{NG-1}) and then flatten with little-endian bit order.
        # Easier: flatten then re-permute the basis indices.
        if sliced.dim() == 1:
            sliced = sliced.unsqueeze(0)
        # The current dim ordering for trailing axes: dim -1 corresponds to qubit 0
        # (since we kept indexing with -(i+1)); good — that's already little-endian.
        flat = sliced.reshape(batch, 2**self.NG)
        s = flat.sum(dim=-1, keepdim=True)
        flat = flat / (s + 1e-12)
        return flat
