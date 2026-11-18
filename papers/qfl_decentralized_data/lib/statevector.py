"""Batched n-qubit statevector simulator in plain PyTorch.

It replaces TensorFlow Quantum, which the paper used but which is not installed in
this repository's environment (see LOG.md, "Dependency Additions"). Only what the
paper's circuits need is provided: single-qubit rotations, the two-qubit XX
rotation, diagonal phase operators and probability read-out.

Conventions

* a state is a complex tensor of shape ``(batch, 2, ..., 2)`` with one axis per
  qubit; wire 0 is the first axis, hence the most significant bit of a basis
  index;
* ``RY(t) = exp(-i t Y / 2)``, ``RZ(t) = exp(-i t Z / 2)``,
  ``RX(t) = exp(-i t X / 2)`` and ``RXX(t) = exp(-i t X⊗X / 2)``.
"""

from __future__ import annotations

import torch


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """Return the complex dtype matching a real dtype (float32 -> complex64)."""
    if real_dtype == torch.float64:
        return torch.complex128
    if real_dtype == torch.float32:
        return torch.complex64
    raise ValueError(f"Unsupported real dtype {real_dtype}; use float32 or float64")


def zero_state(
    batch: int, n_qubits: int, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    """Return ``|0...0>`` repeated ``batch`` times, shape ``(batch, 2, ..., 2)``."""
    state = torch.zeros(
        (batch, *([2] * n_qubits)), dtype=complex_dtype_for(dtype), device=device
    )
    state[(slice(None), *([0] * n_qubits))] = 1.0
    return state


def _to_complex(real: torch.Tensor) -> torch.Tensor:
    return torch.complex(real, torch.zeros_like(real))


def rx_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RX gates for angles of any shape; result shape ``(*shape, 2, 2)``."""
    cos, sin = torch.cos(theta / 2), torch.sin(theta / 2)
    zero = torch.zeros_like(cos)
    real = torch.stack([cos, zero, zero, cos], dim=-1)
    imag = torch.stack([zero, -sin, -sin, zero], dim=-1)
    return torch.complex(real, imag).reshape(*theta.shape, 2, 2)


def ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RY gates for angles of any shape; result shape ``(*shape, 2, 2)``."""
    cos, sin = torch.cos(theta / 2), torch.sin(theta / 2)
    real = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(*theta.shape, 2, 2)
    return _to_complex(real)


def rz_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RZ gates for angles of any shape; result shape ``(*shape, 2, 2)``."""
    cos, sin = torch.cos(theta / 2), torch.sin(theta / 2)
    zero = torch.zeros_like(cos)
    real = torch.stack([cos, zero, zero, cos], dim=-1)
    imag = torch.stack([-sin, zero, zero, sin], dim=-1)
    return torch.complex(real, imag).reshape(*theta.shape, 2, 2)


def rxx_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RXX gate for a scalar angle; result shape ``(4, 4)``."""
    cos, sin = torch.cos(theta / 2), torch.sin(theta / 2)
    identity = torch.eye(4, dtype=theta.dtype, device=theta.device)
    flip_both = torch.flip(identity, dims=[0])  # X⊗X permutes |ab> -> |a^1 b^1>
    return torch.complex(cos * identity, -sin * flip_both)


def apply_one_qubit(state: torch.Tensor, gate: torch.Tensor, wire: int) -> torch.Tensor:
    """Apply a gate to one wire.

    Parameters
    ----------
    state : torch.Tensor
        Shape ``(batch, 2, ..., 2)``.
    gate : torch.Tensor
        Shape ``(2, 2)`` (shared) or ``(batch, 2, 2)`` (one gate per sample).
    wire : int
        Target wire.

    Returns
    -------
    torch.Tensor
        The updated state, same shape as ``state``.
    """
    moved = state.movedim(wire + 1, -1)
    if gate.dim() == 2:
        out = torch.einsum("...a,ba->...b", moved, gate)
    else:
        out = torch.einsum("b...a,bca->b...c", moved, gate)
    return out.movedim(-1, wire + 1)


def apply_two_qubit(
    state: torch.Tensor, gate: torch.Tensor, wire_a: int, wire_b: int
) -> torch.Tensor:
    """Apply a shared ``(4, 4)`` gate to wires ``(wire_a, wire_b)``."""
    tensor = gate.reshape(2, 2, 2, 2)  # (out_a, out_b, in_a, in_b)
    moved = state.movedim((wire_a + 1, wire_b + 1), (-2, -1))
    out = torch.einsum("...ij,klij->...kl", moved, tensor)
    return out.movedim((-2, -1), (wire_a + 1, wire_b + 1))


def marginal_probabilities(state: torch.Tensor, wires: list[int]) -> torch.Tensor:
    """Joint outcome probabilities of ``wires``; shape ``(batch, 2**len(wires))``.

    The first listed wire is the most significant bit of the outcome index.
    """
    probabilities = state.real**2 + state.imag**2
    n_qubits = state.dim() - 1
    other_axes = tuple(axis + 1 for axis in range(n_qubits) if axis not in wires)
    reduced = probabilities.sum(dim=other_axes) if other_axes else probabilities
    kept_in_order = sorted(wires)
    permutation = [kept_in_order.index(wire) + 1 for wire in wires]
    reduced = reduced.permute(0, *permutation)
    return reduced.reshape(state.shape[0], -1)
