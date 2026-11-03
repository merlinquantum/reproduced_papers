"""Batched exact statevector simulation in plain PyTorch.

The paper simulates every circuit with TorchQuantum statevectors and exact
expectation values (§IV-A). This module provides the same noiseless semantics
for the few gates the paper uses, differentiable by autograd.

State layout: a tensor of shape ``(batch, 2, 2, ..., 2)`` with one axis per
qubit; qubit ``k`` is tensor axis ``k + 1``. Flattening the qubit axes gives
the computational-basis index with qubit 0 as the most significant bit, so
amplitude encoding of a vector ``x`` puts ``x[i]`` on basis state ``|i>`` in
that big-endian order.
"""

from __future__ import annotations

import math

import torch


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """Complex dtype matching a real dtype (float32 -> complex64)."""
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def zero_state(
    batch: int, n_qubits: int, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    """Batch of ``|0...0>`` states."""
    state = torch.zeros(
        (batch, 2**n_qubits), dtype=complex_dtype_for(dtype), device=device
    )
    state[:, 0] = 1.0
    return state.reshape((batch,) + (2,) * n_qubits)


def amplitude_state(
    vectors: torch.Tensor, n_qubits: int, eps: float = 1e-12
) -> torch.Tensor:
    """Amplitude-encode real vectors of length ``2**n_qubits`` as ``x / ||x||``."""
    if vectors.shape[-1] != 2**n_qubits:
        raise ValueError(
            f"Amplitude encoding on {n_qubits} qubits needs {2**n_qubits} features, got {vectors.shape[-1]}"
        )
    normalized = vectors / (vectors.norm(dim=-1, keepdim=True) + eps)
    return normalized.to(complex_dtype_for(vectors.dtype)).reshape(
        (vectors.shape[0],) + (2,) * n_qubits
    )


def ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    """``RY(theta) = exp(-i theta Y / 2)`` for any leading batch shape, shape ``(..., 2, 2)``."""
    cos, sin = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack(
        [torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2
    ).to(complex_dtype_for(theta.dtype))


def rz_matrix(theta: torch.Tensor) -> torch.Tensor:
    """``RZ(theta) = diag(exp(-i theta/2), exp(i theta/2))``, shape ``(..., 2, 2)``."""
    phase = torch.exp(-0.5j * theta.to(complex_dtype_for(theta.dtype)))
    zero = torch.zeros_like(phase)
    return torch.stack(
        [torch.stack([phase, zero], -1), torch.stack([zero, phase.conj()], -1)], -2
    )


def apply_single_qubit(
    state: torch.Tensor, matrix: torch.Tensor, qubit: int
) -> torch.Tensor:
    """Apply a 2x2 gate to one qubit.

    ``matrix`` is either shared ``(2, 2)`` or per-sample ``(batch, 2, 2)``.
    """
    axis = qubit + 1
    moved = state.movedim(axis, -1)
    if matrix.dim() == 2:
        updated = moved @ matrix.transpose(0, 1)
    else:
        extra_dims = moved.dim() - 2
        per_sample = matrix.reshape((matrix.shape[0],) + (1,) * extra_dims + (2, 2))
        updated = (per_sample @ moved.unsqueeze(-1)).squeeze(-1)
    return updated.movedim(-1, axis)


def cz_phase(
    n_qubits: int, pairs: list[tuple[int, int]], dtype: torch.dtype, device
) -> torch.Tensor:
    """Diagonal of the product of CZ gates on ``pairs``, shaped like one state."""
    basis = torch.arange(2**n_qubits, device=device)
    bits = [(basis >> (n_qubits - 1 - qubit)) & 1 for qubit in range(n_qubits)]
    parity = torch.zeros_like(basis)
    for control, target in pairs:
        parity ^= bits[control] & bits[target]
    phase = (1 - 2 * parity).to(complex_dtype_for(dtype))
    return phase.reshape((2,) * n_qubits)


def entangling_pairs(n_qubits: int, topology: str) -> list[tuple[int, int]]:
    """Nearest-neighbour CZ pairs: ``"chain"`` (0-1, 1-2, ...) or ``"ring"`` (chain plus last-first)."""
    pairs = [(qubit, qubit + 1) for qubit in range(n_qubits - 1)]
    if topology == "ring" and n_qubits > 2:
        pairs.append((n_qubits - 1, 0))
    elif topology not in {"chain", "ring"}:
        raise ValueError(f"Unknown entangler topology {topology!r}")
    return pairs


def expectation_z(state: torch.Tensor) -> torch.Tensor:
    """``<Z_k>`` for every qubit, shape ``(batch, n_qubits)``."""
    probabilities = state.abs() ** 2
    n_qubits = state.dim() - 1
    values = []
    for qubit in range(n_qubits):
        other_axes = [axis for axis in range(1, n_qubits + 1) if axis != qubit + 1]
        marginal = probabilities.sum(dim=other_axes) if other_axes else probabilities
        values.append(marginal[:, 0] - marginal[:, 1])
    return torch.stack(values, dim=1)


def flatten(state: torch.Tensor) -> torch.Tensor:
    """State as ``(batch, 2**n)`` amplitudes in big-endian basis order."""
    return state.reshape(state.shape[0], -1)


def haar_random_unitary(
    dimension: int, generator: torch.Generator, dtype: torch.dtype
) -> torch.Tensor:
    """Haar-random unitary via QR of a complex Gaussian matrix (Mezzadri 2007)."""
    real = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(
        dimension, dimension, generator=generator, dtype=torch.float64
    )
    q, r = torch.linalg.qr(torch.complex(real, imaginary) / math.sqrt(2.0))
    diagonal = torch.diagonal(r)
    q = q * (diagonal / diagonal.abs()).unsqueeze(0)
    return q.to(complex_dtype_for(dtype))
