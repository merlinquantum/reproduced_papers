"""Minimal batched 8-qubit statevector simulator written in plain PyTorch.

It replaces TorchQuantum, which the paper used but whose pinned qiskit stack does
not install cleanly (see LOG.md, "Dependency Additions"). Only the gates the
paper's circuits need are provided: RX, RY, and a cyclic CZ ring.

Conventions (identical to TorchQuantum):

* wire 0 is the most significant bit of the basis-state index;
* ``RX(t) = exp(-i t X / 2)`` and ``RY(t) = exp(-i t Y / 2)``;
* a state is a complex tensor of shape ``(batch, 16, 16)`` where the row index
  is wires 0-3 and the column index is wires 4-7. A product of single-qubit gates
  then acts as ``A @ state @ B.T`` with ``A`` (resp. ``B``) the Kronecker product
  of the gates on wires 0-3 (resp. 4-7), which is far cheaper than a 256x256
  matrix product.
"""

from __future__ import annotations

import torch

N_WIRES = 8
HALF = N_WIRES // 2
HALF_DIM = 2**HALF  # 16


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """Return the complex dtype matching a real dtype (float32 -> complex64)."""
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def rx_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RX gate matrices for angles of any shape; result shape is ``(*shape, 2, 2)``."""
    cos, sin = torch.cos(theta / 2), torch.sin(theta / 2)
    zero = torch.zeros_like(cos)
    real = torch.stack([cos, zero, zero, cos], dim=-1)
    imag = torch.stack([zero, -sin, -sin, zero], dim=-1)
    return torch.complex(real, imag).reshape(*theta.shape, 2, 2)


def ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RY gate matrices for angles of any shape; result shape is ``(*shape, 2, 2)``."""
    cos, sin = torch.cos(theta / 2), torch.sin(theta / 2)
    real = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(*theta.shape, 2, 2)
    return torch.complex(real, torch.zeros_like(real))


def kron_chain(gates: torch.Tensor) -> torch.Tensor:
    """Kronecker product over the second-to-last batch axis.

    Parameters
    ----------
    gates : torch.Tensor
        Shape ``(..., n, 2, 2)``; gate ``0`` becomes the most significant factor.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 2**n, 2**n)``.
    """
    result = gates[..., 0, :, :]
    for index in range(1, gates.shape[-3]):
        right = gates[..., index, :, :]
        size_l, size_r = result.shape[-1], right.shape[-1]
        result = torch.einsum("...ab,...cd->...acbd", result, right)
        result = result.reshape(*result.shape[:-4], size_l * size_r, size_l * size_r)
    return result


def cz_ring_signs(device: torch.device | str = "cpu") -> torch.Tensor:
    """Diagonal of the cyclic CZ ring (CZ on wires i, i+1 mod 8), shape ``(16, 16)``."""
    indices = torch.arange(2**N_WIRES)
    bits = [(indices >> (N_WIRES - 1 - wire)) & 1 for wire in range(N_WIRES)]
    parity = sum(bits[wire] * bits[(wire + 1) % N_WIRES] for wire in range(N_WIRES))
    signs = 1.0 - 2.0 * (parity % 2).to(torch.float32)
    return signs.reshape(HALF_DIM, HALF_DIM).to(device)


def zero_state(
    batch: int, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    """The |0...0> state for a batch, shape ``(batch, 16, 16)``."""
    state = torch.zeros(batch, HALF_DIM, HALF_DIM, dtype=dtype, device=device)
    state[:, 0, 0] = 1.0
    return state


def apply_wire_gates(state: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Apply one single-qubit gate on every wire.

    Parameters
    ----------
    state : torch.Tensor
        Shape ``(batch, 16, 16)``.
    gates : torch.Tensor
        Shape ``(8, 2, 2)`` (shared by the batch) or ``(batch, 8, 2, 2)``
        (per-sample, used for data encoding).
    """
    top = kron_chain(gates[..., :HALF, :, :])
    bottom = kron_chain(gates[..., HALF:, :, :])
    if gates.dim() == 3:
        return torch.einsum("ij,bjk,lk->bil", top, state, bottom)
    return torch.einsum("bij,bjk,blk->bil", top, state, bottom)


def prepare_parameter_layers(
    angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the per-layer Kronecker blocks for a stack of trainable layers.

    Each layer applies RX on all wires then RY on all wires, i.e. the per-wire
    gate ``RY @ RX``.

    Parameters
    ----------
    angles : torch.Tensor
        Shape ``(layers, 2, 8)``: ``[layer, 0, wire]`` is the RX angle and
        ``[layer, 1, wire]`` the RY angle.

    Returns
    -------
    tuple of torch.Tensor
        ``top`` and ``bottom`` blocks, each of shape ``(layers, 16, 16)``.
    """
    per_wire = ry_matrix(angles[:, 1]) @ rx_matrix(angles[:, 0])
    return kron_chain(per_wire[:, :HALF]), kron_chain(per_wire[:, HALF:])


def run_trainable_layers(
    state: torch.Tensor,
    angles: torch.Tensor,
    cz_signs: torch.Tensor,
    *,
    cz_after_last: bool,
) -> torch.Tensor:
    """Apply consecutive trainable layers, each followed by the CZ ring.

    Parameters
    ----------
    state : torch.Tensor
        Shape ``(batch, 16, 16)``.
    angles : torch.Tensor
        Shape ``(layers, 2, 8)``.
    cz_signs : torch.Tensor
        Output of :func:`cz_ring_signs`.
    cz_after_last : bool
        The paper's code omits the CZ ring after the very last layer of the
        final block.
    """
    top, bottom = prepare_parameter_layers(angles)
    signs = cz_signs.to(state.dtype)
    last = angles.shape[0] - 1
    for layer in range(angles.shape[0]):
        state = torch.einsum("ij,bjk,lk->bil", top[layer], state, bottom[layer])
        if layer < last or cz_after_last:
            state = state * signs
    return state


def apply_cz_ring(state: torch.Tensor, cz_signs: torch.Tensor) -> torch.Tensor:
    """Apply the cyclic CZ ring."""
    return state * cz_signs.to(state.dtype)


def expectation_z(state: torch.Tensor, wires: list[int]) -> torch.Tensor:
    """<Z> on the given wires, shape ``(batch, len(wires))``."""
    probabilities = (state.real**2 + state.imag**2).reshape(state.shape[0], -1)
    indices = torch.arange(2**N_WIRES, device=state.device)
    columns = []
    for wire in wires:
        bit = (indices >> (N_WIRES - 1 - wire)) & 1
        sign = 1.0 - 2.0 * bit.to(probabilities.dtype)
        columns.append(probabilities @ sign)
    return torch.stack(columns, dim=1)


def expectation_x(state: torch.Tensor, wires: list[int]) -> torch.Tensor:
    """<X> on the given wires, shape ``(batch, len(wires))``."""
    batch = state.shape[0]
    flat = state.reshape(batch, 2**N_WIRES)
    columns = []
    for wire in wires:
        view = flat.reshape(batch, 2**wire, 2, 2 ** (N_WIRES - 1 - wire))
        overlap = torch.conj(view[:, :, 0]) * view[:, :, 1]
        columns.append(2.0 * overlap.real.sum(dim=(1, 2)))
    return torch.stack(columns, dim=1)
