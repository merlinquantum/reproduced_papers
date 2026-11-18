"""Batched exact statevector simulation of the paper's small qubit circuits.

Conventions (a wrong convention silently corrupts results):

* Basis ordering: amplitude index ``k = sum_i b_i 2^(n-1-i)``, i.e. qubit 0 is
  the most significant bit. Frozen Lake state 1 is ``|0001>`` (paper Sec. 3.1).
* Tensors carry a leading agent dimension ``A`` (independent agents trained
  in one process) and usually a batch dimension ``B``: states are
  ``(A, B, 2^n)`` complex.
* Gates follow the textbook definitions ``RX(t) = exp(-i t X / 2)`` etc.,
  identical to Cirq's ``rx``, ``ry``, ``rz`` used by the authors.

The simulation is exact (no shot noise), as in the paper's TensorFlow Quantum
runs, and differentiable with PyTorch autograd.
"""

from __future__ import annotations

import torch


def rx_matrices(angles: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    """``RX(angle)`` matrices with shape ``angles.shape + (2, 2)``."""
    cos = torch.cos(angles / 2).to(complex_dtype)
    sin = torch.sin(angles / 2).to(complex_dtype)
    minus_i_sin = -1j * sin
    return torch.stack(
        [
            torch.stack([cos, minus_i_sin], dim=-1),
            torch.stack([minus_i_sin, cos], dim=-1),
        ],
        dim=-2,
    )


def ry_matrices(angles: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    """``RY(angle)`` matrices with shape ``angles.shape + (2, 2)``."""
    cos = torch.cos(angles / 2).to(complex_dtype)
    sin = torch.sin(angles / 2).to(complex_dtype)
    return torch.stack(
        [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2
    )


def rz_matrices(angles: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    """``RZ(angle)`` matrices with shape ``angles.shape + (2, 2)``."""
    phase = torch.exp(-0.5j * angles.to(complex_dtype))
    zero = torch.zeros_like(phase)
    return torch.stack(
        [torch.stack([phase, zero], dim=-1), torch.stack([zero, phase.conj()], dim=-1)],
        dim=-2,
    )


def kron_qubits(single_qubit: torch.Tensor) -> torch.Tensor:
    """Tensor product of one 2x2 operator per qubit.

    Parameters
    ----------
    single_qubit : torch.Tensor
        Shape ``(..., n, 2, 2)``; entry ``[..., i]`` acts on qubit ``i``.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 2^n, 2^n)`` operator ``U_0 (x) U_1 (x) ... (x) U_{n-1}``
        in the qubit-0-most-significant ordering.
    """
    n_qubits = single_qubit.shape[-3]
    result = single_qubit[..., 0, :, :]
    for qubit in range(1, n_qubits):
        factor = single_qubit[..., qubit, :, :]
        rows, cols = result.shape[-2], result.shape[-1]
        result = (
            result[..., :, None, :, None] * factor[..., None, :, None, :]
        ).reshape(*result.shape[:-2], rows * 2, cols * 2)
    return result


def basis_bits(n_qubits: int) -> torch.Tensor:
    """Bit table ``(2^n, n)``; row ``k`` holds the bits of basis state ``k``."""
    indices = torch.arange(2**n_qubits)
    shifts = torch.arange(n_qubits - 1, -1, -1)
    return (indices[:, None] >> shifts) & 1


def cz_phase_diagonal(n_qubits: int, pairs: list[tuple[int, int]]) -> torch.Tensor:
    """Diagonal (+1 / -1) of a product of CZ gates on the given qubit pairs."""
    bits = basis_bits(n_qubits)
    diagonal = torch.ones(2**n_qubits)
    for first, second in pairs:
        diagonal = diagonal * (1 - 2 * (bits[:, first] * bits[:, second]))
    return diagonal


def ring_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """CZ daisy chain closed into a ring: (0,1), (1,2), ..., (n-1,0) (paper Fig. 1)."""
    return [(qubit, (qubit + 1) % n_qubits) for qubit in range(n_qubits)]


def all_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """CZ on every qubit pair (reference-code Frozen Lake ansatz)."""
    return [(first, second) for second in range(n_qubits) for first in range(second)]


def z_product_diagonal(n_qubits: int, qubits: tuple[int, ...]) -> torch.Tensor:
    """Diagonal of the Pauli-Z product acting on ``qubits``."""
    bits = basis_bits(n_qubits)
    diagonal = torch.ones(2**n_qubits)
    for qubit in qubits:
        diagonal = diagonal * (1 - 2 * bits[:, qubit])
    return diagonal


def readout_diagonals(readout: str, n_qubits: int) -> torch.Tensor:
    """Observable diagonals as columns, one per action.

    ``"cartpole_zz_pairs"``: ``O_left = Z0 Z1``, ``O_right = Z2 Z3`` (paper
    eq. 19). ``"single_z_per_qubit"``: ``O_a = Z_a`` (Frozen Lake, Sec. 5.1).

    Returns
    -------
    torch.Tensor
        Shape ``(2^n, n_actions)``.
    """
    if readout == "cartpole_zz_pairs":
        if n_qubits != 4:
            raise ValueError("cartpole_zz_pairs readout needs 4 qubits")
        columns = [z_product_diagonal(4, (0, 1)), z_product_diagonal(4, (2, 3))]
    elif readout == "single_z_per_qubit":
        columns = [z_product_diagonal(n_qubits, (qubit,)) for qubit in range(n_qubits)]
    else:
        raise ValueError(f"Unknown readout {readout!r}")
    return torch.stack(columns, dim=1)


def basis_states(
    indices: torch.Tensor, n_qubits: int, complex_dtype: torch.dtype
) -> torch.Tensor:
    """Computational basis states ``|indices>`` with shape ``indices.shape + (2^n,)``."""
    one_hot = torch.nn.functional.one_hot(indices.long(), num_classes=2**n_qubits)
    return one_hot.to(complex_dtype)


def zero_states(
    leading_shape: tuple[int, ...], n_qubits: int, complex_dtype: torch.dtype, device
) -> torch.Tensor:
    """``|0...0>`` with shape ``leading_shape + (2^n,)``."""
    state = torch.zeros(*leading_shape, 2**n_qubits, dtype=complex_dtype, device=device)
    state[..., 0] = 1
    return state


def apply_operator(state: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    """Apply ``operator`` (``(..., d, d)``, broadcast over batch) to ``state`` (``(..., d)``)."""
    return torch.matmul(operator, state.unsqueeze(-1)).squeeze(-1)


def expectations(state: torch.Tensor, diagonals: torch.Tensor) -> torch.Tensor:
    """Expectation values of diagonal observables.

    Parameters
    ----------
    state : torch.Tensor
        Shape ``(..., 2^n)`` complex.
    diagonals : torch.Tensor
        Shape ``(2^n, n_obs)`` real.

    Returns
    -------
    torch.Tensor
        Shape ``(..., n_obs)`` real.
    """
    probabilities = state.real**2 + state.imag**2
    return probabilities @ diagonals
