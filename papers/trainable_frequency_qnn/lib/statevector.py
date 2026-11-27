"""Batched qubit statevector simulator in PyTorch.

The paper's models are noiseless gate-model QNNs on at most 6 qubits, so a
dense statevector of 2**N complex amplitudes per sample is exact and cheap.
Everything is differentiable with autograd, including higher-order derivatives
with respect to the encoded inputs, which the Navier-Stokes loss needs.

Conventions
-----------
- A batch of states has shape ``(batch, 2, 2, ..., 2)`` with one axis per qubit;
  axis ``1 + q`` is qubit ``q``. Qubit 0 is the top wire in the paper figures.
- ``R_a(angle) = exp(-i * angle * sigma_a / 2)`` for ``a`` in ``{x, y}``, so
  ``R_y(theta * x)`` contributes frequencies ``{0, theta}`` to the output, as in
  the paper's Sec. V spectrum argument (natural gaps {1, 2, 3} for N = 3).
"""

from __future__ import annotations

import torch


def zero_state(
    batch_size: int, n_qubits: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """Return ``batch_size`` copies of ``|0...0>``.

    Parameters
    ----------
    batch_size : int
        Number of independent states.
    n_qubits : int
        Number of qubits N.
    dtype : torch.dtype
        Complex dtype of the amplitudes.
    device : torch.device or str
        Target device.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch_size, 2, ..., 2)``.
    """
    state = torch.zeros((batch_size,) + (2,) * n_qubits, dtype=dtype, device=device)
    state.view(batch_size, -1)[:, 0] = 1.0
    return state


def rotation_matrix(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Build ``exp(-i angle sigma_axis / 2)`` for a scalar or a batch of angles.

    Parameters
    ----------
    axis : str
        ``"x"`` or ``"y"``.
    angle : torch.Tensor
        Real tensor of shape ``()`` or ``(batch,)``.

    Returns
    -------
    torch.Tensor
        Complex tensor of shape ``angle.shape + (2, 2)``.
    """
    half = angle / 2
    cos = torch.cos(half)
    sin = torch.sin(half)
    zero = torch.zeros_like(cos)
    if axis == "y":
        real = torch.stack(
            [torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2
        )
        return torch.complex(real, torch.zeros_like(real))
    if axis == "x":
        real = torch.stack(
            [torch.stack([cos, zero], -1), torch.stack([zero, cos], -1)], -2
        )
        imag = torch.stack(
            [torch.stack([zero, -sin], -1), torch.stack([-sin, zero], -1)], -2
        )
        return torch.complex(real, imag)
    raise ValueError(f"Unsupported rotation axis {axis!r}; expected 'x' or 'y'.")


def apply_single_qubit(
    state: torch.Tensor, matrix: torch.Tensor, qubit: int
) -> torch.Tensor:
    """Apply a 2x2 gate, shared or per-sample, to one qubit.

    Parameters
    ----------
    state : torch.Tensor
        Batched state ``(batch, 2, ..., 2)``.
    matrix : torch.Tensor
        Gate of shape ``(2, 2)`` (shared) or ``(batch, 2, 2)`` (per sample).
    qubit : int
        Target qubit index.

    Returns
    -------
    torch.Tensor
        New state with the same shape.
    """
    moved = torch.movedim(state, 1 + qubit, -1)
    if matrix.dim() == 2:
        updated = moved @ matrix.transpose(0, 1)
    else:
        extra_axes = moved.dim() - 2
        batched = matrix.transpose(-1, -2).reshape(
            (matrix.shape[0],) + (1,) * extra_axes + (2, 2)
        )
        updated = (moved.unsqueeze(-2) @ batched).squeeze(-2)
    return torch.movedim(updated, -1, 1 + qubit)


def apply_cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    """Apply CNOT by flipping the target axis on the control = 1 slice."""
    control_axis = 1 + control
    target_axis = 1 + target
    control_zero = state.narrow(control_axis, 0, 1)
    control_one = state.narrow(control_axis, 1, 1).flip(target_axis)
    return torch.cat([control_zero, control_one], dim=control_axis)


def apply_cz(
    state: torch.Tensor, qubit_a: int, qubit_b: int, n_qubits: int
) -> torch.Tensor:
    """Apply CZ, i.e. a -1 phase on ``|1>_a |1>_b``.

    The paper's sDAQC entangler ``exp(i pi n_k n_l)`` equals this gate.
    """
    sign_shape = [1] * (n_qubits + 1)
    sign_shape[1 + qubit_a] = 2
    sign = torch.ones((2, 2), dtype=state.real.dtype, device=state.device)
    sign[1, 1] = -1.0
    if qubit_a < qubit_b:
        sign_shape[1 + qubit_b] = 2
        sign = sign.reshape(sign_shape)
    else:
        sign_shape[1 + qubit_b] = 2
        sign = sign.t().reshape(sign_shape)
    return state * sign


def magnetization_diagonal(n_qubits: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Return the diagonal of ``sum_m Z_m`` in the computational basis."""
    basis_index = torch.arange(2**n_qubits, device=device)
    shifts = torch.arange(n_qubits - 1, -1, -1, device=device)
    bits = (basis_index.unsqueeze(1) >> shifts) & 1
    return (1 - 2 * bits).sum(dim=1).to(dtype)


def total_magnetization(state: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Return ``<sum_m Z_m>`` for every state in the batch, shape ``(batch,)``."""
    probabilities = (state.real**2 + state.imag**2).reshape(state.shape[0], -1)
    return probabilities @ magnetization_diagonal(
        n_qubits, probabilities.dtype, state.device
    )


def y_eigenbasis(
    n_qubits: int, dtype: torch.dtype, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the product eigenbasis of Pauli Y and its eigenvalue signs.

    ``R_y(a) = V diag(exp(-i a / 2), exp(+i a / 2)) V^dagger`` with
    ``V = [|+i>, |-i>]``, so any product of R_y gates is diagonal in
    ``W = V ⊗ ... ⊗ V`` (qubit 0 most significant).

    Returns
    -------
    tuple of torch.Tensor
        ``W`` of shape ``(2**N, 2**N)`` (columns are eigenvectors) and the sign
        matrix of shape ``(2**N, N)`` whose entry ``(k, m)`` is the Y eigenvalue
        (+1 or -1) of qubit ``m`` in eigenvector ``k``.
    """
    single = torch.tensor([[1, 1], [1j, -1j]], dtype=dtype, device=device) / (2**0.5)
    basis = single
    for _ in range(1, n_qubits):
        basis = torch.kron(basis, single)
    index = torch.arange(2**n_qubits, device=device)
    shifts = torch.arange(n_qubits - 1, -1, -1, device=device)
    bits = (index.unsqueeze(1) >> shifts) & 1
    return basis, 1 - 2 * bits


def dense_operator(
    apply_to_batch, n_qubits: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """Return the ``2**N x 2**N`` matrix of a state-batch map, column j = map(|j>).

    Parameters
    ----------
    apply_to_batch : callable
        Function mapping a batch of states ``(batch, 2, ..., 2)`` to new states.
    n_qubits : int
        Number of qubits N.
    dtype : torch.dtype
        Complex dtype.
    device : torch.device or str
        Target device.

    Returns
    -------
    torch.Tensor
        Matrix ``U`` such that ``new_flat_state = U @ flat_state``.
    """
    dim = 2**n_qubits
    basis = torch.eye(dim, dtype=dtype, device=device).reshape((dim,) + (2,) * n_qubits)
    return apply_to_batch(basis).reshape(dim, dim).transpose(0, 1)
