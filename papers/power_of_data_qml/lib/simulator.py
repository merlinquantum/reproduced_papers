"""Batched statevector primitives for the paper's gate sets.

States are torch tensors of shape ``(batch, 2, 2, ..., 2)`` with one axis per
qubit. Axis ``1 + j`` is qubit ``j``; index 0 on an axis is ``|0>`` (the +1
eigenstate of Z). Every gate acts on the whole batch at once, with a separate
angle per batch element, so one call evolves many data points through their own
data-dependent circuits. The same code runs on CPU and CUDA.
"""

from __future__ import annotations

import math

import torch


def zero_state(batch_size: int, n_qubits: int, *, dtype, device) -> torch.Tensor:
    """Return ``batch_size`` copies of ``|0...0>``.

    Parameters
    ----------
    batch_size : int
        Number of states.
    n_qubits : int
        Number of qubits.
    dtype : torch.dtype
        Complex dtype (``torch.complex64`` or ``torch.complex128``).
    device : torch.device or str
        Device holding the states.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch_size,) + (2,) * n_qubits``.
    """
    states = torch.zeros((batch_size,) + (2,) * n_qubits, dtype=dtype, device=device)
    states.view(batch_size, -1)[:, 0] = 1.0
    return states


def product_state(single_qubit_states: torch.Tensor) -> torch.Tensor:
    """Build tensor-product states from per-qubit amplitudes.

    Parameters
    ----------
    single_qubit_states : torch.Tensor
        Shape ``(batch, n_qubits, 2)``: amplitude of ``|0>`` and ``|1>`` per qubit.

    Returns
    -------
    torch.Tensor
        Shape ``(batch,) + (2,) * n_qubits``.
    """
    batch_size, n_qubits, _ = single_qubit_states.shape
    states = single_qubit_states[:, 0, :]
    for qubit in range(1, n_qubits):
        states = states.unsqueeze(-1) * single_qubit_states[:, qubit, :].reshape(
            (batch_size,) + (1,) * qubit + (2,)
        )
    return states


def _broadcast_angle(angle: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    return angle.reshape((-1,) + (1,) * (states.ndim - 1))


def apply_heisenberg_bond(
    states: torch.Tensor, qubit_a: int, qubit_b: int, theta: torch.Tensor
) -> torch.Tensor:
    """Apply ``exp(-i theta (X_a X_b + Y_a Y_b + Z_a Z_b))`` to each state.

    Uses ``XX + YY + ZZ = 2 SWAP - I``, so the gate equals
    ``exp(i theta) (cos(2 theta) I - i sin(2 theta) SWAP)``. This is the bond
    term of both the E3 embedding and the QNN of the paper (App. J).

    Parameters
    ----------
    states : torch.Tensor
        Batched states.
    qubit_a, qubit_b : int
        The two qubits of the bond.
    theta : torch.Tensor
        Real angles, shape ``(batch,)``.

    Returns
    -------
    torch.Tensor
        Evolved states (new tensor).
    """
    swapped = states.transpose(1 + qubit_a, 1 + qubit_b)
    theta = _broadcast_angle(theta.to(states.real.dtype), states)
    global_phase = torch.polar(torch.ones_like(theta), theta)
    return global_phase * (
        torch.cos(2 * theta) * states - 1j * torch.sin(2 * theta) * swapped
    )


def apply_hadamard_layer(states: torch.Tensor) -> torch.Tensor:
    """Apply a Hadamard gate to every qubit (a Walsh-Hadamard transform)."""
    n_qubits = states.ndim - 1
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for qubit in range(n_qubits):
        axis = 1 + qubit
        zero_part, one_part = states.unbind(dim=axis)
        states = torch.stack(
            ((zero_part + one_part) * inv_sqrt2, (zero_part - one_part) * inv_sqrt2),
            dim=axis,
        )
    return states


def z_eigenvalue_table(n_qubits: int, *, dtype, device) -> torch.Tensor:
    """Return ``z[b, j] = +1/-1``, the Z_j eigenvalue of computational basis state b.

    Basis index ``b`` follows the state layout: qubit 0 is the most significant bit.
    """
    basis = torch.arange(2**n_qubits, device=device)
    shifts = torch.arange(n_qubits - 1, -1, -1, device=device)
    bits = (basis.unsqueeze(1) >> shifts) & 1
    return (1 - 2 * bits).to(dtype)


def apply_diagonal_phase(states: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    """Multiply each state by ``exp(i phases[b])`` on basis state ``b``.

    Parameters
    ----------
    states : torch.Tensor
        Batched states with ``n`` qubit axes.
    phases : torch.Tensor
        Real phases, shape ``(batch, 2**n)``.
    """
    batch_size = states.shape[0]
    flat = states.reshape(batch_size, -1)
    phase_factors = torch.polar(torch.ones_like(phases), phases).to(states.dtype)
    return (flat * phase_factors).reshape(states.shape)


def single_qubit_pauli_expectations(states: torch.Tensor) -> torch.Tensor:
    """Return ``<X_k>, <Y_k>, <Z_k>`` for every qubit k.

    These are the 1-RDM coordinates used by the projected quantum kernel (Eq. J.3).

    Returns
    -------
    torch.Tensor
        Real tensor of shape ``(batch, n_qubits, 3)``, ordered (X, Y, Z).
    """
    batch_size = states.shape[0]
    n_qubits = states.ndim - 1
    expectations = torch.empty(
        (batch_size, n_qubits, 3), dtype=states.real.dtype, device=states.device
    )
    for qubit in range(n_qubits):
        split = states.reshape(batch_size, 2**qubit, 2, -1)
        amp0 = split[:, :, 0, :]
        amp1 = split[:, :, 1, :]
        coherence = (amp0.conj() * amp1).sum(dim=(1, 2))
        expectations[:, qubit, 0] = 2 * coherence.real
        expectations[:, qubit, 1] = 2 * coherence.imag
        expectations[:, qubit, 2] = (amp0.abs() ** 2 - amp1.abs() ** 2).sum(dim=(1, 2))
    return expectations


def z_expectation(states: torch.Tensor, qubit: int) -> torch.Tensor:
    """Return ``<Z_qubit>`` for each state, shape ``(batch,)``."""
    batch_size = states.shape[0]
    split = states.reshape(batch_size, 2**qubit, 2, -1)
    return (split[:, :, 0, :].abs() ** 2 - split[:, :, 1, :].abs() ** 2).sum(dim=(1, 2))
