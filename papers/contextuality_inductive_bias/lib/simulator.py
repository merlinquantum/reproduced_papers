"""Batched three-qubit statevector simulation in PyTorch.

States have shape ``(batch, 2, 2, 2)``; axis ``1 + q`` is qubit ``q`` (qubit 0
is the most significant bit, as in PennyLane's wire ordering). Every gate
accepts either a scalar angle (shared by the batch) or a ``(batch,)`` tensor
(data-dependent encodings), and is differentiable with autograd.

Gate conventions follow the author code (PennyLane), which is the convention
consistent with the paper's spectrum Omega_L (discrepancy D5 in LOG.md):

- ``RZ(t)  = exp(-i t/2 Z)``, ``RY(t) = exp(-i t/2 Y)``
- ``ZZ(t)  = exp(-i t/2 Z (x) Z)``
- ``SWAPROT(t) = exp(-i t/2 (XX + YY + ZZ))`` (the author ``swap_rot``),
  equal to ``exp(-i t SWAP)`` up to a global phase.
- ``ROT(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)`` (PennyLane ``Rot``).
"""

from __future__ import annotations

import math

import torch

N_QUBITS = 3
_LETTERS = "bcd"


def zero_state(batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return ``|000>`` repeated ``batch`` times.

    Parameters
    ----------
    batch : int
        Number of independent states.
    dtype : torch.dtype
        Complex dtype of the state (``torch.complex128`` or ``torch.complex64``).
    device : torch.device
        Device holding the state.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch, 2, 2, 2)``.
    """
    state = torch.zeros((batch, 2, 2, 2), dtype=dtype, device=device)
    state[:, 0, 0, 0] = 1.0
    return state


def _batched_angle(angle: torch.Tensor | float, state: torch.Tensor) -> torch.Tensor:
    angle = torch.as_tensor(angle, dtype=state.real.dtype, device=state.device)
    if angle.ndim == 0:
        return angle.expand(state.shape[0])
    if angle.shape != (state.shape[0],):
        raise ValueError(
            f"Angle shape {tuple(angle.shape)} does not match batch {state.shape[0]}"
        )
    return angle


def apply_single(state: torch.Tensor, matrix: torch.Tensor, qubit: int) -> torch.Tensor:
    """Apply a single-qubit gate.

    Parameters
    ----------
    state : torch.Tensor
        State of shape ``(batch, 2, 2, 2)``.
    matrix : torch.Tensor
        Gate of shape ``(2, 2)`` or ``(batch, 2, 2)``.
    qubit : int
        Target qubit.

    Returns
    -------
    torch.Tensor
        Updated state.
    """
    letters = list(_LETTERS)
    target = letters[qubit]
    output = letters.copy()
    output[qubit] = "z"
    gate = "nz" + target if matrix.ndim == 3 else "z" + target
    equation = f"{gate},n{''.join(letters)}->n{''.join(output)}"
    return torch.einsum(equation, matrix, state)


def apply_two(
    state: torch.Tensor, matrix: torch.Tensor, qubits: tuple[int, int]
) -> torch.Tensor:
    """Apply a two-qubit gate given as a ``(4, 4)`` or ``(batch, 4, 4)`` matrix.

    The matrix basis is ``|q_a q_b>`` with ``q_a = qubits[0]`` as the most
    significant bit.
    """
    first, second = qubits
    if first == second:
        raise ValueError("Two-qubit gate needs two distinct qubits")
    letters = list(_LETTERS)
    output = letters.copy()
    output[first], output[second] = "y", "z"
    reshaped = matrix.reshape(*matrix.shape[:-2], 2, 2, 2, 2)
    gate = "nyz" if reshaped.ndim == 5 else "yz"
    gate += letters[first] + letters[second]
    equation = f"{gate},n{''.join(letters)}->n{''.join(output)}"
    return torch.einsum(equation, reshaped, state)


def _z_signs(qubit: int, device: torch.device) -> torch.Tensor:
    shape = [1, 1, 1, 1]
    shape[1 + qubit] = 2
    return torch.tensor([1.0, -1.0], device=device).reshape(shape)


def apply_rz(
    state: torch.Tensor, angle: torch.Tensor | float, qubit: int
) -> torch.Tensor:
    """Apply ``RZ(angle)`` (a diagonal phase) on ``qubit``."""
    theta = _batched_angle(angle, state).reshape(-1, 1, 1, 1)
    phase = torch.exp(-0.5j * theta * _z_signs(qubit, state.device))
    return state * phase


def apply_zz(
    state: torch.Tensor, angle: torch.Tensor | float, qubits: tuple[int, int]
) -> torch.Tensor:
    """Apply ``exp(-i angle/2 Z_a Z_b)`` (a diagonal phase)."""
    theta = _batched_angle(angle, state).reshape(-1, 1, 1, 1)
    parity = _z_signs(qubits[0], state.device) * _z_signs(qubits[1], state.device)
    return state * torch.exp(-0.5j * theta * parity)


def ry_matrix(angle: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``RY(angle)`` for a scalar or ``(batch,)`` angle."""
    cos, sin = torch.cos(angle / 2), torch.sin(angle / 2)
    matrix = torch.stack(
        [torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2
    )
    return matrix.to(dtype)


def apply_ry(
    state: torch.Tensor, angle: torch.Tensor | float, qubit: int
) -> torch.Tensor:
    """Apply ``RY(angle)`` on ``qubit``."""
    theta = torch.as_tensor(angle, dtype=state.real.dtype, device=state.device)
    return apply_single(state, ry_matrix(theta, state.dtype), qubit)


def apply_hadamard(state: torch.Tensor, qubit: int) -> torch.Tensor:
    """Apply a Hadamard gate on ``qubit``."""
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=state.dtype, device=state.device)
    return apply_single(state, h / math.sqrt(2.0), qubit)


def rot_matrix(
    phi: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """PennyLane ``Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)``."""
    cos, sin = torch.cos(theta / 2), torch.sin(theta / 2)
    plus, minus = (phi + omega) / 2, (phi - omega) / 2
    m00 = torch.exp(-1j * plus) * cos
    m01 = -torch.exp(1j * minus) * sin
    m10 = torch.exp(-1j * minus) * sin
    m11 = torch.exp(1j * plus) * cos
    return torch.stack(
        [torch.stack([m00, m01], -1), torch.stack([m10, m11], -1)], -2
    ).to(dtype)


def apply_rot(state: torch.Tensor, phi, theta, omega, qubit: int) -> torch.Tensor:
    """Apply PennyLane ``Rot(phi, theta, omega)`` on ``qubit``."""
    real = state.real.dtype
    args = [
        torch.as_tensor(a, dtype=real, device=state.device) for a in (phi, theta, omega)
    ]
    return apply_single(state, rot_matrix(*args, state.dtype), qubit)


def swaprot_matrix(angle: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``exp(-i angle/2 (XX + YY + ZZ))`` in the ``|00>, |01>, |10>, |11>`` basis.

    With ``XX + YY + ZZ = 2 SWAP - I``, the gate is
    ``e^{i angle/2} (cos(angle) I - i sin(angle) SWAP)``.
    """
    global_phase = torch.exp(0.5j * angle)
    cos, sin = torch.cos(angle), torch.sin(angle)
    diag_outer = torch.exp(-0.5j * angle)
    inner_diag = global_phase * cos
    inner_off = -1j * global_phase * sin
    zero = torch.zeros_like(inner_diag)
    rows = [
        torch.stack([diag_outer, zero, zero, zero], -1),
        torch.stack([zero, inner_diag, inner_off, zero], -1),
        torch.stack([zero, inner_off, inner_diag, zero], -1),
        torch.stack([zero, zero, zero, diag_outer], -1),
    ]
    return torch.stack(rows, -2).to(dtype)


def apply_swaprot(
    state: torch.Tensor, angle: torch.Tensor | float, qubits: tuple[int, int]
) -> torch.Tensor:
    """Apply the SWAP-generated rotation of the biased ansatz on ``qubits``."""
    theta = torch.as_tensor(angle, dtype=state.real.dtype, device=state.device)
    return apply_two(state, swaprot_matrix(theta, state.dtype), qubits)


def apply_cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    """Apply a CNOT gate (a permutation of amplitudes, no parameters)."""
    if control == target:
        raise ValueError("CNOT needs distinct control and target")
    cnot = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=state.dtype,
        device=state.device,
    )
    return apply_two(state, cnot, (control, target))


def expectation_z(state: torch.Tensor) -> torch.Tensor:
    """Return ``<Z_q>`` for q = 0, 1, 2 as a ``(batch, 3)`` real tensor."""
    probabilities = state.real**2 + state.imag**2
    marginals = []
    for qubit in range(N_QUBITS):
        signs = _z_signs(qubit, state.device).to(probabilities.dtype)
        marginals.append((probabilities * signs).sum(dim=(1, 2, 3)))
    return torch.stack(marginals, dim=-1)


def expectation_operator(state: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    """Return ``<psi| O |psi>`` for an ``(8, 8)`` Hermitian operator.

    Used in tests (bias operator, commutation checks); not on the training path.
    """
    flat = state.reshape(state.shape[0], 8)
    return torch.einsum("ni,ij,nj->n", flat.conj(), operator.to(flat.dtype), flat).real
