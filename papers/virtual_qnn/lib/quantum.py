"""Minimal differentiable density-matrix simulator used by every experiment.

Conventions
-----------
- Qubit 0 is the most significant (leftmost) tensor factor, as in QuAIRKit and
  PennyLane, so ``|q0 q1 ... q_{n-1}>`` maps to the basis index
  ``q0 * 2**(n-1) + ... + q_{n-1}``.
- Density matrices have shape ``(..., 2**n, 2**n)``; leading dimensions are
  batch dimensions (used for the classification data set).
- Rotations follow ``R_P(theta) = exp(-i theta P / 2)``.
- Depolarizing noise on ``k`` qubits is ``D_p(rho) = (1 - p) rho + p I/2**k
  ⊗ Tr_k(rho)``, i.e. paper Eq. (21). This is the convention implied by the
  paper's numbers: the unmitigated squared HS error of a D_0.2 channel on one
  half of a Bell pair is ``0.75 p**2 = 0.03`` (Fig. 5, budget 1).
"""

from __future__ import annotations

import string
from collections.abc import Sequence

import torch

_LETTERS = string.ascii_letters


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """Return the complex dtype matching a real floating dtype.

    Parameters
    ----------
    real_dtype : torch.dtype
        ``torch.float32`` or ``torch.float64``.

    Returns
    -------
    torch.dtype
        ``torch.complex64`` or ``torch.complex128``.

    Raises
    ------
    ValueError
        If ``real_dtype`` is not float32 or float64.
    """
    mapping = {torch.float32: torch.complex64, torch.float64: torch.complex128}
    if real_dtype not in mapping:
        raise ValueError(f"Unsupported real dtype for simulation: {real_dtype}")
    return mapping[real_dtype]


def _apply_one_side(
    tensor: torch.Tensor, operator: torch.Tensor, axes: list[int]
) -> torch.Tensor:
    """Contract ``operator`` (shape ``[2]*2k``) into ``tensor`` along ``axes``."""
    k = len(axes)
    contracted = torch.tensordot(tensor, operator, dims=(axes, list(range(k, 2 * k))))
    return torch.movedim(
        contracted, list(range(contracted.ndim - k, contracted.ndim)), axes
    )


def apply_two_sided(
    rho: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    targets: Sequence[int],
    n_qubits: int,
) -> torch.Tensor:
    """Return ``left_targets @ rho @ right_targets^dagger`` embedded on ``targets``.

    Parameters
    ----------
    rho : torch.Tensor
        Density matrix of shape ``(..., 2**n, 2**n)``.
    left, right : torch.Tensor
        ``(2**k, 2**k)`` operators acting on the ordered ``targets``.
    targets : Sequence[int]
        Qubit indices the operators act on, in operator order.
    n_qubits : int
        Total number of qubits ``n``.

    Returns
    -------
    torch.Tensor
        Transformed operator with the same shape as ``rho``.
    """
    k = len(targets)
    batch_shape = rho.shape[:-2]
    n_batch = len(batch_shape)
    dim = 2**n_qubits
    tensor = rho.reshape(*batch_shape, *([2] * (2 * n_qubits)))
    row_axes = [n_batch + t for t in targets]
    col_axes = [n_batch + n_qubits + t for t in targets]
    tensor = _apply_one_side(tensor, left.reshape([2] * (2 * k)), row_axes)
    tensor = _apply_one_side(tensor, right.conj().reshape([2] * (2 * k)), col_axes)
    return tensor.reshape(*batch_shape, dim, dim)


def apply_unitary(
    rho: torch.Tensor, unitary: torch.Tensor, targets: Sequence[int], n_qubits: int
) -> torch.Tensor:
    """Apply ``U rho U^dagger`` with ``U`` acting on ``targets``."""
    return apply_two_sided(rho, unitary, unitary, targets, n_qubits)


def apply_kraus(
    rho: torch.Tensor,
    kraus_ops: Sequence[torch.Tensor],
    targets: Sequence[int],
    n_qubits: int,
) -> torch.Tensor:
    """Apply the channel ``sum_j K_j rho K_j^dagger`` on ``targets``."""
    return sum(apply_two_sided(rho, op, op, targets, n_qubits) for op in kraus_ops)


def _einsum_partial_trace(n_qubits: int, keep: Sequence[int]) -> str:
    rows = list(_LETTERS[:n_qubits])
    cols = list(_LETTERS[n_qubits : 2 * n_qubits])
    for qubit in range(n_qubits):
        if qubit not in keep:
            cols[qubit] = rows[qubit]
    out = "".join(rows[q] for q in keep) + "".join(cols[q] for q in keep)
    return "..." + "".join(rows) + "".join(cols) + "->..." + out


def partial_trace(
    rho: torch.Tensor, keep: Sequence[int], n_qubits: int
) -> torch.Tensor:
    """Trace out every qubit not listed in ``keep`` (kept in the given order).

    Parameters
    ----------
    rho : torch.Tensor
        Operator of shape ``(..., 2**n, 2**n)``.
    keep : Sequence[int]
        Qubits that remain, in the order they appear in the output.
    n_qubits : int
        Total number of qubits ``n``.

    Returns
    -------
    torch.Tensor
        Reduced operator of shape ``(..., 2**len(keep), 2**len(keep))``.
    """
    keep = list(keep)
    batch_shape = rho.shape[:-2]
    tensor = rho.reshape(*batch_shape, *([2] * (2 * n_qubits)))
    reduced = torch.einsum(_einsum_partial_trace(n_qubits, keep), tensor)
    dim = 2 ** len(keep)
    return reduced.reshape(*batch_shape, dim, dim)


def depolarize(
    rho: torch.Tensor, targets: Sequence[int], n_qubits: int, probability: float
) -> torch.Tensor:
    """Apply ``D_p`` (paper Eq. 21) jointly on ``targets``.

    ``D_p(rho) = (1 - p) rho + p * Tr_targets(rho) ⊗ I/2**k``, with the
    identity re-inserted on the target qubits. For two targets this is the
    two-qubit ("generalized") depolarizing channel of the noisy ansatz.
    """
    if probability == 0.0:
        return rho
    targets = list(targets)
    others = [q for q in range(n_qubits) if q not in targets]
    batch_shape = rho.shape[:-2]
    reduced = partial_trace(rho, others, n_qubits)
    reduced = reduced.reshape(*batch_shape, *([2] * (2 * len(others))))
    eye = torch.eye(2, dtype=rho.dtype, device=rho.device) / 2.0

    rows = list(_LETTERS[:n_qubits])
    cols = list(_LETTERS[n_qubits : 2 * n_qubits])
    operands = [reduced]
    subscripts = [
        "..." + "".join(rows[q] for q in others) + "".join(cols[q] for q in others)
    ]
    for qubit in targets:
        operands.append(eye)
        subscripts.append(rows[qubit] + cols[qubit])
    expression = ",".join(subscripts) + "->..." + "".join(rows) + "".join(cols)
    mixed = torch.einsum(expression, *operands).reshape(rho.shape)
    return (1.0 - probability) * rho + probability * mixed


def amplitude_damping_kraus(
    gamma: float, dtype: torch.dtype, device: torch.device | str = "cpu"
) -> list[torch.Tensor]:
    """Kraus operators of the amplitude-damping channel A_q (paper Eq. 22)."""
    k0 = torch.tensor([[1.0, 0.0], [0.0, (1.0 - gamma) ** 0.5]], dtype=dtype)
    k1 = torch.tensor([[0.0, gamma**0.5], [0.0, 0.0]], dtype=dtype)
    return [k0.to(device), k1.to(device)]


def apply_noise_channel(
    rho: torch.Tensor,
    model: str,
    level: float,
    target: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply the single-qubit noise channel ``N`` to be mitigated.

    Parameters
    ----------
    rho : torch.Tensor
        Density matrix.
    model : str
        ``"depolarizing"`` (D_p) or ``"amplitude_damping"`` (A_q).
    level : float
        Noise level ``p`` or ``q``.
    target : int
        Qubit the noise acts on.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    torch.Tensor
        Noisy density matrix.

    Raises
    ------
    ValueError
        If ``model`` is unknown.
    """
    if model == "depolarizing":
        return depolarize(rho, [target], n_qubits, level)
    if model == "amplitude_damping":
        kraus = amplitude_damping_kraus(level, rho.dtype, rho.device)
        return apply_kraus(rho, kraus, [target], n_qubits)
    raise ValueError(f"Unknown noise model {model!r}")


def rx(theta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Single-qubit ``RX(theta)`` matrix."""
    c = torch.cos(theta / 2).to(dtype)
    s = torch.sin(theta / 2).to(dtype)
    return torch.stack([torch.stack([c, -1j * s]), torch.stack([-1j * s, c])])


def ry(theta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Single-qubit ``RY(theta)`` matrix."""
    c = torch.cos(theta / 2).to(dtype)
    s = torch.sin(theta / 2).to(dtype)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def rz(theta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Single-qubit ``RZ(theta)`` matrix."""
    phase = torch.exp(-0.5j * theta.to(dtype))
    zero = torch.zeros((), dtype=dtype, device=theta.device)
    return torch.stack([torch.stack([phase, zero]), torch.stack([zero, phase.conj()])])


def cnot(dtype: torch.dtype, device: torch.device | str = "cpu") -> torch.Tensor:
    """CNOT with the first target as control."""
    matrix = torch.eye(4, dtype=dtype, device=device)
    return matrix[[0, 1, 3, 2]]


def cz(dtype: torch.dtype, device: torch.device | str = "cpu") -> torch.Tensor:
    """Controlled-Z."""
    return torch.diag(torch.tensor([1, 1, 1, -1], dtype=dtype, device=device))


def bell_state(dtype: torch.dtype, device: torch.device | str = "cpu") -> torch.Tensor:
    """Projector onto ``|Phi+> = (|00> + |11>)/sqrt(2)``."""
    vector = torch.zeros(4, dtype=dtype, device=device)
    vector[0] = vector[3] = 2**-0.5
    return torch.outer(vector, vector.conj())


def zero_state(
    n_qubits: int, dtype: torch.dtype, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Density matrix ``|0...0><0...0|``."""
    dim = 2**n_qubits
    rho = torch.zeros(dim, dim, dtype=dtype, device=device)
    rho[0, 0] = 1.0
    return rho


def kron_all(operators: Sequence[torch.Tensor]) -> torch.Tensor:
    """Kronecker product of operators in qubit order (qubit 0 leftmost)."""
    result = operators[0]
    for operator in operators[1:]:
        result = torch.kron(result, operator)
    return result


def squared_hs_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared Hilbert-Schmidt distance ``Tr[(a - b)^2]`` (paper Eq. 18).

    The paper calls this quantity the "HS norm" in Table I and Figs. 4-5.
    """
    diff = a - b
    return torch.real(torch.einsum("...ij,...ji->...", diff, diff))
