"""Virtual variational quantum eigensolver (paper Sec. III C, Alg. 4, App. G).

Each branch is a noisy 6-qubit layered circuit (RX, RZ, CNOT ring) applied to
``|0...0>``; the branch state ``rho_i`` is the reduced state of qubits 0-1,
which host the 2-qubit tapered HeH+ Hamiltonian (authors' notebook). The
virtual state is ``sigma = c1 rho_1 + c2 rho_2`` with budget coefficients.

Loss (authors' notebook form of Eq. 28)::

    L = Tr[H sigma] + alpha * | Tr[p_K(sigma / gamma)] |

where ``p_K`` is the K-truncated Chebyshev series of the negativity function
``f(x) = (|x| - x)/2`` on ``[-1, 1]``. The paper writes the penalty as
``alpha c1 Tr[p_K(sigma/|c1|)]``; the reference code, which produced Fig. 7,
normalizes by ``gamma`` and takes an absolute value (LOG.md).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from . import quantum
from .circuits import LayeredAnsatz, ring_pairs
from .vqem import budget_coefficients

HAMILTONIAN_QUBITS = (0, 1)


def negativity_chebyshev_coefficients(order: int) -> list[float]:
    """Chebyshev coefficients ``a_0..a_K`` of ``f(x) = (|x| - x)/2`` (Eq. G4).

    ``a_0 = 1/pi``, ``a_1 = -1/2``, ``a_{2k} = 2 (-1)^{k+1} / (pi (4k^2 - 1))``
    and zero for odd ``n > 1``. The reference notebook obtains the same
    coefficients by trapezoidal quadrature; the analytic values are used here.
    """
    coefficients = [0.0] * (order + 1)
    coefficients[0] = 1.0 / math.pi
    if order >= 1:
        coefficients[1] = -0.5
    for n in range(2, order + 1, 2):
        k = n // 2
        coefficients[n] = 2.0 * (-1) ** (k + 1) / (math.pi * (4 * k * k - 1))
    return coefficients


def chebyshev_matrix_series(
    matrix: torch.Tensor, coefficients: list[float]
) -> torch.Tensor:
    """Evaluate ``sum_n a_n T_n(matrix)`` by the three-term recursion (Eq. G3)."""
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    previous, current = identity, matrix
    result = coefficients[0] * previous
    if len(coefficients) > 1:
        result = result + coefficients[1] * current
    for coefficient in coefficients[2:]:
        previous, current = current, 2.0 * matrix @ current - previous
        result = result + coefficient * current
    return result


class VirtualVQE(nn.Module):
    """Two-branch VVQE; ``budget = 1`` is the conventional (noisy) VQE.

    Parameters
    ----------
    n_qubits : int
        Circuit width (reference: 6).
    depth : int
        Layers per branch (paper: 5).
    budget : float
        Sampling-overhead budget ``gamma``.
    gate_noise : tuple[float, float]
        Single- and two-qubit depolarizing levels (paper: 0.01, 0.02).
    generator : torch.Generator
        Parameter-initialization RNG.
    complex_dtype : torch.dtype
        Simulation dtype.
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int,
        budget: float,
        gate_noise: tuple[float, float],
        generator: torch.Generator,
        complex_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.complex_dtype = complex_dtype
        self.budget = float(budget)
        self.c1, self.c2 = budget_coefficients(self.budget)
        n_branches = 1 if self.c2 == 0.0 else 2
        self.branches = nn.ModuleList(
            LayeredAnsatz(
                n_qubits=n_qubits,
                depth=depth,
                rotation_gates=("rx", "rz"),
                entangler="cnot",
                entangling_pairs=ring_pairs(n_qubits),
                single_qubit_noise=gate_noise[0],
                two_qubit_noise=gate_noise[1],
                generator=generator,
                complex_dtype=complex_dtype,
            )
            for _ in range(n_branches)
        )

    def virtual_state(self, device: torch.device) -> torch.Tensor:
        """Return ``sigma = sum_i c_i rho_i`` on the Hamiltonian qubits."""
        initial = quantum.zero_state(self.n_qubits, self.complex_dtype, device)
        coefficients = (self.c1, self.c2)[: len(self.branches)]
        sigma = 0.0
        for coefficient, branch in zip(coefficients, self.branches):
            reduced = quantum.partial_trace(
                branch(initial), HAMILTONIAN_QUBITS, self.n_qubits
            )
            sigma = sigma + coefficient * reduced
        return sigma

    def n_parameters(self) -> int:
        """Trainable parameters of the branches that carry nonzero weight."""
        return sum(branch.n_parameters() for branch in self.branches)


def vvqe_terms(
    sigma: torch.Tensor,
    hamiltonian: torch.Tensor,
    budget: float,
    penalty: float,
    coefficients: list[float],
) -> dict[str, torch.Tensor]:
    """Energy, Chebyshev negativity penalty and loss for a virtual state."""
    energy = torch.real(torch.einsum("ij,ji->", hamiltonian, sigma))
    hermitian = 0.5 * (sigma + sigma.conj().transpose(-1, -2))
    negativity_proxy = torch.real(
        torch.trace(chebyshev_matrix_series(hermitian / budget, coefficients))
    )
    loss = energy + penalty * negativity_proxy.abs()
    return {"energy": energy, "negativity_proxy": negativity_proxy, "loss": loss}


def exact_negativity(sigma: torch.Tensor) -> tuple[float, float]:
    """Return ``(sum of |negative eigenvalues|, minimum eigenvalue)`` of ``sigma``."""
    hermitian = 0.5 * (sigma + sigma.conj().transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(hermitian.detach())
    negativity = float(eigenvalues.clamp(max=0.0).abs().sum())
    return negativity, float(eigenvalues.min())
