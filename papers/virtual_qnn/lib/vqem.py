"""Variational quantum error mitigation (paper Sec. III A, Algs. 1-2).

Register layout (from the authors' notebooks), 4 qubits::

    q0 = A (reference half of the Bell pair)
    q1 = B (system hit by the unknown noise N; output qubit of C_i)
    q2, q3 = ancillas initialised in |0>, traced out after C_i

Each branch ``C_i`` is a 3-qubit layered circuit on ``(q1, q2, q3)``. The
branch output is the 2-qubit operator ``sigma_i = Tr_{q2 q3}[C_i(N(Phi+) ⊗
|00><00|)]`` and the virtual state is ``sigma = c1 sigma_1 + c2 sigma_2``
(Eq. 17). Training drives ``sigma`` towards ``Phi+`` in squared HS distance.
"""

from __future__ import annotations

import torch
from torch import nn

from . import quantum
from .circuits import LayeredAnsatz, ladder_pairs

N_QUBITS = 4
BRANCH_REGISTER = (1, 2, 3)
OUTPUT_QUBITS = (0, 1)


def budget_coefficients(budget: float) -> tuple[float, float]:
    """Fixed coefficients for a sampling-overhead budget ``gamma >= 1``.

    ``c1 = (1 + gamma)/2`` and ``c2 = -(gamma - 1)/2`` so that ``c1 + c2 = 1``
    (trace preservation) and ``|c1| + |c2| = gamma``. This is the
    parameterization of the authors' Alg. 2 notebook and of Secs. III B-C.
    """
    if budget < 1.0:
        raise ValueError(f"Sampling-overhead budget must be >= 1, got {budget}")
    return (1.0 + budget) / 2.0, -(budget - 1.0) / 2.0


class VQEMModel(nn.Module):
    """Two-branch VQNN mitigating a single-qubit noise channel.

    Parameters
    ----------
    noise_model : str
        ``"depolarizing"`` or ``"amplitude_damping"``.
    noise_level : float
        Level ``p`` or ``q`` of the channel to mitigate.
    depth : int
        Layers per branch (paper: 10 noiseless, 5 noisy).
    coefficient_mode : str
        ``"trainable"`` (Alg. 1: ``c1 = t``, ``c2 = 1 - t`` with ``t``
        trained, inferred from the reference ``param_quasi`` call) or
        ``"budget"`` (Alg. 2: fixed coefficients from ``budget``).
    initial_t : float
        Initial ``t`` for ``"trainable"`` (reference: 2.0, i.e. overhead 3).
    budget : float
        Overhead budget for ``"budget"``.
    gate_noise : tuple[float, float]
        Single- and two-qubit depolarizing levels inside each branch.
    generator : torch.Generator
        Parameter-initialization RNG.
    complex_dtype : torch.dtype
        Simulation dtype.
    n_branches : int
        Number of branches ``N``. Default value is 2.
    """

    def __init__(
        self,
        noise_model: str,
        noise_level: float,
        depth: int,
        coefficient_mode: str,
        initial_t: float,
        budget: float,
        gate_noise: tuple[float, float],
        generator: torch.Generator,
        complex_dtype: torch.dtype,
        n_branches: int = 2,
    ) -> None:
        super().__init__()
        if n_branches != 2:
            raise ValueError("The paper and this reproduction use N = 2 branches")
        if coefficient_mode not in ("trainable", "budget"):
            raise ValueError(f"Unknown coefficient_mode {coefficient_mode!r}")
        self.noise_model = noise_model
        self.noise_level = float(noise_level)
        self.coefficient_mode = coefficient_mode
        self.complex_dtype = complex_dtype
        self.branches = nn.ModuleList(
            LayeredAnsatz(
                n_qubits=3,
                depth=depth,
                rotation_gates=("rx", "rz"),
                entangler="cnot",
                entangling_pairs=ladder_pairs(3),
                single_qubit_noise=gate_noise[0],
                two_qubit_noise=gate_noise[1],
                generator=generator,
                complex_dtype=complex_dtype,
            )
            for _ in range(n_branches)
        )
        real = torch.float64 if complex_dtype == torch.complex128 else torch.float32
        if coefficient_mode == "trainable":
            self.t = nn.Parameter(torch.tensor(float(initial_t), dtype=real))
        else:
            c1, c2 = budget_coefficients(float(budget))
            self.register_buffer(
                "fixed_coefficients", torch.tensor([c1, c2], dtype=real)
            )

    def coefficients(self) -> torch.Tensor:
        """Return ``(c1, c2)`` as a differentiable tensor."""
        if self.coefficient_mode == "trainable":
            return torch.stack([self.t, 1.0 - self.t])
        return self.fixed_coefficients

    def noisy_input(self, device: torch.device) -> torch.Tensor:
        """``N`` applied to ``B`` of ``Phi+_{AB} ⊗ |00><00|``."""
        bell = quantum.bell_state(self.complex_dtype, device)
        ancillas = quantum.zero_state(2, self.complex_dtype, device)
        rho = torch.kron(bell, ancillas)
        return quantum.apply_noise_channel(
            rho, self.noise_model, self.noise_level, target=1, n_qubits=N_QUBITS
        )

    def branch_states(self, device: torch.device) -> list[torch.Tensor]:
        """Branch outputs ``sigma_i`` on ``(A, B)``."""
        rho = self.noisy_input(device)
        return [
            quantum.partial_trace(
                branch(rho, register=BRANCH_REGISTER, total_qubits=N_QUBITS),
                OUTPUT_QUBITS,
                N_QUBITS,
            )
            for branch in self.branches
        ]

    def forward(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Return the virtual state, its squared HS error and the overhead."""
        coefficients = self.coefficients()
        states = self.branch_states(device)
        sigma = sum(c.to(self.complex_dtype) * s for c, s in zip(coefficients, states))
        target = quantum.bell_state(self.complex_dtype, device)
        return {
            "sigma": sigma,
            "hs_squared": quantum.squared_hs_distance(sigma, target),
            "overhead": coefficients.abs().sum(),
            "coefficients": coefficients,
        }

    def unmitigated_hs_squared(self, device: torch.device) -> float:
        """Squared HS error of ``(I ⊗ N)(Phi+)`` without any mitigation."""
        noisy_bell = quantum.partial_trace(
            self.noisy_input(device), OUTPUT_QUBITS, N_QUBITS
        )
        target = quantum.bell_state(self.complex_dtype, device)
        return float(quantum.squared_hs_distance(noisy_bell, target))


def vqem_loss(
    outputs: dict[str, torch.Tensor], algorithm: str, penalty: float
) -> torch.Tensor:
    """Loss of Alg. 1 (Eq. 19: ``||c||_1 + alpha * HS^2``) or Alg. 2 (Eq. 20: ``HS^2``).

    Raises
    ------
    ValueError
        If ``algorithm`` is not ``"penalty"`` or ``"budget"``.
    """
    if algorithm == "penalty":
        return outputs["overhead"] + penalty * outputs["hs_squared"]
    if algorithm == "budget":
        return outputs["hs_squared"]
    raise ValueError(f"Unknown VQEM algorithm {algorithm!r}")
