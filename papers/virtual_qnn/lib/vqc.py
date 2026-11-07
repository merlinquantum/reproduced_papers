"""Virtual quantum classifier (paper Sec. III B, Alg. 3).

Implementation follows the authors' notebook, which differs from the paper
text in two documented ways (LOG.md, "Reference-code findings"):

- Encoding: two qubits, qubit ``i`` prepared as ``RZ(arccos(x_j^2)) RY(arcsin(x_j))
  |0>`` with ``j = i mod 2`` (noiseless), instead of paper Eq. (E1).
- Ansatz: (RX, RY) rotations and a CZ entangler, with 1-qubit depolarizing
  ``p = 0.02`` after each rotation and 2-qubit depolarizing ``p = 0.05`` after
  the CZ.

Prediction (Alg. 3 step 8): ``y = (1 + c1 <Z_0>_1 + c2 <Z_0>_2) / 2`` with
``c1 = (1 + gamma)/2`` and ``c2 = -(gamma - 1)/2``.
"""

from __future__ import annotations

import torch
from torch import nn

from . import quantum
from .circuits import LayeredAnsatz, ladder_pairs
from .vqem import budget_coefficients


def encode_points(
    x: torch.Tensor, n_qubits: int, complex_dtype: torch.dtype
) -> torch.Tensor:
    """Encode 2D points into pure-state density matrices.

    Parameters
    ----------
    x : torch.Tensor
        Points of shape ``(batch, 2)`` with entries in ``[-1, 1]``.
    n_qubits : int
        Number of encoding qubits (reference: 2).
    complex_dtype : torch.dtype
        Simulation dtype.

    Returns
    -------
    torch.Tensor
        Density matrices of shape ``(batch, 2**n, 2**n)``.
    """
    states = None
    for qubit in range(n_qubits):
        feature = x[:, qubit % 2]
        ry_angle = torch.arcsin(feature)
        rz_angle = torch.arccos(feature**2)
        amplitude0 = torch.cos(ry_angle / 2).to(complex_dtype) * torch.exp(
            -0.5j * rz_angle.to(complex_dtype)
        )
        amplitude1 = torch.sin(ry_angle / 2).to(complex_dtype) * torch.exp(
            0.5j * rz_angle.to(complex_dtype)
        )
        qubit_state = torch.stack([amplitude0, amplitude1], dim=-1)
        states = (
            qubit_state
            if states is None
            else torch.einsum("bi,bj->bij", states, qubit_state).reshape(len(x), -1)
        )
    return torch.einsum("bi,bj->bij", states, states.conj())


class VirtualQuantumClassifier(nn.Module):
    """Two-branch noisy classifier; ``budget = 1`` is the conventional QC.

    Parameters
    ----------
    n_qubits : int
        Circuit width (reference: 2).
    depth : int
        Layers per branch (reference: 2).
    budget : float
        Sampling-overhead budget ``gamma``.
    gate_noise : tuple[float, float]
        Single- and two-qubit depolarizing levels (reference: 0.02, 0.05).
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
        # At budget 1 the second branch has zero weight; it is not built, so
        # the parameter count reflects the conventional classifier.
        n_branches = 1 if self.c2 == 0.0 else 2
        self.branches = nn.ModuleList(
            LayeredAnsatz(
                n_qubits=n_qubits,
                depth=depth,
                rotation_gates=("rx", "ry"),
                entangler="cz",
                entangling_pairs=ladder_pairs(n_qubits),
                single_qubit_noise=gate_noise[0],
                two_qubit_noise=gate_noise[1],
                generator=generator,
                complex_dtype=complex_dtype,
            )
            for _ in range(n_branches)
        )
        z_first = torch.diag(torch.tensor([1.0, -1.0], dtype=complex_dtype))
        observable = quantum.kron_all(
            [z_first] + [torch.eye(2, dtype=complex_dtype)] * (n_qubits - 1)
        )
        self.register_buffer("observable", observable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions ``y`` of shape ``(batch,)``."""
        rho = encode_points(x, self.n_qubits, self.complex_dtype)
        coefficients = (self.c1, self.c2)[: len(self.branches)]
        combination = 0.0
        for coefficient, branch in zip(coefficients, self.branches):
            output = branch(rho)
            expectation = torch.real(torch.einsum("bij,ji->b", output, self.observable))
            combination = combination + coefficient * expectation
        return 0.5 * (1.0 + combination)

    def n_parameters(self) -> int:
        """Trainable parameters of the branches that carry nonzero weight."""
        return sum(branch.n_parameters() for branch in self.branches)


def reference_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Accuracy rule of the reference notebook: correct if ``||y| - label| <= 0.5``."""
    correct = (predictions.abs() - labels).abs() <= 0.5
    return float(correct.to(torch.float64).mean())
