"""Parametrized (optionally noisy) layered circuits used as VQNN branches.

Every task uses the paper's "strong entangled layer" family (Fig. 3): each layer
applies one rotation per qubit for every gate in ``rotation_gates``, then a fixed
sequence of two-qubit entanglers. In the noisy variant (Fig. 3b) a single-qubit
depolarizing channel follows every rotation and a two-qubit depolarizing
channel follows every entangler. Per-task gate choices follow the authors'
reference notebooks (see LOG.md, "Reference-code findings"):

- VQEM: rotations (RX, RZ), CNOT ladder (0,1), (1,2) on the 3-qubit register.
- VQC: rotations (RX, RY), CZ on (0,1).
- VVQE: rotations (RX, RZ), CNOT ring (0,1), ..., (4,5), (5,0) on 6 qubits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from . import quantum

_ROTATIONS = {"rx": quantum.rx, "ry": quantum.ry, "rz": quantum.rz}
_ENTANGLERS = {"cnot": quantum.cnot, "cz": quantum.cz}


class LayeredAnsatz(nn.Module):
    """Layered rotation + entangler circuit acting on a density matrix.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit register.
    depth : int
        Number of layers ``d``.
    rotation_gates : Sequence[str]
        Rotation names applied to every qubit in each layer, e.g. ``("rx", "rz")``.
    entangler : str
        ``"cnot"`` or ``"cz"``.
    entangling_pairs : Sequence[tuple[int, int]]
        Ordered (control, target) pairs applied after the rotations of each layer.
    single_qubit_noise : float
        Depolarizing level after each rotation. Default value is 0.0.
    two_qubit_noise : float
        Two-qubit depolarizing level after each entangler. Default value is 0.0.
    generator : torch.Generator | None
        Source of the uniform ``[0, 2*pi)`` parameter initialization. Default
        value is None (global RNG).
    complex_dtype : torch.dtype
        Simulation dtype. Default value is ``torch.complex128``.
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int,
        rotation_gates: Sequence[str],
        entangler: str,
        entangling_pairs: Sequence[tuple[int, int]],
        single_qubit_noise: float = 0.0,
        two_qubit_noise: float = 0.0,
        generator: torch.Generator | None = None,
        complex_dtype: torch.dtype = torch.complex128,
    ) -> None:
        super().__init__()
        for gate in rotation_gates:
            if gate not in _ROTATIONS:
                raise ValueError(f"Unknown rotation gate {gate!r}")
        if entangler not in _ENTANGLERS:
            raise ValueError(f"Unknown entangler {entangler!r}")
        self.n_qubits = n_qubits
        self.depth = depth
        self.rotation_gates = tuple(rotation_gates)
        self.entangler = entangler
        self.entangling_pairs = [tuple(pair) for pair in entangling_pairs]
        self.single_qubit_noise = float(single_qubit_noise)
        self.two_qubit_noise = float(two_qubit_noise)
        self.complex_dtype = complex_dtype
        real_dtype = (
            torch.float64 if complex_dtype == torch.complex128 else torch.float32
        )
        shape = (depth, len(self.rotation_gates), n_qubits)
        initial = torch.rand(shape, generator=generator, dtype=real_dtype) * 2 * math.pi
        self.theta = nn.Parameter(initial)

    def forward(
        self,
        rho: torch.Tensor,
        register: Sequence[int] | None = None,
        total_qubits: int | None = None,
    ) -> torch.Tensor:
        """Apply the circuit to ``rho``.

        Parameters
        ----------
        rho : torch.Tensor
            Density matrix of shape ``(..., 2**N, 2**N)``.
        register : Sequence[int] | None
            Global qubit indices hosting the circuit's local qubits
            ``0..n_qubits-1``. Default value is None (``range(n_qubits)``).
        total_qubits : int | None
            Total qubit count ``N`` of ``rho``. Default value is None
            (``n_qubits``).

        Returns
        -------
        torch.Tensor
            Output density matrix, same shape as ``rho``.
        """
        register = list(range(self.n_qubits)) if register is None else list(register)
        total = self.n_qubits if total_qubits is None else total_qubits
        device = rho.device
        entangler = _ENTANGLERS[self.entangler](self.complex_dtype, device)
        for layer in range(self.depth):
            for gate_index, gate in enumerate(self.rotation_gates):
                for local in range(self.n_qubits):
                    angle = self.theta[layer, gate_index, local]
                    unitary = _ROTATIONS[gate](angle, self.complex_dtype)
                    rho = quantum.apply_unitary(rho, unitary, [register[local]], total)
                    rho = quantum.depolarize(
                        rho, [register[local]], total, self.single_qubit_noise
                    )
            for control, target in self.entangling_pairs:
                pair = [register[control], register[target]]
                rho = quantum.apply_unitary(rho, entangler, pair, total)
                rho = quantum.depolarize(rho, pair, total, self.two_qubit_noise)
        return rho

    def n_parameters(self) -> int:
        """Number of trainable rotation angles."""
        return self.theta.numel()


def ladder_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """Nearest-neighbour CNOT ladder ``(0,1), (1,2), ..., (n-2, n-1)``."""
    return [(q, q + 1) for q in range(n_qubits - 1)]


def ring_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """Ladder closed by ``(n-1, 0)``, as in the VVQE reference circuit."""
    return ladder_pairs(n_qubits) + [(n_qubits - 1, 0)]
