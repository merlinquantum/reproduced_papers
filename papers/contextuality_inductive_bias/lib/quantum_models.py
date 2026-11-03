"""Three-qubit quantum model classes of Sec. 8.2 and App. E.

Both models map the compressed features x = (x1, ..., x6) of Eq. (93) to the
three expectation values (<Z_1>, <Z_2>, <Z_3>), read as E[Y_k | x] for the
three players. Circuit layout (Fig. 7 / Fig. 10)::

    [V] U ( S1_x U S2_x U ) x L

so there are 2L + 1 trainable unitaries U, each made of B blocks of 9
parameters. S1_x applies RZ(x_k) on qubit k (k = 1, 2, 3) and S2_x applies
ZZ(x4 x5) on qubits (1, 2), ZZ(x5 x6) on (2, 3) and ZZ(x4 x6) on (1, 3), with
the gate conventions of ``lib.simulator``.

The biased model starts with V = H^3, RY(alpha) on qubit 1 and
RY(alpha - pi) on qubit 2, which prepares <Z1 + Z2 + Z3> = 0, and uses only
generators commuting with H = Z1 + Z2 + Z3 afterwards, so the zero-sum bias
holds for every x and every parameter value (Sec. 7.1).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from . import simulator as sim

PARAMS_PER_BLOCK = 9
PAIRS = ((0, 1), (1, 2), (0, 2))
# Entangler of the generic block. "paper": Fig. 10 (CNOT 1->2, 2->3, 1->3).
# "code": the author notebook applies CNOT(wires=[0, 1]) three times, which is
# a single CNOT 1->2 (discrepancy D3).
CNOT_PATTERNS = {
    "paper": ((0, 1), (1, 2), (0, 2)),
    "code": ((0, 1), (0, 1), (0, 1)),
}


def _uniform_angles(
    shape: tuple[int, ...], generator: torch.Generator, dtype: torch.dtype
) -> torch.Tensor:
    # App. F: "initialised by sampling each of their parameters uniformly in [0, 2 pi]".
    return torch.rand(shape, generator=generator, dtype=dtype) * (2.0 * math.pi)


def encode_single(state: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """S1_x: RZ(x1), RZ(x2), RZ(x3) on qubits 1, 2, 3."""
    for qubit in range(3):
        state = sim.apply_rz(state, features[:, qubit], qubit)
    return state


def encode_pairs(state: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """S2_x: ZZ(x4 x5) on (1, 2), ZZ(x5 x6) on (2, 3), ZZ(x4 x6) on (1, 3)."""
    x4, x5, x6 = features[:, 3], features[:, 4], features[:, 5]
    state = sim.apply_zz(state, x4 * x5, (0, 1))
    state = sim.apply_zz(state, x5 * x6, (1, 2))
    return sim.apply_zz(state, x4 * x6, (0, 2))


class _LayeredQuantumModel(nn.Module):
    """Shared data re-uploading structure of the two model classes."""

    def __init__(
        self, layers: int, blocks: int, generator: torch.Generator, dtype: torch.dtype
    ) -> None:
        super().__init__()
        if layers < 1 or blocks < 1:
            raise ValueError(
                f"layers and blocks must be >= 1, got L={layers}, B={blocks}"
            )
        self.layers = layers
        self.blocks = blocks
        self.real_dtype = dtype
        self.complex_dtype = (
            torch.complex128 if dtype == torch.float64 else torch.complex64
        )
        self.weights = nn.Parameter(
            _uniform_angles(
                (2 * layers + 1, blocks, PARAMS_PER_BLOCK), generator, dtype
            )
        )

    def prepare(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def unitary(self, state: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def final_state(self, features: torch.Tensor) -> torch.Tensor:
        """Return the output state for a ``(batch, 6)`` feature tensor."""
        if features.ndim != 2 or features.shape[1] != 6:
            raise ValueError(
                f"Expected features of shape (batch, 6), got {tuple(features.shape)}"
            )
        state = sim.zero_state(features.shape[0], self.complex_dtype, features.device)
        state = self.prepare(state)
        for layer in range(self.layers):
            state = self.unitary(state, self.weights[2 * layer])
            state = encode_single(state, features)
            state = self.unitary(state, self.weights[2 * layer + 1])
            state = encode_pairs(state, features)
        return self.unitary(state, self.weights[2 * self.layers])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Expectation values ``(<Z_1>, <Z_2>, <Z_3>)`` of shape ``(batch, 3)``."""
        return sim.expectation_z(self.final_state(features))


class BiasedQuantumModel(_LayeredQuantumModel):
    """Zero-sum-biased model of Fig. 7 (state-based approach, Sec. 7.1).

    Parameters
    ----------
    layers : int
        Number of data re-uploading layers L (paper: 2).
    blocks : int
        Blocks per trainable unitary B (paper: 2).
    generator : torch.Generator
        Source of the uniform [0, 2 pi] initialisation.
    dtype : torch.dtype
        Real parameter dtype; states use the matching complex dtype.
    zz_repetitions : int
        How often each ZZ rotation of a block is applied with its shared
        parameter. 1 follows Fig. 7; 3 reproduces the author code, where the
        ZZ gates sit inside the per-qubit loop (discrepancy D4, same model
        class). Default value is 1.
    """

    def __init__(
        self,
        layers: int,
        blocks: int,
        generator: torch.Generator,
        dtype: torch.dtype,
        zz_repetitions: int = 1,
    ) -> None:
        super().__init__(layers, blocks, generator, dtype)
        if zz_repetitions < 1:
            raise ValueError(f"zz_repetitions must be >= 1, got {zz_repetitions}")
        self.zz_repetitions = zz_repetitions
        self.alpha = nn.Parameter(_uniform_angles((), generator, dtype))

    def prepare(self, state: torch.Tensor) -> torch.Tensor:
        # V_theta of Fig. 7: on |+>, RY(a) gives <Z> = -sin(a) and RY(a - pi)
        # gives +sin(a), so <Z1> + <Z2> + <Z3> = 0 for every a.
        for qubit in range(3):
            state = sim.apply_hadamard(state, qubit)
        state = sim.apply_ry(state, self.alpha, 0)
        return sim.apply_ry(state, self.alpha - math.pi, 1)

    def unitary(self, state: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        for block in weights:
            for qubit in range(3):
                state = sim.apply_rz(state, block[qubit], qubit)
            for _ in range(self.zz_repetitions):
                state = sim.apply_zz(state, block[3], (0, 1))
                state = sim.apply_zz(state, block[4], (0, 2))
                state = sim.apply_zz(state, block[5], (1, 2))
            for index, pair in enumerate(PAIRS):
                state = sim.apply_swaprot(state, block[6 + index], pair)
        return state


class GenericQuantumModel(_LayeredQuantumModel):
    """Unbiased model of App. E (Fig. 10): general rotations plus CNOTs.

    Parameters
    ----------
    layers, blocks, generator, dtype
        As for :class:`BiasedQuantumModel`.
    cnot_pattern : str
        ``"paper"`` (Fig. 10) or ``"code"`` (author notebook, D3).
    """

    def __init__(
        self,
        layers: int,
        blocks: int,
        generator: torch.Generator,
        dtype: torch.dtype,
        cnot_pattern: str,
    ) -> None:
        super().__init__(layers, blocks, generator, dtype)
        if cnot_pattern not in CNOT_PATTERNS:
            raise ValueError(
                f"cnot_pattern must be one of {sorted(CNOT_PATTERNS)}, got {cnot_pattern!r}"
            )
        self.cnot_pattern = cnot_pattern

    def unitary(self, state: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        for block in weights:
            for qubit in range(3):
                state = sim.apply_rot(
                    state,
                    block[3 * qubit],
                    block[3 * qubit + 1],
                    block[3 * qubit + 2],
                    qubit,
                )
            for control, target in CNOT_PATTERNS[self.cnot_pattern]:
                state = sim.apply_cnot(state, control, target)
        return state
