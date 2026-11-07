"""Gate-based quantum reservoir for level generation.

The reservoir is the composition of three subcircuits applied in order:
  1. an embedding circuit for the input feature ``x_t`` (Ry rotations
     interlaced with CNOT gates - Fig. 1b right of the paper),
  2. an embedding circuit for the hidden state ``h_t`` (same shape but with
     angles derived from the 2**q probability vector),
  3. a fixed random circuit composed of {X, H, CNOT} gates sampled at
     construction time (Fig. 1b left).

Only the downstream feed-forward neural network (lib.fnn) is trainable; the
reservoir parameters are fixed throughout the run, in keeping with reservoir
computing.

State-vector simulation is implemented with numpy. The number of qubits in
the paper (q in {4, 5, 6, 7, 8}) makes 2**q probability vectors trivially
tractable on CPU. Depolarising noise is supported via a density-matrix path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Single-qubit gate primitives operating on a state vector
# ---------------------------------------------------------------------------


def _apply_single(
    state: np.ndarray, gate: np.ndarray, qubit: int, n: int
) -> np.ndarray:
    """Apply a 2x2 gate to one qubit of an n-qubit state vector."""
    state = state.reshape((2,) * n)
    state = np.moveaxis(state, qubit, 0)
    flat = state.reshape(2, -1)
    flat = gate @ flat
    state = flat.reshape((2,) + state.shape[1:])
    state = np.moveaxis(state, 0, qubit)
    return state.reshape(2**n)


def _apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    state = state.reshape((2,) * n)
    state = np.moveaxis(state, [control, target], [0, 1])
    flat = state.reshape(2, 2, -1)
    out = np.empty_like(flat)
    out[0, 0] = flat[0, 0]
    out[0, 1] = flat[0, 1]
    out[1, 0] = flat[1, 1]
    out[1, 1] = flat[1, 0]
    state = out.reshape((2, 2) + state.shape[2:])
    state = np.moveaxis(state, [0, 1], [control, target])
    return state.reshape(2**n)


_PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_HADAMARD = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def _ry(angle: float) -> np.ndarray:
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# Reservoir definition
# ---------------------------------------------------------------------------


@dataclass
class RandomGateOp:
    name: str  # "X", "H", or "CNOT"
    qubits: tuple[int, ...]


def sample_random_reservoir(
    n_qubits: int, n_gates: int, rng: np.random.Generator
) -> list[RandomGateOp]:
    """Sample a fixed random circuit drawn from {X, H, CNOT}."""
    ops: list[RandomGateOp] = []
    for _ in range(n_gates):
        choice = rng.integers(0, 3)
        if choice == 0:
            ops.append(RandomGateOp("X", (int(rng.integers(0, n_qubits)),)))
        elif choice == 1:
            ops.append(RandomGateOp("H", (int(rng.integers(0, n_qubits)),)))
        else:
            ctrl, tgt = rng.choice(n_qubits, size=2, replace=False)
            ops.append(RandomGateOp("CNOT", (int(ctrl), int(tgt))))
    return ops


def _apply_random(state: np.ndarray, ops: Sequence[RandomGateOp], n: int) -> np.ndarray:
    for op in ops:
        if op.name == "X":
            state = _apply_single(state, _PAULI_X, op.qubits[0], n)
        elif op.name == "H":
            state = _apply_single(state, _HADAMARD, op.qubits[0], n)
        elif op.name == "CNOT":
            state = _apply_cnot(state, op.qubits[0], op.qubits[1], n)
        else:
            raise ValueError(f"Unknown gate {op.name}")
    return state


def _apply_embedding(state: np.ndarray, angles: np.ndarray, n: int) -> np.ndarray:
    """One layer of Ry rotations followed by a ring of CNOTs (Fig. 1b right).

    ``angles`` must have shape ``(n,)``. The CNOT ring couples qubit i with
    qubit (i+1) % n.
    """
    for q in range(n):
        state = _apply_single(state, _ry(angles[q]), q, n)
    for q in range(n):
        state = _apply_cnot(state, q, (q + 1) % n, n)
    return state


def _state_to_density(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def _depolarize_density(rho: np.ndarray, p: float) -> np.ndarray:
    """Global depolarising channel with overall depolarising strength ``p``.

    A standalone application of a single global depolarising channel is used
    instead of per-gate decomposition; in the paper the reported metric varies
    smoothly with ``p`` and a global channel is sufficient for the
    qualitative reproduction (see README for the deviation note).
    """
    if p <= 0:
        return rho
    dim = rho.shape[0]
    identity = np.eye(dim, dtype=np.complex128) / dim
    return (1 - p) * rho + p * identity


# ---------------------------------------------------------------------------
# Public QRC class
# ---------------------------------------------------------------------------


class QubitQRC:
    """Black-box quantum reservoir over ``n_qubits`` qubits."""

    def __init__(
        self,
        n_qubits: int,
        num_features: int,
        n_random_gates: int = 30,
        embedding_layers: int = 1,
        input_scale: float = 1.0,
        feedback_scale: float = 1.0,
        depolarizing_p: float = 0.0,
        shots: int = 0,
        seed: int = 0,
    ):
        self.n_qubits = int(n_qubits)
        self.num_features = int(num_features)
        self.embedding_layers = int(embedding_layers)
        self.input_scale = float(input_scale)
        self.feedback_scale = float(feedback_scale)
        self.depolarizing_p = float(depolarizing_p)
        self.shots = int(shots)
        self.dim = 2**self.n_qubits

        rng = np.random.default_rng(seed)
        self.random_ops = sample_random_reservoir(
            self.n_qubits, int(n_random_gates), rng
        )

        # Per-feature angle book: row ``x`` of ``input_angles_book`` is the
        # vector of ``n_qubits`` angles used to encode feature ``x``. Drawing
        # angles from U(-pi, pi) makes distinct features map to clearly
        # separated initial states (large-angle rotations are non-linear, as
        # the paper notes) while keeping the table fixed throughout - i.e.
        # part of the reservoir's untrained random structure.
        self.input_angles_book = rng.uniform(
            -np.pi, np.pi, size=(self.num_features, self.n_qubits)
        )
        self.hidden_projection = rng.normal(0.0, 1.0, size=(self.dim, self.n_qubits))

    # -- helpers -----------------------------------------------------------------

    def _input_angles(self, x: int) -> np.ndarray:
        return self.input_scale * self.input_angles_book[int(x)]

    def _hidden_angles(self, h: np.ndarray) -> np.ndarray:
        # The hidden state is a probability vector summing to 1, so a Gaussian
        # projection gives angles whose magnitudes scale roughly as
        # sqrt(2 / dim). Multiply by ``feedback_scale * sqrt(dim)`` to keep the
        # feedback amplitude comparable to the input encoding.
        return self.feedback_scale * np.sqrt(self.dim) * (h @ self.hidden_projection)

    # -- public API --------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        return self.dim

    def initial_hidden(self) -> np.ndarray:
        return np.full(self.dim, 1.0 / self.dim, dtype=np.float64)

    def step(self, x_t: int, h_t: np.ndarray) -> np.ndarray:
        """Return the next probability vector ``p_t``.

        ``x_t`` is an integer feature index in ``[0, num_features)``; ``h_t``
        is a probability vector of size ``2**n_qubits``.
        """
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0

        for _ in range(self.embedding_layers):
            state = _apply_embedding(state, self._input_angles(x_t), self.n_qubits)
            state = _apply_embedding(state, self._hidden_angles(h_t), self.n_qubits)

        state = _apply_random(state, self.random_ops, self.n_qubits)

        if self.depolarizing_p > 0.0:
            rho = _state_to_density(state)
            rho = _depolarize_density(rho, self.depolarizing_p)
            probs = np.real(np.diag(rho))
        else:
            probs = np.abs(state) ** 2

        probs = np.clip(probs, 0.0, None)
        probs /= probs.sum()

        if self.shots > 0:
            samples = np.random.multinomial(self.shots, probs)
            probs = samples / float(self.shots)

        return probs.astype(np.float64)
