"""Contextual QNN with the quantum batch gradient update (paper Sec. IV, V).

Registers (big-endian, see ``lib.data``): ``T * q`` context qubits followed by
``q`` prediction qubits, ``q = log2(levels)``.

QBGU input: ``sum_c sqrt(P(c)) |c> |0>`` (Eq. 16 with the prediction register
in ``|0>``). Target: ``sum_{c,o} sqrt(P(c, o)) |c, o>``. The SWAP test between
the circuit output ``|y>`` and the target returns ancilla outcome 0 with
probability ``P(0) = (1 + F) / 2``, ``F = |<y|target>|^2`` (Eq. 13), and the
loss is ``L_f = 1 - P(0) = (1 - F) / 2`` (Eq. 14). For pure states this is
exactly the statistics of the SWAP-test circuit (checked against an explicit
CSWAP circuit in ``tests/test_model.py``), so the ancilla is sampled from a
binomial instead of simulating the 2n + 1 qubit circuit.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .statevector import basis_state, probabilities

StateMap = Callable[[np.ndarray], np.ndarray]


class ContextualTask:
    """Exact input and target states for one asset's empirical distribution.

    Parameters
    ----------
    joint : numpy.ndarray
        Empirical ``P(context, next)`` of shape ``(levels**T, levels)``.
    context_state : numpy.ndarray or None
        State of the context register to use as input. ``None`` uses the
        exact ``sum_c sqrt(P(c)) |c>``; a trained loader's state can be passed
        instead (experiment E2/E3 loader variant). Default value is None.
    """

    def __init__(
        self, joint: np.ndarray, context_state: np.ndarray | None = None
    ) -> None:
        self.joint = np.asarray(joint, dtype=float)
        self.n_contexts, self.levels = self.joint.shape
        self.qubits_per_step = int(math.log2(self.levels))
        self.n_context_qubits = int(math.log2(self.n_contexts))
        self.n_qubits = self.n_context_qubits + self.qubits_per_step

        if context_state is None:
            context_state = np.sqrt(self.joint.sum(axis=1)).astype(np.complex128)
        if context_state.shape != (self.n_contexts,):
            raise ValueError("context_state has the wrong dimension")
        prediction_zero = basis_state(0, self.qubits_per_step)
        self.input_state = np.kron(context_state, prediction_zero)
        self.target_state = np.sqrt(self.joint).reshape(-1).astype(np.complex128)

    def output(self, circuit: StateMap) -> np.ndarray:
        return circuit(self.input_state)

    def fidelity(self, circuit: StateMap) -> float:
        return float(abs(np.vdot(self.target_state, self.output(circuit))) ** 2)

    def swap_test_loss(
        self, circuit: StateMap, shots: int | None, rng: np.random.Generator
    ) -> float:
        """Eq. 14 loss, estimated from ``shots`` ancilla samples (exact if None)."""

        p_zero = 0.5 * (1.0 + self.fidelity(circuit))
        if shots is not None:
            p_zero = rng.binomial(shots, p_zero) / shots
        return 1.0 - p_zero

    def mse_loss(
        self, circuit: StateMap, shots: int | None, rng: np.random.Generator
    ) -> float:
        """Eq. 12 loss on the measured joint distribution (all qubits measured).

        Squared Euclidean distance between the sampled output frequencies and
        the empirical joint distribution.
        """

        probs = probabilities(self.output(circuit))
        if shots is not None:
            probs = rng.multinomial(shots, probs / probs.sum()) / shots
        return float(np.sum((probs - self.joint.reshape(-1)) ** 2))

    def loss(
        self, kind: str, circuit: StateMap, shots: int | None, rng: np.random.Generator
    ) -> float:
        if kind == "swap_test_fidelity":
            return self.swap_test_loss(circuit, shots, rng)
        if kind == "mse":
            return self.mse_loss(circuit, shots, rng)
        raise ValueError(f"Unknown loss '{kind}'")

    def sample_window(self, rng: np.random.Generator) -> tuple[int, int]:
        """Draw one training window ``(context, next)`` from the empirical joint.

        Equivalent to picking a training window uniformly at random, since the
        joint is the normalized window count.
        """

        flat = rng.choice(self.joint.size, p=self.joint.reshape(-1))
        return int(flat // self.levels), int(flat % self.levels)

    def per_sample_swap_loss(
        self,
        circuit: StateMap,
        context: int,
        outcome: int,
        shots: int | None,
        rng: np.random.Generator,
    ) -> float:
        """Eq. 14 loss for one window: input ``|c>|0>``, one-hot target ``|c, o>``.

        This is the per-sample objective of Eq. 18-19, estimated with the same
        SWAP-test statistics as the QBGU loss.
        """

        fidelity = per_sample_fidelity(
            circuit, context, outcome, self.n_contexts, self.levels
        )
        p_zero = 0.5 * (1.0 + fidelity)
        if shots is not None:
            p_zero = rng.binomial(shots, p_zero) / shots
        return 1.0 - p_zero

    def expected_per_sample_fidelity(self, circuit: StateMap) -> float:
        """``sum_{c,o} P(c, o) |<c, o| U |c, 0>|^2``: the per-sample objective.

        Linear in the model's probabilities ``|<c, o|U|c, 0>|^2``, so for each
        context it is maximized by putting all weight on the most frequent
        next step rather than on ``P(o | c)``.
        """

        return float(np.sum(self.joint * self.one_hot_fidelities(circuit)))

    def one_hot_fidelities(self, circuit: StateMap) -> np.ndarray:
        """``|<c, o| U |c, 0>|^2`` for every window type, shape ``(contexts, levels)``."""

        fidelities = np.zeros_like(self.joint)
        for context in range(self.n_contexts):
            state = basis_state(context * self.levels, self.n_qubits)
            probs = probabilities(circuit(state)).reshape(self.n_contexts, self.levels)
            fidelities[context] = probs[context]
        return fidelities

    def minibatch_swap_loss(
        self,
        circuit: StateMap,
        window_counts: np.ndarray,
        shots_per_window: int | None,
        rng: np.random.Generator,
    ) -> float:
        """Mean Eq. 14 loss over a mini-batch of windows (per-sample objective).

        ``window_counts[c, o]`` is how many windows of type ``(c, o)`` the
        batch holds; each window gets its own SWAP test with
        ``shots_per_window`` shots (exact if None). Windows of one type share
        the same success probability, so their ancilla outcomes are pooled
        into one binomial draw.
        """

        p_zero = 0.5 * (1.0 + self.one_hot_fidelities(circuit))
        batch = int(window_counts.sum())
        if shots_per_window is None:
            return float(1.0 - np.sum(window_counts * p_zero) / batch)
        successes = rng.binomial(window_counts.astype(int) * shots_per_window, p_zero)
        return float(1.0 - successes.sum() / (batch * shots_per_window))

    def output_joint(self, circuit: StateMap) -> np.ndarray:
        """Measured distribution of the full output state, ``(contexts, levels)``."""

        return probabilities(self.output(circuit)).reshape(self.n_contexts, self.levels)

    def inference_conditional(self, circuit: StateMap) -> np.ndarray:
        """``q(next | c)`` from running the circuit on each basis context ``|c>|0>``.

        This is the paper's inference use (Sec. V.C): load one context, measure
        the prediction register. Probabilities are marginalized over the
        context register, which the circuit may have changed.
        """

        conditional = np.zeros((self.n_contexts, self.levels))
        for context in range(self.n_contexts):
            state = basis_state(context * self.levels, self.n_qubits)
            probs = probabilities(circuit(state)).reshape(self.n_contexts, self.levels)
            conditional[context] = probs.sum(axis=0)
        return conditional


def per_sample_fidelity(
    circuit: StateMap, context: int, outcome: int, n_contexts: int, levels: int
) -> float:
    """Fidelity of ``U |c>|0>`` with the one-hot target ``|c, o>`` (per-sample SGD, E5)."""

    n_qubits = int(math.log2(n_contexts * levels))
    output = circuit(basis_state(context * levels, n_qubits))
    return float(abs(output[context * levels + outcome]) ** 2)


def loader_mse(
    circuit: StateMap,
    context_marginal: np.ndarray,
    shots: int | None,
    rng: np.random.Generator,
) -> float:
    """MSE between the loader's measured distribution and ``P(c)`` (Sec. III, Eq. 12)."""

    n_qubits = int(math.log2(len(context_marginal)))
    probs = probabilities(circuit(basis_state(0, n_qubits)))
    if shots is not None:
        probs = rng.multinomial(shots, probs / probs.sum()) / shots
    return float(np.sum((probs - context_marginal) ** 2))
