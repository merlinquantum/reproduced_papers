"""Batched statevector simulation of a feature-map + ansatz circuit.

The author code evaluates every loss call through Qiskit Aer's
``QasmSimulator`` with 1024 shots, one circuit per sample. At five to seven
qubits the per-circuit overhead dominates (about 5 s per 400-sample forward
pass with Aer SamplerV2 on our machines). This module compiles the Qiskit
circuit once into a list of numpy gate operations and evaluates all samples
in one batch. Ideal (noiseless) QASM sampling with measurements at the end is
a multinomial draw from the exact Born probabilities, so drawing
``shots`` counts per sample from the exact class probabilities reproduces the
same distribution as the author's sampler. ``tests/test_simulator.py``
checks the exact probabilities against ``qiskit.quantum_info.Statevector``.

Bit ordering follows Qiskit: basis index ``sum_q b_q 2**q``. Qubit ``q`` is
stored on tensor axis ``1 + (n - 1 - q)`` so that reshaping to
``(batch, 2**n)`` gives Qiskit's integer ordering.
"""

from __future__ import annotations

import numpy as np
import sympy
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterExpression

BASIS_GATES = ["h", "x", "sx", "rx", "ry", "rz", "p", "u", "cx", "cz"]


def _single_qubit_matrix(name: str, params: list) -> np.ndarray:
    """Return a ``(2, 2)`` or ``(batch, 2, 2)`` matrix for a named gate."""

    if name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    if name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "sx":
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
    theta = np.asarray(params[0], dtype=float)
    cos, sin = np.cos(theta / 2), np.sin(theta / 2)
    zero = np.zeros_like(theta)
    if name == "rx":
        matrix = [[cos + 0j, -1j * sin], [-1j * sin, cos + 0j]]
    elif name == "ry":
        matrix = [[cos + 0j, -sin + 0j], [sin + 0j, cos + 0j]]
    elif name == "rz":
        matrix = [[np.exp(-0.5j * theta), zero + 0j], [zero + 0j, np.exp(0.5j * theta)]]
    elif name == "p":
        matrix = [
            [np.ones_like(theta) + 0j, zero + 0j],
            [zero + 0j, np.exp(1j * theta)],
        ]
    elif name == "u":
        phi = np.asarray(params[1], dtype=float)
        lam = np.asarray(params[2], dtype=float)
        matrix = [
            [cos + 0j, -np.exp(1j * lam) * sin],
            [np.exp(1j * phi) * sin, np.exp(1j * (phi + lam)) * cos],
        ]
    else:
        raise ValueError(f"unsupported single-qubit gate {name!r}")
    stacked = np.array(matrix, dtype=complex)  # (2, 2) or (2, 2, batch)
    return stacked if stacked.ndim == 2 else np.moveaxis(stacked, -1, 0)


class CompiledCircuit:
    """A feature map composed with an ansatz, compiled for batched evaluation.

    Parameters
    ----------
    feature_map : qiskit.QuantumCircuit
        Encoding circuit; its parameters are the input features.
    ansatz : qiskit.QuantumCircuit
        Trainable circuit; its parameters are the weights.
    """

    def __init__(self, feature_map: QuantumCircuit, ansatz: QuantumCircuit) -> None:
        if feature_map.num_qubits != ansatz.num_qubits:
            raise ValueError("feature map and ansatz act on different qubit counts")
        self.n_qubits = feature_map.num_qubits
        self.input_params = list(feature_map.parameters)
        self.weight_params = list(ansatz.parameters)
        self.num_inputs = len(self.input_params)
        self.num_weights = len(self.weight_params)
        # optimization_level=0 keeps one instruction per original rotation, so
        # every parameter expression survives unchanged.
        circuit = transpile(
            feature_map.compose(ansatz), basis_gates=BASIS_GATES, optimization_level=0
        )
        symbols = [sympy.Symbol(f"in{i}") for i in range(self.num_inputs)] + [
            sympy.Symbol(f"w{j}") for j in range(self.num_weights)
        ]
        names = {
            p.name: s for p, s in zip(self.input_params + self.weight_params, symbols)
        }
        self._operations = []
        for instruction in circuit.data:
            name = instruction.operation.name
            if name == "barrier":
                continue
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            params = []
            for value in instruction.operation.params:
                if isinstance(value, ParameterExpression) and value.parameters:
                    expression = value.sympify().subs(
                        {sympy.Symbol(n): s for n, s in names.items()}
                    )
                    params.append(sympy.lambdify(symbols, expression, modules="numpy"))
                else:
                    params.append(float(value))
            self._operations.append((name, qubits, params))

    def _axis(self, qubit: int) -> int:
        return 1 + (self.n_qubits - 1 - qubit)

    def statevector(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return final states of shape ``(batch, 2**n_qubits)``.

        Parameters
        ----------
        X : numpy.ndarray
            Inputs, shape ``(batch, n_qubits)``.
        weights : numpy.ndarray
            Ansatz parameters, shape ``(num_weights,)``.
        """

        X = np.atleast_2d(np.asarray(X, dtype=float))
        weights = np.asarray(weights, dtype=float)
        if X.shape[1] != self.num_inputs or weights.shape != (self.num_weights,):
            raise ValueError(
                f"expected X (*, {self.num_inputs}) and weights ({self.num_weights},)"
            )
        batch = X.shape[0]
        arguments = [X[:, i] for i in range(self.num_inputs)] + list(weights)
        state = np.zeros((batch,) + (2,) * self.n_qubits, dtype=complex)
        state[(slice(None),) + (0,) * self.n_qubits] = 1.0
        for name, qubits, params in self._operations:
            values = [p(*arguments) if callable(p) else p for p in params]
            if name in ("cx", "cz"):
                state = self._apply_controlled(state, name, *qubits)
            else:
                state = self._apply_single(
                    state, _single_qubit_matrix(name, values), qubits[0], batch
                )
        return state.reshape(batch, -1)

    def _apply_single(self, state, matrix, qubit, batch):
        axis = self._axis(qubit)
        moved = np.moveaxis(state, axis, -1)
        shape = moved.shape
        flat = moved.reshape(batch, -1, 2)
        if matrix.ndim == 2:
            flat = flat @ matrix.T
        else:
            if matrix.shape[0] != batch:
                matrix = np.broadcast_to(matrix, (batch, 2, 2))
            flat = np.einsum("bkj,bij->bki", flat, matrix)
        return np.moveaxis(flat.reshape(shape), -1, axis)

    def _apply_controlled(self, state, name, control, target):
        state = state.copy()
        control_axis, target_axis = self._axis(control), self._axis(target)
        index = [slice(None)] * state.ndim
        index[control_axis] = 1
        sub = state[tuple(index)]
        sub_target_axis = target_axis - 1 if target_axis > control_axis else target_axis
        if name == "cx":
            state[tuple(index)] = np.flip(sub, axis=sub_target_axis)
        else:
            flipped = sub.copy()
            target_index = [slice(None)] * sub.ndim
            target_index[sub_target_axis] = 1
            flipped[tuple(target_index)] *= -1
            state[tuple(index)] = flipped
        return state

    def probabilities(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Exact Born probabilities, shape ``(batch, 2**n_qubits)``."""

        return np.abs(self.statevector(X, weights)) ** 2


def class_probabilities(
    circuit: CompiledCircuit,
    X: np.ndarray,
    weights: np.ndarray,
    num_classes: int,
    shots: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Class probabilities with qiskit-ml VQC's readout, optionally shot-sampled.

    qiskit-machine-learning (0.6.0 used by the author, and 0.8.4) interprets
    the measured integer ``x`` as class ``x % num_classes``. With ``shots``
    set, counts are drawn from a multinomial per sample and divided by
    ``shots``, which matches ideal QASM sampling in distribution.

    Parameters
    ----------
    circuit : CompiledCircuit
    X : numpy.ndarray
        Inputs, shape ``(batch, n_qubits)``.
    weights : numpy.ndarray
        Ansatz parameters.
    num_classes : int
        Number of one-hot classes.
    shots : int | None
        Shots per sample; None returns exact probabilities.
    rng : numpy.random.Generator
        Source of shot noise.

    Returns
    -------
    numpy.ndarray
        Shape ``(batch, num_classes)``.
    """

    probabilities = circuit.probabilities(X, weights)
    readout = np.zeros((probabilities.shape[1], num_classes))
    readout[
        np.arange(probabilities.shape[1]),
        np.arange(probabilities.shape[1]) % num_classes,
    ] = 1.0
    exact = probabilities @ readout
    exact = np.clip(exact, 0.0, None)
    exact /= exact.sum(axis=1, keepdims=True)
    if shots is None:
        return exact
    return rng.multinomial(int(shots), exact) / float(shots)
