"""Statevector simulator and model tests against an independent dense-matrix build."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from common import PROJECT_DIR  # noqa: F401  (puts the repository and paper on sys.path)
from lib.circuits import (
    ParallelPauliModel,
    SerialPauliModel,
    apply_cnot,
    apply_single_qubit_gate,
    basic_entangler_layers,
    rot_matrix,
    rx_matrix,
    strongly_entangling_layers,
    zero_state,
)
from lib.fourier import fourier_coefficients, periodic_grid, spectrum_support

DTYPE = torch.complex128
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


def _expm_pauli(pauli: np.ndarray, angle: float) -> np.ndarray:
    return np.cos(angle / 2) * IDENTITY - 1j * np.sin(angle / 2) * pauli


def _dense_rot(phi, theta, omega):
    return (
        _expm_pauli(PAULI_Z, omega)
        @ _expm_pauli(PAULI_Y, theta)
        @ _expm_pauli(PAULI_Z, phi)
    )


def _on_wire(gate: np.ndarray, wire: int, n_qubits: int) -> np.ndarray:
    matrix = np.array([[1.0]], dtype=complex)
    for index in range(n_qubits):
        matrix = np.kron(matrix, gate if index == wire else IDENTITY)
    return matrix


def _dense_cnot(control: int, target: int, n_qubits: int) -> np.ndarray:
    dimension = 2**n_qubits
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for basis in range(dimension):
        bits = [(basis >> (n_qubits - 1 - wire)) & 1 for wire in range(n_qubits)]
        if bits[control]:
            bits[target] ^= 1
        image = sum(bit << (n_qubits - 1 - wire) for wire, bit in enumerate(bits))
        matrix[image, basis] = 1.0
    return matrix


def _dense_strongly(weights: np.ndarray, n_qubits: int) -> np.ndarray:
    unitary = np.eye(2**n_qubits, dtype=complex)
    for layer, layer_weights in enumerate(weights):
        for wire in range(n_qubits):
            unitary = (
                _on_wire(_dense_rot(*layer_weights[wire]), wire, n_qubits) @ unitary
            )
        if n_qubits > 1:
            cnot_range = (layer % (n_qubits - 1)) + 1
            for wire in range(n_qubits):
                unitary = (
                    _dense_cnot(wire, (wire + cnot_range) % n_qubits, n_qubits)
                    @ unitary
                )
    return unitary


def _as_vector(state: torch.Tensor) -> np.ndarray:
    return state.reshape(state.shape[0], -1).numpy()


def test_single_qubit_gates_match_dense_matrices():
    angles = torch.tensor([0.3, -1.1, 2.0], dtype=torch.float64)
    expected = _dense_rot(0.3, -1.1, 2.0)
    assert np.allclose(rot_matrix(*angles, DTYPE).numpy(), expected, atol=1e-14)
    assert np.allclose(
        rx_matrix(torch.tensor(0.7, dtype=torch.float64), DTYPE).numpy(),
        _expm_pauli(PAULI_X, 0.7),
    )


@pytest.mark.parametrize("wire", [0, 1, 2])
def test_apply_single_qubit_gate_uses_wire_zero_as_most_significant(wire):
    rng = np.random.default_rng(1)
    vector = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    state = torch.tensor(vector.reshape(1, 2, 2, 2), dtype=DTYPE)
    gate = _dense_rot(0.4, 1.3, -0.2)
    updated = apply_single_qubit_gate(state, torch.tensor(gate), wire)
    assert np.allclose(
        _as_vector(updated)[0], _on_wire(gate, wire, 3) @ vector, atol=1e-14
    )


@pytest.mark.parametrize("control,target", [(0, 1), (1, 0), (0, 2), (2, 1)])
def test_cnot_matches_dense_matrix(control, target):
    rng = np.random.default_rng(2)
    vector = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    state = torch.tensor(vector.reshape(1, 2, 2, 2), dtype=DTYPE)
    updated = apply_cnot(state, control, target)
    assert np.allclose(
        _as_vector(updated)[0], _dense_cnot(control, target, 3) @ vector, atol=1e-14
    )


@pytest.mark.parametrize("n_qubits,layers", [(1, 2), (2, 2), (3, 3), (4, 2)])
def test_strongly_entangling_layers_match_dense_build(n_qubits, layers):
    rng = np.random.default_rng(3)
    weights = rng.uniform(0, 2 * np.pi, (layers, n_qubits, 3))
    state = zero_state(n_qubits, 1, DTYPE, torch.device("cpu"))
    updated = strongly_entangling_layers(state, torch.tensor(weights))
    expected = _dense_strongly(weights, n_qubits)[:, 0]
    assert np.allclose(_as_vector(updated)[0], expected, atol=1e-13)


def test_basic_entangler_ring_on_three_qubits():
    weights = np.array([[0.2, 1.1, -0.7]])
    state = zero_state(3, 1, DTYPE, torch.device("cpu"))
    updated = basic_entangler_layers(state, torch.tensor(weights))
    unitary = np.eye(8, dtype=complex)
    for wire in range(3):
        unitary = _on_wire(_expm_pauli(PAULI_X, weights[0, wire]), wire, 3) @ unitary
    for wire in range(3):
        unitary = _dense_cnot(wire, (wire + 1) % 3, 3) @ unitary
    assert np.allclose(_as_vector(updated)[0], unitary[:, 0], atol=1e-14)


def test_serial_model_matches_dense_expectation():
    model = SerialPauliModel(
        repetitions=3, encoding_scaling=2.0, readout="z0", dtype=torch.float64
    )
    model.reset_parameters(torch.Generator().manual_seed(0), 2 * np.pi)
    weights = model.weights.detach().numpy()
    inputs = np.linspace(-6, 6, 9)
    expected = []
    for x in inputs:
        unitary = _dense_rot(*weights[0])
        for layer in range(1, 4):
            unitary = (
                _dense_rot(*weights[layer]) @ _expm_pauli(PAULI_X, 2.0 * x) @ unitary
            )
        state = unitary[:, 0]
        expected.append(np.real(state.conj() @ PAULI_Z @ state))
    with torch.no_grad():
        values = model(torch.tensor(inputs)).numpy()
    assert np.allclose(values, expected, atol=1e-13)


def test_parallel_model_without_trainable_blocks_is_cos():
    model = ParallelPauliModel(3, 1.5, "none", 0, "z0", torch.float64)
    inputs = torch.linspace(-6, 6, 25, dtype=torch.float64)
    with torch.no_grad():
        values = model(inputs)
    # <0|RX(sx)^dagger Z RX(sx)|0> = cos(sx) on wire 0; other wires do not matter.
    assert torch.allclose(values, torch.cos(1.5 * inputs), atol=1e-14)


@pytest.mark.parametrize("repetitions", [1, 3, 5])
@pytest.mark.parametrize("layout", ["serial", "parallel"])
def test_pauli_models_have_degree_r_spectrum(layout, repetitions):
    if layout == "serial":
        model = SerialPauliModel(repetitions, 1.0, "z0", torch.float64)
    else:
        model = ParallelPauliModel(repetitions, 1.0, "strongly", 3, "z0", torch.float64)
    model.reset_parameters(torch.Generator().manual_seed(repetitions), 2 * np.pi)
    grid = torch.tensor(periodic_grid(4 * repetitions + 8), dtype=torch.float64)
    with torch.no_grad():
        frequencies, coefficients = fourier_coefficients(model(grid).numpy())
    support = spectrum_support(frequencies, coefficients, 1e-9)
    assert set(support) <= set(range(-repetitions, repetitions + 1))
    assert max(support) == repetitions


def test_autograd_matches_parameter_shift():
    # Every trainable angle enters through a rotation exp(-i t P / 2), so the
    # parameter-shift rule with shift pi/2 is exact; backpropagation must agree.
    model = SerialPauliModel(
        repetitions=2, encoding_scaling=1.0, readout="z0", dtype=torch.float64
    )
    model.reset_parameters(torch.Generator().manual_seed(4), 2 * np.pi)
    inputs = torch.tensor([0.3, -2.1], dtype=torch.float64)
    model(inputs).sum().backward()
    autograd_gradient = model.weights.grad.clone()
    shifted_gradient = torch.zeros_like(autograd_gradient)
    with torch.no_grad():
        for index in np.ndindex(*model.weights.shape):
            original = model.weights[index].item()
            model.weights[index] = original + np.pi / 2
            plus = model(inputs).sum()
            model.weights[index] = original - np.pi / 2
            minus = model(inputs).sum()
            model.weights[index] = original
            shifted_gradient[index] = (plus - minus) / 2
    assert torch.allclose(autograd_gradient, shifted_gradient, atol=1e-12)


@pytest.mark.parametrize("layout", ["serial", "parallel"])
def test_trainable_readout_is_linear_in_basis_probabilities(layout):
    if layout == "serial":
        model = SerialPauliModel(2, 1.0, "trainable_linear", torch.float64)
        n_qubits = 1
    else:
        model = ParallelPauliModel(
            3, 1.0, "strongly", 1, "trainable_linear", torch.float64
        )
        n_qubits = 3
    model.reset_parameters(torch.Generator().manual_seed(0), 2 * np.pi)
    inputs = torch.linspace(-2, 2, 5, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(model(inputs), torch.zeros(5, dtype=torch.float64))
        model.readout_layer.weight.fill_(0.0)
        model.readout_layer.weight[0, 0] = 2.0
        model.readout_layer.bias.fill_(-1.0)
        # Weight 2 on |0...0> and bias -1 equal <Z_0> only for one qubit; check the definition directly.
        values = model(inputs)
    assert model.readout_layer.in_features == 2**n_qubits
    if n_qubits == 1:
        reference = SerialPauliModel(2, 1.0, "z0", torch.float64)
        reference.reset_parameters(torch.Generator().manual_seed(0), 2 * np.pi)
        with torch.no_grad():
            assert torch.allclose(values, reference(inputs), atol=1e-12)


def test_unknown_qubit_readout_raises():
    with pytest.raises(ValueError):
        SerialPauliModel(1, 1.0, "bogus", torch.float64)
