from __future__ import annotations

import numpy as np
import torch
from common import PROJECT_DIR  # noqa: F401  (puts the project on sys.path)
from lib import statevector as sv
from scipy.linalg import expm

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
DTYPE = torch.float64


def _random_state(n_qubits: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(1, *([2] * n_qubits), generator=generator, dtype=DTYPE)
    imag = torch.randn(1, *([2] * n_qubits), generator=generator, dtype=DTYPE)
    state = torch.complex(real, imag)
    return state / state.abs().pow(2).sum().sqrt()


def test_single_qubit_rotations_match_matrix_exponentials():
    theta = 0.83
    for matrix_fn, pauli in ((sv.rx_matrix, X), (sv.ry_matrix, Y), (sv.rz_matrix, Z)):
        got = matrix_fn(torch.tensor(theta, dtype=DTYPE)).numpy()
        np.testing.assert_allclose(got, expm(-1j * theta / 2 * pauli), atol=1e-12)


def test_rxx_matches_matrix_exponential():
    theta = 1.17
    got = sv.rxx_matrix(torch.tensor(theta, dtype=DTYPE)).numpy()
    np.testing.assert_allclose(got, expm(-1j * theta / 2 * np.kron(X, X)), atol=1e-12)


def test_gates_on_non_adjacent_wires_match_dense_construction():
    state = _random_state(4)
    ry = sv.ry_matrix(torch.tensor(0.4, dtype=DTYPE))
    rxx = sv.rxx_matrix(torch.tensor(0.9, dtype=DTYPE))
    out = sv.apply_one_qubit(state, ry, 2)
    out = sv.apply_two_qubit(out, rxx, 3, 0)
    ry_dense = np.kron(np.kron(np.eye(4), ry.numpy()), np.eye(2))
    x_wire0 = np.kron(X, np.eye(8))
    x_wire3 = np.kron(np.eye(8), X)
    rxx_dense = expm(-1j * 0.9 / 2 * (x_wire0 @ x_wire3))
    expected = rxx_dense @ ry_dense @ state.reshape(-1).numpy()
    np.testing.assert_allclose(out.reshape(-1).numpy(), expected, atol=1e-12)


def test_per_sample_gates_match_shared_gates():
    state = _random_state(3, seed=1).repeat(2, 1, 1, 1)
    angles = torch.tensor([0.3, 0.3], dtype=DTYPE)
    batched = sv.apply_one_qubit(state, sv.ry_matrix(angles), 1)
    shared = sv.apply_one_qubit(state, sv.ry_matrix(angles[0]), 1)
    torch.testing.assert_close(batched, shared)


def test_marginal_probabilities_order_wires_most_significant_first():
    state = torch.zeros(1, 2, 2, 2, dtype=torch.complex128)
    state[0, 1, 0, 1] = 1.0  # |101>
    one = torch.tensor(1.0, dtype=DTYPE)
    assert torch.isclose(sv.marginal_probabilities(state, [0, 2])[0, 0b11], one)
    assert torch.isclose(sv.marginal_probabilities(state, [1, 0])[0, 0b01], one)
    assert torch.isclose(sv.marginal_probabilities(state, [0, 1])[0, 0b10], one)
