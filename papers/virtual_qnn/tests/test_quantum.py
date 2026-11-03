"""Simulator primitives against explicit matrix constructions."""

from __future__ import annotations

import pytest
import torch
from common import PROJECT_DIR  # noqa: F401  (puts the repo root on sys.path)
from virtual_qnn.lib import quantum

DTYPE = torch.complex128
I2 = torch.eye(2, dtype=DTYPE)
PAULIS = [
    I2,
    torch.tensor([[0, 1], [1, 0]], dtype=DTYPE),
    torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE),
    torch.diag(torch.tensor([1, -1], dtype=DTYPE)),
]


@pytest.fixture
def random_state():
    generator = torch.Generator().manual_seed(0)
    real = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    imag = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    matrix = real + 1j * imag
    rho = matrix @ matrix.conj().T
    return rho / torch.trace(rho)


def test_single_qubit_unitary_matches_kron(random_state):
    unitary = quantum.rx(torch.tensor(0.7, dtype=torch.float64), DTYPE)
    full = quantum.kron_all([I2, unitary, I2])
    expected = full @ random_state @ full.conj().T
    assert torch.allclose(
        quantum.apply_unitary(random_state, unitary, [1], 3), expected
    )


def test_cnot_target_order(random_state):
    cnot = quantum.cnot(DTYPE)
    swap = torch.eye(4, dtype=DTYPE)[[0, 2, 1, 3]]
    full = torch.kron(I2, swap @ cnot @ swap)
    expected = full @ random_state @ full.conj().T
    assert torch.allclose(
        quantum.apply_unitary(random_state, cnot, [2, 1], 3), expected
    )


def test_depolarize_equals_pauli_twirl(random_state):
    probability = 0.3
    twirl = (
        sum(
            quantum.apply_unitary(random_state, torch.kron(a, b), [0, 2], 3)
            for a in PAULIS
            for b in PAULIS
        )
        / 16
    )
    expected = (1 - probability) * random_state + probability * twirl
    assert torch.allclose(
        quantum.depolarize(random_state, [0, 2], 3, probability), expected
    )


def test_partial_trace_keeps_requested_order(random_state):
    expected = torch.einsum("abcdbf->cafd", random_state.reshape([2] * 6)).reshape(4, 4)
    assert torch.allclose(quantum.partial_trace(random_state, [2, 0], 3), expected)


def test_rotations_match_matrix_exponential():
    theta = torch.tensor(0.4, dtype=torch.float64)
    for gate, pauli in (
        (quantum.rx, PAULIS[1]),
        (quantum.ry, PAULIS[2]),
        (quantum.rz, PAULIS[3]),
    ):
        assert torch.allclose(
            gate(theta, DTYPE), torch.linalg.matrix_exp(-0.5j * theta * pauli)
        )


def test_depolarizing_convention_matches_paper_fig5():
    """Unmitigated squared HS error of D_0.2 on half a Bell pair is 0.75 p^2 = 0.03."""
    bell = quantum.bell_state(DTYPE)
    noisy = quantum.depolarize(bell, [1], 2, 0.2)
    assert quantum.squared_hs_distance(bell, noisy).item() == pytest.approx(0.03)
