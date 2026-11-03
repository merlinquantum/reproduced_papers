from __future__ import annotations

import math

import common  # noqa: F401  (puts the paper package on sys.path)
import torch
from trainable_frequency_qnn.lib.statevector import (
    apply_cnot,
    apply_cz,
    apply_single_qubit,
    rotation_matrix,
    total_magnetization,
    zero_state,
)

N_QUBITS = 3


def _random_state(batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    real = torch.randn(
        (batch,) + (2,) * N_QUBITS, generator=generator, dtype=torch.float64
    )
    imag = torch.randn(
        (batch,) + (2,) * N_QUBITS, generator=generator, dtype=torch.float64
    )
    state = torch.complex(real, imag)
    return state / state.reshape(batch, -1).norm(dim=1).reshape(
        (batch,) + (1,) * N_QUBITS
    )


def _dense_two_qubit(gate: torch.Tensor, first: int, second: int) -> torch.Tensor:
    """Embed a 4x4 gate on (first, second) into the 8x8 operator, qubit 0 = MSB."""
    dim = 2**N_QUBITS
    operator = torch.zeros((dim, dim), dtype=torch.complex128)
    for column in range(dim):
        bits = [(column >> (N_QUBITS - 1 - q)) & 1 for q in range(N_QUBITS)]
        local_in = 2 * bits[first] + bits[second]
        for local_out in range(4):
            amplitude = gate[local_out, local_in]
            if amplitude == 0:
                continue
            out_bits = list(bits)
            out_bits[first], out_bits[second] = local_out >> 1, local_out & 1
            row = sum(b << (N_QUBITS - 1 - q) for q, b in enumerate(out_bits))
            operator[row, column] += amplitude
    return operator


def test_single_qubit_ry_expectation_is_cosine():
    angles = torch.linspace(-3.0, 3.0, 7, dtype=torch.float64)
    state = zero_state(len(angles), 1, torch.complex128, "cpu")
    state = apply_single_qubit(state, rotation_matrix("y", angles), 0)
    assert torch.allclose(total_magnetization(state, 1), torch.cos(angles))


def test_rx_matches_matrix_exponential():
    angle = torch.tensor(0.7, dtype=torch.float64)
    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    expected = torch.linalg.matrix_exp(-0.5j * angle * pauli_x)
    assert torch.allclose(rotation_matrix("x", angle), expected)


def test_per_sample_gate_matches_shared_gate():
    state = _random_state(batch=3)
    angles = torch.full((3,), 0.4, dtype=torch.float64)
    shared = apply_single_qubit(state, rotation_matrix("y", angles[0]), 1)
    per_sample = apply_single_qubit(state, rotation_matrix("y", angles), 1)
    assert torch.allclose(shared, per_sample)


def test_cnot_and_cz_match_dense_operators():
    state = _random_state()
    flat = state.reshape(state.shape[0], -1)
    cnot = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.complex128
    )
    cz = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex128))
    for control, target in [(0, 1), (2, 0), (1, 2)]:
        expected = flat @ _dense_two_qubit(cnot, control, target).T
        result = apply_cnot(state, control, target).reshape(state.shape[0], -1)
        assert torch.allclose(result, expected)
        expected_cz = flat @ _dense_two_qubit(cz, control, target).T
        result_cz = apply_cz(state, control, target, N_QUBITS).reshape(
            state.shape[0], -1
        )
        assert torch.allclose(result_cz, expected_cz)


def test_total_magnetization_of_basis_state():
    state = zero_state(1, N_QUBITS, torch.complex128, "cpu")
    state = apply_single_qubit(
        state, rotation_matrix("x", torch.tensor(math.pi, dtype=torch.float64)), 2
    )
    assert torch.allclose(
        total_magnetization(state, N_QUBITS), torch.tensor([1.0], dtype=torch.float64)
    )
