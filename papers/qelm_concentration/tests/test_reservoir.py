from __future__ import annotations

import common  # noqa: F401  (puts the paper package on sys.path)
import numpy as np
import pytest
import torch
from qelm_concentration.lib.reservoir import (
    KickedIsingReservoir,
    ReservoirArchitecture,
    ReservoirHyperparameters,
    pair_reduced_density_matrices,
    sample_block_parameters,
)

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Z = np.diag([1.0, -1.0]).astype(complex)
TABLE_I = ReservoirHyperparameters(0.2, 0.707, 0.031, 0.683, 0.034, 0.237, 0.038, 0.1)


def _embed(op: np.ndarray, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """Dense operator acting as ``op`` on ``qubits`` (qubit 0 most significant)."""
    dim = 2**n
    full = np.zeros((dim, dim), dtype=complex)
    k = len(qubits)
    for col in range(dim):
        bits = [(col >> (n - 1 - q)) & 1 for q in range(n)]
        sub_col = sum(bits[q] << (k - 1 - i) for i, q in enumerate(qubits))
        for sub_row in range(2**k):
            new_bits = list(bits)
            for i, q in enumerate(qubits):
                new_bits[q] = (sub_row >> (k - 1 - i)) & 1
            row = sum(b << (n - 1 - q) for q, b in enumerate(new_bits))
            full[row, col] += op[sub_row, sub_col]
    return full


def _qiskit_rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def _qiskit_rx(theta: float) -> np.ndarray:
    return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * PAULI_X


def _qiskit_rzz(theta: float) -> np.ndarray:
    return np.diag(np.exp(-0.5j * theta * np.array([1, -1, -1, 1])))


def _block_unitary(J: float, h: float, b: float, block_order: str) -> np.ndarray:
    """4x4 block unitary on the (first, second) qubit of a pair."""
    if block_order == "figure1":
        # Paper Fig. 1, time order: RZ(h) on the second qubit, RZZ(J), RX(b) on both,
        # RZZ(J), RZ(h) on the second qubit (Qiskit gate conventions).
        rz_second = np.kron(np.eye(2), _qiskit_rz(h))
        kick = np.kron(_qiskit_rx(b), _qiskit_rx(b))
        unitary = np.eye(4, dtype=complex)
        for gate in (rz_second, _qiskit_rzz(J), kick, _qiskit_rzz(J), rz_second):
            unitary = gate @ unitary
        return unitary
    zz = np.kron(PAULI_Z, PAULI_Z)
    z_sum = np.kron(PAULI_Z, np.eye(2)) + np.kron(np.eye(2), PAULI_Z)
    diagonal = np.diag(np.exp(-1j * np.diag(J * zz + h * z_sum)))
    rx = np.cos(b) * np.eye(2) - 1j * np.sin(b) * PAULI_X
    kick = np.kron(rx, rx)
    return diagonal @ kick if block_order == "kick_then_ising" else kick @ diagonal


def _dense_reference(arch, params, a_in, inputs):
    n = arch.n_qubits
    n_blocks = arch.n_blocks
    if arch.bell_pairs == "even":
        bell_pairs = [(2 * m, 2 * m + 1) for m in range(n_blocks)]
    else:
        bell_pairs = [((2 * m + 1) % n, (2 * m + 2) % n) for m in range(n_blocks)]
    state = np.zeros(2**n, dtype=complex)
    for index in range(2**n):
        bits = [(index >> (n - 1 - q)) & 1 for q in range(n)]
        if all(bits[a] == bits[b] for a, b in bell_pairs):
            state[index] = 2 ** (-n_blocks / 2)
    encoding_layer = 0
    for layer in range(arch.n_layers):
        feed = None
        if arch.is_encoding_layer(layer):
            feed = arch.input_indices(encoding_layer, len(inputs))
            encoding_layer += 1
        for block, pair in enumerate(arch.layer_pairs(layer)):
            b = (
                a_in * inputs[feed[block]]
                if feed is not None
                else params.b[layer, block]
            )
            unitary = _block_unitary(
                params.J[layer, block], params.h[layer, block], b, arch.block_order
            )
            state = _embed(unitary, pair, n) @ state
    return state


@pytest.mark.parametrize(
    "n_qubits,n_layers,shift,offset,block_order,bell_pairs",
    [
        (4, 6, 1, 0, "figure1", "odd"),
        (6, 5, 4, 0, "figure1", "odd"),
        (6, 5, 4, 1, "figure1", "even"),
        (6, 5, 4, 0, "ising_then_kick", "even"),
        (6, 5, 4, 1, "kick_then_ising", "odd"),
    ],
)
def test_statevector_matches_dense_reference(
    n_qubits, n_layers, shift, offset, block_order, bell_pairs
):
    arch = ReservoirArchitecture(
        n_qubits, n_layers, 4, shift, 1.0, offset, block_order, bell_pairs
    )
    params = sample_block_parameters(TABLE_I, arch, np.random.default_rng(3))
    reservoir = KickedIsingReservoir(arch, TABLE_I, params, "cpu", torch.float64)
    inputs = np.random.default_rng(4).uniform(-1, 1, size=(3, 5))
    states = reservoir.evolve(torch.as_tensor(inputs)).numpy()
    for sample in range(3):
        expected = _dense_reference(
            arch, params, TABLE_I.input_strength, inputs[sample]
        )
        np.testing.assert_allclose(states[sample], expected, atol=1e-10)


def test_pair_rdms_match_dense_partial_trace():
    arch = ReservoirArchitecture(
        6, 5, encoding_period=4, input_shift=4, angle_scale=1.0
    )
    params = sample_block_parameters(TABLE_I, arch, np.random.default_rng(0))
    reservoir = KickedIsingReservoir(arch, TABLE_I, params, "cpu", torch.float64)
    states = reservoir.evolve(
        torch.as_tensor(np.random.default_rng(1).uniform(-1, 1, (2, 3)))
    )
    pairs = arch.ring_neighbour_pairs()
    rdms = pair_reduced_density_matrices(states, pairs, 6).numpy()
    for sample in range(2):
        psi = states[sample].numpy().reshape((2,) * 6)
        for index, (qa, qb) in enumerate(pairs):
            moved = np.moveaxis(psi, (qa, qb), (0, 1)).reshape(4, -1)
            np.testing.assert_allclose(
                rdms[sample, index], moved @ moved.conj().T, atol=1e-12
            )
            assert np.isclose(np.trace(rdms[sample, index]).real, 1.0)
            assert np.linalg.eigvalsh(rdms[sample, index]).min() > -1e-12


def test_layer_layout_is_brickwall_on_ring():
    arch = ReservoirArchitecture(
        8, 8, encoding_period=4, input_shift=4, angle_scale=1.0
    )
    assert arch.layer_pairs(0) == [(0, 1), (2, 3), (4, 5), (6, 7)]
    assert arch.layer_pairs(1) == [(1, 2), (3, 4), (5, 6), (7, 0)]
    shifted = ReservoirArchitecture(8, 8, 4, 4, 1.0, brickwall_offset=1)
    assert shifted.layer_pairs(0) == arch.layer_pairs(1)
    assert [arch.is_encoding_layer(layer) for layer in range(8)] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    np.testing.assert_array_equal(arch.input_indices(1, 4), [0, 1, 2, 3])
    np.testing.assert_array_equal(arch.input_indices(1, 72), [4, 5, 6, 7])


def test_sampled_h_respects_lower_bound():
    hyper = ReservoirHyperparameters(0.2, 0.7, 0.03, 0.12, 0.2, 0.2, 0.03, 0.1)
    arch = ReservoirArchitecture(
        8, 40, encoding_period=4, input_shift=4, angle_scale=1.0
    )
    params = sample_block_parameters(hyper, arch, np.random.default_rng(0))
    assert params.h.min() > 0.1


def test_common_random_numbers_share_draws_across_settings():
    arch = ReservoirArchitecture(
        8, 4, encoding_period=4, input_shift=4, angle_scale=1.0
    )
    other = ReservoirHyperparameters(0.5, 1.0, 0.1, 0.9, 0.05, 0.4, 0.07, 0.1)
    first = sample_block_parameters(TABLE_I, arch, np.random.default_rng(9))
    second = sample_block_parameters(other, arch, np.random.default_rng(9))
    np.testing.assert_allclose((first.J - 0.237) / 0.038, (second.J - 0.4) / 0.07)
