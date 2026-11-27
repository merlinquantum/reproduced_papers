"""Simulator and embedding checks against dense matrices built from Paulis."""

from __future__ import annotations

import common  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import pytest
import torch
from lib import embeddings, simulator
from scipy.linalg import expm

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def op_on(n_qubits, placements):
    """Kronecker product with ``placements = {qubit: 2x2 matrix}``; qubit 0 leftmost."""
    matrix = np.array([[1.0 + 0j]])
    for qubit in range(n_qubits):
        matrix = np.kron(matrix, placements.get(qubit, I2))
    return matrix


def flat(states: torch.Tensor) -> np.ndarray:
    return states.reshape(states.shape[0], -1).numpy()


def random_states(batch, n_qubits, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((batch, 2**n_qubits)) + 1j * rng.standard_normal(
        (batch, 2**n_qubits)
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return torch.as_tensor(vectors.reshape((batch,) + (2,) * n_qubits))


CONVENTIONS = embeddings.EmbeddingConventions(
    e2_pair_coefficient=2.0,
    e2_phase_sign=1.0,
    e3_register="n_plus_one",
    e3_time="n_over_3",
    e3_trotter_steps=20,
    qnn_trotter_steps=10,
    qnn_total_time=10.0,
    qnn_boundary="open",
    qnn_observable_qubit=0,
    input_scale=1.0,
)


def test_heisenberg_bond_matches_expm():
    n_qubits, batch = 4, 3
    states = random_states(batch, n_qubits)
    thetas = torch.tensor([0.3, -1.1, 2.4], dtype=torch.float64)
    for qubit_a, qubit_b in [(0, 1), (2, 3), (1, 3)]:
        evolved = flat(
            simulator.apply_heisenberg_bond(states, qubit_a, qubit_b, thetas)
        )
        heisenberg = sum(op_on(n_qubits, {qubit_a: P, qubit_b: P}) for P in (X, Y, Z))
        for index, theta in enumerate(thetas.numpy()):
            expected = expm(-1j * theta * heisenberg) @ flat(states)[index]
            np.testing.assert_allclose(evolved[index], expected, atol=1e-12)


def test_hadamard_layer_and_product_state():
    n_qubits = 3
    states = random_states(2, n_qubits, seed=1)
    full_h = op_on(n_qubits, dict.fromkeys(range(n_qubits), H))
    np.testing.assert_allclose(
        flat(simulator.apply_hadamard_layer(states)),
        flat(states) @ full_h.T,
        atol=1e-12,
    )
    singles = torch.as_tensor(
        np.random.default_rng(2).standard_normal((2, n_qubits, 2)) + 0j
    )
    expected = [np.kron(np.kron(s[0], s[1]), s[2]) for s in singles.numpy()]
    np.testing.assert_allclose(
        flat(simulator.product_state(singles)), np.array(expected), atol=1e-12
    )


def test_pauli_expectations_match_dense():
    n_qubits = 3
    states = random_states(4, n_qubits, seed=3)
    expectations = simulator.single_qubit_pauli_expectations(states).numpy()
    for qubit in range(n_qubits):
        for column, pauli in enumerate((X, Y, Z)):
            operator = op_on(n_qubits, {qubit: pauli})
            expected = np.einsum(
                "bi,ij,bj->b", flat(states).conj(), operator, flat(states)
            ).real
            np.testing.assert_allclose(
                expectations[:, qubit, column], expected, atol=1e-12
            )
    np.testing.assert_allclose(
        simulator.z_expectation(states, 1).numpy(), expectations[:, 1, 2], atol=1e-12
    )


def _features(batch, n, seed=4):
    return torch.as_tensor(np.random.default_rng(seed).standard_normal((batch, n)))


def test_e1_matches_rx_rotations_and_closed_forms():
    features = _features(3, 3)
    states = flat(
        embeddings.embed(
            "E1", features, CONVENTIONS, haar_product_state=None, dtype=torch.complex128
        )
    )
    for index, x in enumerate(features.numpy()):
        unitary = op_on(3, {j: expm(-1j * x[j] * X) for j in range(3)})
        np.testing.assert_allclose(states[index], unitary[:, 0], atol=1e-12)
    gram = np.abs(states.conj() @ states.T) ** 2
    np.testing.assert_allclose(
        embeddings.e1_quantum_kernel(features.numpy()), gram, atol=1e-12
    )
    expectations = simulator.single_qubit_pauli_expectations(
        torch.as_tensor(states.reshape(3, 2, 2, 2))
    )
    np.testing.assert_allclose(
        embeddings.e1_pauli_features(features.numpy()), expectations.numpy(), atol=1e-12
    )


@pytest.mark.parametrize("pair_coefficient", [1.0, 2.0])
def test_e2_matches_dense_iqp_circuit(pair_coefficient):
    n = 3
    conventions = embeddings.EmbeddingConventions(
        **{**CONVENTIONS.__dict__, "e2_pair_coefficient": pair_coefficient}
    )
    features = _features(2, n, seed=5)
    states = flat(
        embeddings.embed(
            "E2", features, conventions, haar_product_state=None, dtype=torch.complex128
        )
    )
    full_h = op_on(n, dict.fromkeys(range(n), H))
    zero = np.zeros(2**n, dtype=complex)
    zero[0] = 1.0
    for index, x in enumerate(features.numpy()):
        # Paper App. J E2 with the imaginary unit restored (AMB-1); the double sum over
        # j != j' counts each unordered pair twice, i.e. pair_coefficient = 2.
        generator = sum(x[j] * op_on(n, {j: Z}) for j in range(n))
        generator = generator + pair_coefficient * sum(
            x[j] * x[k] * op_on(n, {j: Z, k: Z})
            for j in range(n)
            for k in range(j + 1, n)
        )
        u_z = expm(1j * generator)
        expected = u_z @ full_h @ u_z @ full_h @ zero
        np.testing.assert_allclose(states[index], expected, atol=1e-10)


def test_e3_matches_dense_trotter_circuit():
    n = 3
    n_qubits = n + 1
    features = _features(2, n, seed=6)
    haar = embeddings.sample_haar_product_state(n_qubits, np.random.default_rng(7))
    states = flat(
        embeddings.embed(
            "E3", features, CONVENTIONS, haar_product_state=haar, dtype=torch.complex128
        )
    )
    initial = haar[0]
    for qubit in range(1, n_qubits):
        initial = np.kron(initial, haar[qubit])
    steps, total_time = 20, n / 3.0
    for index, x in enumerate(features.numpy()):
        step = np.eye(2**n_qubits, dtype=complex)
        for bond in range(n):
            heisenberg = sum(op_on(n_qubits, {bond: P, bond + 1: P}) for P in (X, Y, Z))
            step = expm(-1j * (total_time / steps) * x[bond] * heisenberg) @ step
        expected = np.linalg.matrix_power(step, steps) @ initial
        np.testing.assert_allclose(states[index], expected, atol=1e-10)


def test_qnn_labels_match_dense_evolution():
    n = 3
    features = _features(2, n, seed=8)
    states = embeddings.embed(
        "E1", features, CONVENTIONS, haar_product_state=None, dtype=torch.complex128
    )
    couplings = np.random.default_rng(9).standard_normal(
        embeddings.qnn_coupling_count(n, CONVENTIONS)
    )
    labels = simulator.z_expectation(
        embeddings.apply_qnn(states, couplings, CONVENTIONS), 0
    ).numpy()
    step = np.eye(2**n, dtype=complex)
    for bond, coupling in enumerate(couplings):
        heisenberg = sum(op_on(n, {bond: P, bond + 1: P}) for P in (X, Y, Z))
        step = expm(-1j * 1.0 * coupling * heisenberg) @ step  # t/T = 10/10
    unitary = np.linalg.matrix_power(step, 10)
    z_first = op_on(n, {0: Z})
    for index, psi in enumerate(flat(states)):
        evolved = unitary @ psi
        np.testing.assert_allclose(
            labels[index], (evolved.conj() @ z_first @ evolved).real, atol=1e-10
        )


def test_quantum_bank_kernel_and_features_consistent():
    features = np.random.default_rng(10).standard_normal((12, 3))
    for embedding in embeddings.EMBEDDINGS:
        n_qubits = embeddings.register_size(embedding, 3, CONVENTIONS)
        haar = (
            embeddings.sample_haar_product_state(n_qubits, np.random.default_rng(11))
            if embedding == "E3"
            else None
        )
        bank = embeddings.compute_quantum_bank(
            embedding,
            features,
            CONVENTIONS,
            haar_product_state=haar,
            qnn_couplings=None,
            need_quantum_kernel=True,
            device="cpu",
            complex_dtype=torch.complex128,
            batch_memory_mb=0.001,
            e1_analytic=False,
        )
        states = flat(
            embeddings.embed(
                embedding,
                torch.as_tensor(features),
                CONVENTIONS,
                haar_product_state=haar,
                dtype=torch.complex128,
            )
        )
        np.testing.assert_allclose(
            bank.quantum_kernel, np.abs(states.conj() @ states.T) ** 2, atol=1e-12
        )
        assert bank.pauli_features.shape == (12, n_qubits, 3)
        assert np.allclose(np.diag(bank.quantum_kernel), 1.0)
