from __future__ import annotations

import numpy as np
from common import PROJECT_DIR  # noqa: F401  (puts the repository on sys.path)
from QuaRK.lib.reservoir import (
    batched_ry_layer,
    contraction_channel,
    ising_unitary,
    kron_qubits,
)


def _random_density_matrix(dim, rng):
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def _ry(angle):
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -s], [s, c]])


def _swap_dilation(rho_reservoir, contraction):
    """The authors' Fig. 2 circuit on n=2: |+> ancillas, Ry coin, controlled-SWAPs, trace out."""
    n = 2
    total = 2 * n + 1  # qubits 0..1 reservoir, 2..3 ancillas, 4 coin
    zero = np.diag([1.0, 0.0])
    env = kron_qubits([zero] * (n + 1))
    state = np.kron(env, rho_reservoir)  # environment qubits are the high bits
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    theta = 2 * np.arcsin(np.sqrt(1 - contraction))
    prep = kron_qubits([np.eye(2)] * n + [hadamard] * n + [_ry(theta)])
    state = prep @ state @ prep.conj().T
    dim = 2**total
    permutation = np.arange(dim)
    for index in range(dim):
        bits = [(index >> q) & 1 for q in range(total)]
        if bits[4]:
            for q in range(n):
                bits[q], bits[n + q] = bits[n + q], bits[q]
        permutation[index] = sum(b << q for q, b in enumerate(bits))
    cswap = np.zeros((dim, dim))
    cswap[permutation, np.arange(dim)] = 1.0
    state = cswap @ state @ cswap.T
    reshaped = state.reshape(2 ** (n + 1), 2**n, 2 ** (n + 1), 2**n)
    return np.einsum("aibj,ab->ij", reshaped, np.eye(2 ** (n + 1)))


def test_closed_form_channel_equals_swap_dilation():
    rng = np.random.default_rng(0)
    rho = _random_density_matrix(4, rng)
    for contraction in (0.1, 0.5, 0.9):
        expected = _swap_dilation(rho, contraction)
        np.testing.assert_allclose(
            contraction_channel(rho, contraction), expected, atol=1e-12
        )


def test_channel_is_contractive_in_hilbert_schmidt_norm():
    rng = np.random.default_rng(1)
    rho, sigma = _random_density_matrix(8, rng), _random_density_matrix(8, rng)
    for contraction in (0.1, 0.34, 0.9):
        before = np.linalg.norm(rho - sigma)
        after = np.linalg.norm(
            contraction_channel(rho, contraction)
            - contraction_channel(sigma, contraction)
        )
        assert np.isclose(after, contraction * before)


def test_ising_unitary_is_unitary_and_ry_layer_follows_qiskit_order():
    rng = np.random.default_rng(2)
    unitary = ising_unitary(
        3, rng.uniform(-3, 3, 3), rng.uniform(-3, 3, 3), rng.uniform(-3, 3, 3)
    )
    np.testing.assert_allclose(unitary @ unitary.conj().T, np.eye(8), atol=1e-12)
    angles = rng.uniform(-3, 3, size=(2, 3))
    layer = batched_ry_layer(angles)
    expected = kron_qubits([_ry(a) for a in angles[1]])
    np.testing.assert_allclose(layer[1], expected, atol=1e-12)
    # Qubit 0 is the least significant bit: Ry on qubit 0 alone flips index 0 <-> 1.
    single = batched_ry_layer(np.array([[np.pi, 0.0, 0.0]]))[0]
    assert np.isclose(abs(single[1, 0]), 1.0)
