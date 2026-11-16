from __future__ import annotations

import common  # noqa: F401
import numpy as np
from distributed_secure_kernels.lib.noise import (
    depolarizing_noise_model,
    noisy_zero_probability,
    swap_test_observable,
    teleport_transfer_tensor,
)
from distributed_secure_kernels.lib.protocol import (
    distributed_circuit,
    run_ancilla_zeros,
)
from qiskit import transpile
from qiskit_aer import AerSimulator


def _states(seed, count, n):
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(count, 2**n))
    return states / np.linalg.norm(states, axis=1, keepdims=True)


def test_noiseless_teleport_channel_is_hadamard_conjugation():
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    expected = np.einsum("ai,bj->abij", hadamard, hadamard.conj())
    np.testing.assert_allclose(teleport_transfer_tensor(0.0), expected, atol=1e-12)


def test_noiseless_observable_gives_the_ideal_swap_test():
    states = _states(0, 4, 2)
    np.testing.assert_allclose(
        noisy_zero_probability(states, states, 0.0),
        0.5 + 0.5 * (states @ states.T) ** 2,
        atol=1e-12,
    )
    observable = swap_test_observable(2, 0.0)
    np.testing.assert_allclose(observable, observable.conj().T, atol=1e-12)


def test_noisy_probability_matches_per_shot_aer_on_the_full_circuit():
    # Identical states: p0 = 1 without noise, so noise is clearly visible.
    states = _states(1, 1, 2)
    rate, shots = 0.05, 20_000
    predicted = noisy_zero_probability(states, states, rate)[0, 0]
    zeros = run_ancilla_zeros(
        distributed_circuit(states[0], states[0]), shots, rate, seed=3
    )
    standard_error = np.sqrt(predicted * (1 - predicted) / shots)
    assert predicted < 0.95
    assert abs(zeros / shots - predicted) < 4 * standard_error


def test_transpiled_noisy_circuit_uses_only_modelled_gates():
    states = _states(2, 2, 2)
    simulator = AerSimulator(noise_model=depolarizing_noise_model(0.01))
    names = set(
        transpile(distributed_circuit(states[0], states[1]), simulator).count_ops()
    )
    assert names <= {
        "u1",
        "u2",
        "u3",
        "cx",
        "measure",
        "reset",
        "if_else",
        "initialize",
    }
    assert "initialize" in names  # state preparation stays native and noiseless


def test_noisy_initialize_variant_matches_aer_with_decomposed_initialize():
    states = _states(4, 2, 2)
    rate, shots = 0.05, 20_000
    predicted = noisy_zero_probability(
        states[:1], states[1:], rate, noisy_initialize=True
    )[0, 0]
    simulator = AerSimulator(
        noise_model=depolarizing_noise_model(rate), seed_simulator=11
    )
    basis = ["u1", "u2", "u3", "cx", "measure", "reset", "if_else"]
    compiled = transpile(distributed_circuit(states[0], states[1]), basis_gates=basis)
    counts = simulator.run(compiled, shots=shots).result().get_counts()
    zeros = shots - sum(v for k, v in counts.items() if k.replace(" ", "")[0] == "1")
    assert abs(zeros / shots - predicted) < 4 * np.sqrt(
        predicted * (1 - predicted) / shots
    )
