from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the repo on sys.path)
from qiskit.quantum_info import Statevector
from QNN_hyperparameters.lib.circuits import build_ansatz, build_feature_map, paper_grid
from QNN_hyperparameters.lib.simulator import CompiledCircuit, class_probabilities


@pytest.mark.parametrize("grid_index", [0, 4, 8, 24, 29, 31, 40, 53, 57, 62, 78, 83])
def test_statevector_matches_qiskit(grid_index):
    circuit = paper_grid()[grid_index]
    rng = np.random.default_rng(grid_index)
    feature_map = build_feature_map(
        circuit["feature_map"], 7, circuit["feature_map_entanglement"]
    )
    ansatz = build_ansatz(circuit["ansatz"], 7, circuit["ansatz_entanglement"])
    compiled = CompiledCircuit(feature_map, ansatz)
    X = rng.uniform(-4, 4, (2, 7))
    weights = rng.uniform(-3, 3, compiled.num_weights)
    ours = compiled.probabilities(X, weights)
    reference_circuit = feature_map.compose(ansatz)
    for row in range(2):
        binding = dict(zip(feature_map.parameters, X[row]))
        binding.update(zip(ansatz.parameters, weights))
        reference = Statevector(
            reference_circuit.assign_parameters(binding)
        ).probabilities()
        np.testing.assert_allclose(ours[row], reference, atol=1e-10)


def test_lda_qubit_counts_compile():
    compiled = CompiledCircuit(
        build_feature_map("ZZFeatureMap", 5, "sca"),
        build_ansatz("TwoLocal", 5, "pairwise"),
    )
    probabilities = compiled.probabilities(
        np.zeros((3, 5)), np.zeros(compiled.num_weights)
    )
    assert probabilities.shape == (3, 32)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_class_readout_is_integer_modulo_classes():
    compiled = CompiledCircuit(
        build_feature_map("ZFeatureMap", 3, None),
        build_ansatz("RealAmplitudes", 3, "linear"),
    )
    X = np.array([[0.3, -1.0, 2.0]])
    weights = np.linspace(0, 1, compiled.num_weights)
    exact = compiled.probabilities(X, weights)[0]
    classes = class_probabilities(
        compiled, X, weights, 3, None, np.random.default_rng(0)
    )[0]
    expected = [exact[np.arange(8) % 3 == k].sum() for k in range(3)]
    np.testing.assert_allclose(classes, expected, atol=1e-12)


def test_shot_sampling_is_unbiased():
    compiled = CompiledCircuit(
        build_feature_map("ZFeatureMap", 2, None),
        build_ansatz("RealAmplitudes", 2, "linear"),
    )
    X = np.tile([[0.4, 1.2]], (4000, 1))
    weights = np.full(compiled.num_weights, 0.7)
    exact = class_probabilities(
        compiled, X[:1], weights, 2, None, np.random.default_rng(0)
    )[0]
    sampled = class_probabilities(
        compiled, X, weights, 2, 1024, np.random.default_rng(1)
    )
    assert np.all(sampled * 1024 == np.round(sampled * 1024))
    np.testing.assert_allclose(sampled.mean(axis=0), exact, atol=2e-3)
