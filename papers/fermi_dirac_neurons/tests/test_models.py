"""Gradient correctness and model structure (Daleckii-Krein vs finite differences)."""

from __future__ import annotations

import common  # noqa: F401  (sets sys.path)
import numpy as np
import pytest
from fermi_dirac_neurons.lib.data import haar_pure_states
from fermi_dirac_neurons.lib.models import (
    Dataset,
    HamiltonianNeuron,
    PauliFeatureNeuron,
)
from fermi_dirac_neurons.lib.pauli import build_terms, is_commuting

ACTIVATIONS = ["tanh", "softplus", "softplus_code", "silu", "linear", "exp"]


def _finite_difference(model, theta, data, loss, step=1e-6):
    gradient = np.zeros_like(theta)
    for j in range(len(theta)):
        shift = np.zeros_like(theta)
        shift[j] = step
        plus, _ = model.loss_and_gradient(theta + shift, data, loss)
        minus, _ = model.loss_and_gradient(theta - shift, data, loss)
        gradient[j] = (plus - minus) / (2 * step)
    return gradient


def _regression_data(rng, n_qubits=2, n_states=7):
    states = haar_pure_states(n_qubits, n_states, rng)
    return Dataset(states, rng.uniform(-1.0, 1.0, size=n_states))


def _classification_data(rng, n_qubits=2, n_states=9):
    states = haar_pure_states(n_qubits, n_states, rng)
    return Dataset(states, rng.choice([-1.0, 1.0], size=n_states))


@pytest.mark.parametrize("model_class", [HamiltonianNeuron, PauliFeatureNeuron])
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize(
    "term_set,identity", [("tfim", True), ("heisenberg", False), ("ising", True)]
)
def test_squared_loss_gradient_matches_finite_differences(
    model_class, activation, term_set, identity
):
    rng = np.random.default_rng(1)
    model = model_class(term_set, 2, identity, activation, 2.0)
    theta = rng.uniform(-1.0, 1.0, size=model.n_parameters)
    data = _regression_data(rng)
    _, gradient = model.loss_and_gradient(theta, data, "squared")
    np.testing.assert_allclose(
        gradient, _finite_difference(model, theta, data, "squared"), atol=1e-7
    )


@pytest.mark.parametrize("model_class", [HamiltonianNeuron, PauliFeatureNeuron])
@pytest.mark.parametrize("term_set", ["heisenberg", "fcim"])
def test_logistic_loss_gradient_matches_finite_differences(model_class, term_set):
    rng = np.random.default_rng(2)
    model = model_class(term_set, 3, False, "tanh", 2.0)
    theta = rng.uniform(-1.0, 1.0, size=model.n_parameters)
    data = _classification_data(rng, n_qubits=3)
    _, gradient = model.loss_and_gradient(theta, data, "logistic")
    np.testing.assert_allclose(
        gradient, _finite_difference(model, theta, data, "logistic"), atol=1e-7
    )


def test_gradient_is_finite_on_degenerate_spectrum():
    # theta = 0 except the identity term: every eigenvalue is equal.
    model = HamiltonianNeuron("tfim", 2, True, "tanh", 2.0)
    theta = np.zeros(model.n_parameters)
    theta[-1] = 0.3
    data = _regression_data(np.random.default_rng(3))
    _, gradient = model.loss_and_gradient(theta, data, "squared")
    np.testing.assert_allclose(
        gradient, _finite_difference(model, theta, data, "squared"), atol=1e-7
    )


def test_logistic_loss_matches_direct_matrix_logarithm():
    rng = np.random.default_rng(4)
    model = HamiltonianNeuron("heisenberg", 2, False, "tanh", 2.0)
    theta = rng.uniform(-1.0, 1.0, size=model.n_parameters)
    data = _classification_data(rng)
    value, _ = model.loss_and_gradient(theta, data, "logistic")
    hamiltonian = np.einsum("j,jab->ab", theta, model.terms)
    direct = 0.0
    for rho, y in zip(data.states, data.targets):
        values, vectors = np.linalg.eigh(-y * hamiltonian / 2.0)
        log_term = (vectors * np.log1p(np.exp(values))) @ vectors.conj().T
        direct += 2.0 * np.real(np.trace(log_term @ rho))
    assert value == pytest.approx(direct / len(data.targets), abs=1e-12)


def test_commuting_flags_and_classical_equivalence():
    assert HamiltonianNeuron("ising", 3, True, "tanh", 2.0).commuting
    assert HamiltonianNeuron("fcim", 3, False, "tanh", 2.0).commuting
    assert not HamiltonianNeuron("tfim", 3, True, "tanh", 2.0).commuting
    assert not HamiltonianNeuron("heisenberg", 3, False, "tanh", 2.0).commuting


def test_linear_pauli_features_equal_quantum_linear_model():
    rng = np.random.default_rng(5)
    quantum = HamiltonianNeuron("tfim", 3, True, "linear", 2.0)
    features = PauliFeatureNeuron("tfim", 3, True, "linear", 2.0)
    theta = rng.uniform(-1.0, 1.0, size=quantum.n_parameters)
    data = _regression_data(rng, n_qubits=3)
    np.testing.assert_allclose(
        quantum.outputs(theta, data), features.outputs(theta, data), atol=1e-12
    )


def test_prediction_rules_return_signs():
    rng = np.random.default_rng(6)
    model = HamiltonianNeuron("heisenberg", 2, False, "tanh", 2.0)
    theta = rng.uniform(-1.0, 1.0, size=model.n_parameters)
    data = _classification_data(rng)
    for rule in ("activation", "linear"):
        assert set(model.classify(theta, data, rule)) <= {-1.0, 1.0}
    with pytest.raises(ValueError, match="rule"):
        model.classify(theta, data, "nope")


@pytest.mark.parametrize(
    "term_set,identity,expected",
    [
        ("tfim", True, lambda n: 2 * n),
        ("ising", True, lambda n: 2 * n),
        ("heisenberg", False, lambda n: 6 * n - 3),
        ("fcim", False, lambda n: n * (n + 1) // 2),
    ],
)
@pytest.mark.parametrize("n_qubits", [2, 3, 5])
def test_term_counts_match_paper(term_set, identity, expected, n_qubits):
    labels, terms = build_terms(term_set, n_qubits, identity)
    assert len(labels) == expected(n_qubits) == len(set(labels))
    assert terms.shape == (expected(n_qubits), 2**n_qubits, 2**n_qubits)
    np.testing.assert_allclose(terms, terms.conj().transpose(0, 2, 1))
    assert is_commuting(terms) == (term_set in ("ising", "fcim"))


@pytest.mark.parametrize(
    "term_set, include_identity", [("heisenberg", False), ("tfim", True)]
)
def test_trace_gradient_matches_per_term_daleckii_krein(term_set, include_identity):
    from fermi_dirac_neurons.lib.activations import divided_differences

    model = HamiltonianNeuron(term_set, 3, include_identity, "tanh", 2.0)
    rng = np.random.default_rng(7)
    theta = rng.uniform(-1.0, 1.0, model.n_parameters)
    matrix = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    weighted_state = matrix + matrix.conj().T
    eigenvalues, eigenvectors = model._spectrum(theta)
    divided = divided_differences(model.activation, eigenvalues)
    rotated_state = eigenvectors.conj().T @ weighted_state @ eigenvectors
    expected = [
        np.real(
            np.sum(
                divided
                * (eigenvectors.conj().T @ term @ eigenvectors)
                * rotated_state.T
            )
        )
        for term in model.terms
    ]
    gradient = model._trace_gradient(
        model.activation, eigenvalues, eigenvectors, weighted_state
    )
    np.testing.assert_allclose(gradient, expected, atol=1e-12)


def test_finite_shot_gradient_is_unbiased():
    from fermi_dirac_neurons.lib.data import build_states

    model = HamiltonianNeuron("tfim", 2, True, "tanh", 2.0)
    rng = np.random.default_rng(3)
    theta = rng.uniform(-0.5, 0.5, model.n_parameters)
    states = build_states({"set": "code_squared_loss"}, 2, rng)
    data = Dataset(states, rng.uniform(-0.5, 0.5, len(states)))
    exact_loss, exact_gradient = model.loss_and_gradient(theta, data, "squared")
    repetitions = 4000
    estimates = np.array(
        [
            model.finite_shot_loss_and_gradient(theta, data, 50, rng)[1]
            for _ in range(repetitions)
        ]
    )
    loss, _ = model.finite_shot_loss_and_gradient(theta, data, 50, rng)
    assert loss == pytest.approx(exact_loss, rel=1e-12)
    standard_error = estimates.std(axis=0, ddof=1) / np.sqrt(repetitions)
    assert np.all(np.abs(estimates.mean(axis=0) - exact_gradient) < 4 * standard_error)


def test_finite_shot_gradient_requires_tanh():
    model = HamiltonianNeuron("tfim", 2, True, "softplus", 2.0)
    with pytest.raises(ValueError, match="tanh"):
        model.finite_shot_loss_and_gradient(
            np.zeros(4),
            Dataset(np.eye(4)[None] / 4, np.zeros(1)),
            10,
            np.random.default_rng(0),
        )


def test_explicit_labels_build_the_given_terms():
    from fermi_dirac_neurons.lib.models import build_model

    model = build_model(
        {
            "name": "m",
            "kind": "hamiltonian",
            "term_set": "explicit",
            "term_labels": ["XZ", "ZI", "YY"],
            "include_identity": False,
            "activation": "tanh",
        },
        2,
        2.0,
    )
    assert model.labels == ["XZ", "ZI", "YY"]
    with pytest.raises(ValueError, match="distinct"):
        HamiltonianNeuron(
            "explicit", 2, False, "tanh", 2.0, explicit_labels=["XZ", "XZ"]
        )
    with pytest.raises(ValueError, match="exactly when"):
        HamiltonianNeuron("tfim", 2, False, "tanh", 2.0, explicit_labels=["XZ"])
