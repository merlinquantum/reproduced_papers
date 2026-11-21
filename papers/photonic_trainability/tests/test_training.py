from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the package on sys.path)
from photonic_trainability.lib import clements, correlators, photonic_sim, training
from photonic_trainability.lib.observables import (
    QuadraticNetworkObservable,
    evaluate_polynomial,
    make_monomial,
)


@pytest.mark.parametrize("order", [1, 2, 4, 8])
def test_gpsr_coefficient_identity(order):
    _, coefficients = training.gpsr_shifts_and_coefficients(order)
    assert np.isclose(np.sum(coefficients**2), (2 * order**2 + 1) / 6)


def test_gpsr_is_exact_for_trigonometric_polynomials():
    rng = np.random.default_rng(0)
    order = 3
    amplitudes = rng.normal(size=(order + 1, 2))

    def loss(batch):
        theta = batch[:, 0]
        return sum(
            a * np.cos(k * theta) + b * np.sin(k * theta)
            for k, (a, b) in enumerate(amplitudes)
        )

    point = np.array([0.37])
    exact = sum(
        k * (-a * np.sin(k * 0.37) + b * np.cos(k * 0.37))
        for k, (a, b) in enumerate(amplitudes)
    )
    assert np.isclose(training.gpsr_gradient(loss, point, order)[0], exact)


def _mesh_expectation(modes, network_layers, seed):
    rng = np.random.default_rng(seed)
    mesh = clements.decompose(photonic_sim.haar_unitary(modes, rng))
    network = QuadraticNetworkObservable.random(modes, network_layers, rng)
    occupation = tuple(1 if i % 2 == 0 else 0 for i in range(modes))
    expectation = training.PhotonicExpectation(
        clements.perceval_circuit(mesh), occupation, network
    )
    return mesh, network, occupation, expectation


def test_gpsr_matches_autograd_and_finite_differences_on_mesh():
    mesh, network, _, expectation = _mesh_expectation(4, 2, seed=1)
    assert network.polynomial_order == 4
    parameters = mesh.parameters()
    order = network.polynomial_order
    gpsr = training.gpsr_gradient(expectation.exact_losses, parameters, order)
    autograd = expectation.autograd_gradient(parameters)
    step = 1e-5
    eye = np.eye(len(parameters))
    finite = (
        expectation.exact_losses(parameters + step * eye)
        - expectation.exact_losses(parameters - step * eye)
    ) / (2 * step)
    np.testing.assert_allclose(gpsr, autograd, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(gpsr, finite, rtol=1e-5, atol=1e-7)


def test_mesh_probabilities_match_unitary_distribution():
    mesh, network, occupation, expectation = _mesh_expectation(5, 1, seed=2)
    probabilities = expectation.probabilities(mesh.parameters()[None, :])[0]
    _, reference = photonic_sim.output_distribution(
        mesh.unitary(), occupation, expectation.patterns
    )
    np.testing.assert_allclose(probabilities, reference, atol=1e-12)


def test_shot_estimates_are_unbiased():
    mesh, _, _, expectation = _mesh_expectation(4, 1, seed=3)
    batch = mesh.parameters()[None, :]
    exact = expectation.exact_losses(batch)[0]
    variance = expectation.shot_variances(batch)[0]
    rng = np.random.default_rng(4)
    estimates = np.array(
        [expectation.shot_losses(batch, 200, rng)[0] for _ in range(400)]
    )
    assert abs(estimates.mean() - exact) < 5 * np.sqrt(variance / 200 / 400)
    assert np.isclose(estimates.var(), variance / 200, rtol=0.25)


def test_loop_circuit_matches_unitary():
    modes = 5
    blocks = clements.loop_circuit_blocks(modes, 2)
    angles = np.random.default_rng(5).uniform(0, np.pi, len(blocks))
    circuit = clements.perceval_loop_circuit(modes, blocks)
    for index, angle in enumerate(angles):
        circuit.param(clements.parameter_name(index)).set_value(2 * angle)
    perceval_matrix = np.array(circuit.compute_unitary())
    np.testing.assert_allclose(
        perceval_matrix, clements.loop_unitary(modes, blocks, angles).T, atol=1e-12
    )


def test_correlator_expectations_match_distribution():
    rng = np.random.default_rng(6)
    unitary = photonic_sim.haar_unitary(5, rng)
    occupation = (1, 1, 0, 1, 0)
    polynomial = {
        make_monomial({0: 1}): 0.4,
        make_monomial({1: 2}): -1.1,
        make_monomial({2: 1, 4: 1}): 0.8,
        make_monomial({3: 2, 0: 1}): 0.3,
    }
    patterns, probabilities = photonic_sim.output_distribution(unitary, occupation)
    reference = probabilities @ evaluate_polynomial(polynomial, patterns)
    assert np.isclose(
        correlators.polynomial_expectation(unitary, occupation, polynomial), reference
    )


def test_learning_rate_selection_ignores_diverged_candidates_and_breaks_ties():
    from photonic_trainability.lib.exp_training import select_learning_rate

    rows = [
        {"learning_rate": 0.3, "final_exact_loss": float("nan")},
        {"learning_rate": 0.1, "final_exact_loss": -1.0},
        {"learning_rate": 0.03, "final_exact_loss": -1.0 + 1e-13},
        {"learning_rate": 0.01, "final_exact_loss": -0.5},
    ]
    assert select_learning_rate(rows) == 0.03
