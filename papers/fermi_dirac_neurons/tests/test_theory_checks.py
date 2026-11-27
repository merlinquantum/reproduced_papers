"""Paper identities that must hold, and the printed formulas that must not (D14, D16, D19)."""

from __future__ import annotations

import common  # noqa: F401  (sets sys.path)
import numpy as np
import pytest
from fermi_dirac_neurons.lib import theory_checks as tc

TEMPERATURE = 2.0


@pytest.fixture(scope="module")
def problem():
    return tc.random_problem(2, "tfim", True, np.random.default_rng(0))


@pytest.mark.parametrize("name", ["mu", "gamma", "xi"])
def test_kernels_are_normalized(name):
    assert tc.density_normalization(name) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("activation", ["tanh", "softplus", "silu"])
def test_commuting_hamiltonian_reduces_to_classical_neuron(activation):
    record = tc.check_classical_reduction(
        3, activation, TEMPERATURE, np.random.default_rng(1)
    )
    assert record["max_abs_error"] < 1e-12


def test_theorems_1_2_5_6_and_corollary_1_hold(problem):
    theta, terms, rho = problem
    records = [
        tc.check_theorem1(theta, terms, rho, TEMPERATURE),
        tc.check_theorem2(theta, terms, rho, TEMPERATURE),
        tc.check_theorem5(theta, terms, rho, TEMPERATURE, 1.0),
        tc.check_theorem5(theta, terms, rho, TEMPERATURE, -1.0),
        tc.check_theorem6(theta, terms, rho, TEMPERATURE),
        tc.check_corollary1(theta, terms),
    ]
    for record in records:
        assert record["max_abs_error"] < 1e-9


def test_theorem8_as_printed_does_not_match_exact_gradient(problem):
    theta, terms, rho = problem
    assert (
        tc.check_theorem8_as_printed(theta, terms, rho, TEMPERATURE)["max_abs_error"]
        > 1e-2
    )


def test_theorem13_printed_fails_and_corrected_holds(problem):
    theta, terms, rho = problem
    records = tc.check_theorem13(theta, terms, rho, TEMPERATURE, 1.0)
    assert records["printed"]["max_abs_error"] > 1e-2
    assert records["corrected"]["max_abs_error"] < 1e-9


def test_convolution_identities_temperature_conventions():
    records = tc.check_convolution_identities(0.7, 1.3, np.array([-2.0, 0.0, 1.0, 3.0]))
    assert records["eq41_tanh_T=2*T1*T2"]["max_abs_error"] < 1e-9
    assert records["eq90_erf_T=2*T1*T2"]["max_abs_error"] < 1e-9
    assert records["eq73_softplus_T=T1*T2"]["max_abs_error"] < 1e-9
    assert records["eq94_grelu_T=T1*T2"]["max_abs_error"] < 1e-9
    assert records["eq41_tanh_T=T1*T2"]["max_abs_error"] > 1e-2


def test_single_shot_algorithms_are_unbiased(problem):
    theta, terms, rho = problem
    hamiltonian = np.einsum("j,jab->ab", theta, terms)
    results = tc.simulate_single_shot_neurons(
        hamiltonian, rho, 0.7, 1.3, 100000, np.random.default_rng(2)
    )
    for record in results.values():
        assert record["abs_error"] < 5 * record["standard_error"]


def test_algorithm1_estimate_is_unbiased(problem):
    theta, terms, rho = problem
    sampler = tc.TabulatedDensity("mu")
    samples = tc.simulate_algorithm1(
        theta, terms, rho, TEMPERATURE, 0, 4000, np.random.default_rng(3), sampler
    )
    exact = tc.exact_trace_gradient("tanh", TEMPERATURE, theta, terms, rho)[0]
    assert abs(samples.mean() - exact) < 5 * samples.std(ddof=1) / np.sqrt(len(samples))
