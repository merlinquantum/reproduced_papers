"""Baseline B1 (Lim-Oh estimator) and the C9 comparison experiment."""

from __future__ import annotations

import json

import numpy as np
import pytest
from common import PROJECT_DIR, load_runtime_ready_config
from photonic_trainability.lib import lim_oh, photonic_sim, runner
from photonic_trainability.lib.observables import evaluate_monomial, make_monomial


@pytest.mark.parametrize("exponent, photons", [(1, 2), (2, 3), (2, 8)])
def test_radial_density_is_normalised_by_laguerre_orthonormality(exponent, photons):
    sampler = lim_oh.DiagonalModeSampler([k**exponent for k in range(photons + 1)])
    # Trapezoid grid: discretisation bias ~2e-6 relative, far below sampling error.
    assert np.isclose(sampler.grid_norm_squared, sampler.norm_squared, rtol=1e-5)


@pytest.mark.parametrize(
    "monomial",
    [make_monomial({0: 1}), make_monomial({1: 2}), make_monomial({0: 1, 2: 1})],
)
def test_estimator_is_unbiased_and_respects_the_variance_bound(monomial):
    rng = np.random.default_rng(0)
    unitary = photonic_sim.haar_unitary(4, rng)
    occupation = (1, 0, 1, 0)
    patterns, probabilities = photonic_sim.output_distribution(unitary, occupation)
    exact = probabilities @ evaluate_monomial(monomial, patterns)
    draws = lim_oh.monomial_samples(
        unitary, occupation, monomial, 200_000, np.random.default_rng(1)
    )
    standard_error = draws.std() / np.sqrt(len(draws))
    assert abs(draws.mean() - exact) < 5 * standard_error
    norm_squared = 1.0
    for _, exponent in monomial:
        norm_squared *= sum(k ** (2 * exponent) for k in range(sum(occupation) + 1))
    # Var(X) <= ||O_A||^2 ||rho_A||^2 <= ||O_A||^2 (Lim-Oh Eq. S30).
    assert draws.var() <= 1.05 * norm_squared


def test_estimator_is_deterministic_for_a_seed():
    unitary = photonic_sim.haar_unitary(3, np.random.default_rng(2))
    first = lim_oh.monomial_samples(
        unitary, (1, 0, 1), make_monomial({0: 1}), 100, np.random.default_rng(5)
    )
    second = lim_oh.monomial_samples(
        unitary, (1, 0, 1), make_monomial({0: 1}), 100, np.random.default_rng(5)
    )
    np.testing.assert_array_equal(first, second)


def test_estimator_rejects_bunched_inputs():
    unitary = photonic_sim.haar_unitary(3, np.random.default_rng(3))
    with pytest.raises(ValueError):
        lim_oh.monomial_samples(
            unitary, (2, 0, 0), make_monomial({0: 1}), 10, np.random.default_rng(0)
        )


def test_comparison_smoke_run_checks_b2_and_neyman_allocation(tmp_path):
    cfg = load_runtime_ready_config()
    for key in ("cases", "monte_carlo"):
        cfg.pop(key, None)
    cfg.update(
        json.loads(
            (PROJECT_DIR / "configs" / "estimator_scaling_smoke.json").read_text()
        )
    )
    runner.train_and_evaluate(cfg, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["max_b2_abs_error"] < 1e-9
    assert metrics["lim_oh_unbiased"] is True
    import csv

    with (tmp_path / "estimator_cases.csv").open() as handle:
        cases = list(csv.DictReader(handle))
    with (tmp_path / "lim_oh_terms.csv").open() as handle:
        terms = list(csv.DictReader(handle))
    assert len(cases) == 2 * 2 * 3 * 2  # unitaries x families x sizes x draws
    assert all(float(case["n_lim_oh"]) > 0 for case in cases)
    assert all(abs(float(term["z"])) < 5 for term in terms)


def test_neyman_allocation_never_exceeds_equal_allocation():
    from photonic_trainability.lib.exp_estimator_scaling import neyman_samples

    rng = np.random.default_rng(4)
    for _ in range(50):
        coefficients = rng.uniform(-1, 1, size=20)
        sigmas = rng.uniform(0.1, 5.0, size=20)
        equal = len(sigmas) * np.sum(coefficients**2 * sigmas**2) / 0.1**2
        assert neyman_samples(coefficients, sigmas, 0.1) <= equal + 1e-9
    assert np.isclose(neyman_samples(np.ones(4), np.ones(4), 1.0), 16.0)
