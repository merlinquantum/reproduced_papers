from __future__ import annotations

import math

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401
from QuaRK.lib.bound import generalization_bound_terms
from QuaRK.lib.kernel_ridge import matern_from_distances, solve_dual
from QuaRK.lib.measurement import (
    exact_expectations,
    k_local_paulis,
    pauli_features,
    pauli_matrix,
)
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import Matern


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, 5.0])
def test_matern_matches_scikit_learn(nu):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(40, 6)), rng.normal(size=(30, 6))
    ours = matern_from_distances(cdist(a, b), 1.7, nu)
    np.testing.assert_allclose(
        ours, Matern(length_scale=1.7, nu=nu)(a, b), rtol=1e-10, atol=1e-12
    )


def test_solve_dual_interpolates_at_small_ridge():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(50, 4)), rng.normal(size=50)
    gram = matern_from_distances(cdist(x, x), 1.0, 2.5)
    assert np.max(np.abs(gram @ solve_dual(gram, y, 1e-12) - y)) < 1e-6


def test_observable_count_and_exact_expectation():
    paulis = k_local_paulis(2, 5)
    assert len(paulis) == 15 + 90
    rng = np.random.default_rng(2)
    a = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    values = exact_expectations(rho[None, None], paulis)[0, 0]
    k = 37
    assert np.isclose(values[k], np.trace(pauli_matrix(paulis[k], 5) @ rho).real)


def test_random_pauli_shadows_are_unbiased():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    states = np.broadcast_to(rho, (400, 1, 8, 8)).copy()
    exact = pauli_features(states[:1], locality=2, estimator="exact", shots=0, seed=0)[
        0
    ]
    shadows = pauli_features(
        states, locality=2, estimator="random_pauli_shadows", shots=1000, seed=4
    )
    # Median of means on a symmetric distribution: the average over 400 copies is unbiased
    # to within about 4 standard errors.
    spread = shadows.std(axis=0) / math.sqrt(states.shape[0])
    assert np.all(np.abs(shadows.mean(axis=0) - exact) < 5 * spread + 1e-3)


def test_bound_terms():
    terms = generalization_bound_terms(
        rkhs_norm=2.0,
        label_bound=1.0,
        num_train=100,
        delta=0.05,
        nu=2.5,
        xi=1.0,
        num_features=10,
        max_contraction=0.5,
        window=2,
    )
    assert math.isclose(terms["rademacher"], 4 * math.sqrt(2) * 2 * 3 / 10)
    assert math.isclose(terms["mixing"], 3 * 9 * math.sqrt(math.log(80) / 100))
    assert math.isclose(
        terms["truncation"], 4 * 2 * 3 * math.sqrt(2.5 * 10 / 1.5) * 0.25
    )
    invalid = generalization_bound_terms(
        rkhs_norm=2.0,
        label_bound=1.0,
        num_train=100,
        delta=0.05,
        nu=0.5,
        xi=1.0,
        num_features=10,
        max_contraction=0.5,
        window=2,
    )
    assert invalid["truncation"] is None and invalid["total"] is None
