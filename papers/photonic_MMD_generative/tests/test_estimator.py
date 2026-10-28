"""V1: the paper's MMD estimator against exact small-size references (Prop. 1-2, Fig. 1)."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the repo root on sys.path)
from photonic_MMD_generative.lib import circuits, estimator, exact, kernels

M, N, SIGMA = 6, 2, 1.0
INPUT_MODES = [1, 3]


@pytest.fixture(scope="module")
def haar_setup():
    unitary_fn = circuits.make_unitary_fn(M, "haar")
    params = jax.random.normal(jax.random.PRNGKey(3), (2 * M * M,))
    return unitary_fn, params, np.asarray(unitary_fn(params))


@pytest.fixture(scope="module")
def target_samples():
    rng = np.random.default_rng(0)
    collision_free = [s for s in exact.fock_states(M, N) if max(s) <= 1]
    weights = rng.random(len(collision_free))
    rows = rng.choice(len(collision_free), size=400, p=weights / weights.sum())
    return np.array([collision_free[i] for i in rows], dtype=float)


def _conditional_target(model_distribution: dict, X: np.ndarray) -> float:
    # E_{K,Z}[estimator | X] = sum_k P(k) [G_k^2 - 2 G_k T_k + (N T_k^2 - 1)/(N - 1)].
    probability = exact.gaussian_operator_probability(SIGMA)
    count = len(X)
    total = 0.0
    for bits in itertools.product((0, 1), repeat=M):
        k = np.array(bits)
        G = sum(p * (-1) ** int(k @ np.array(s)) for s, p in model_distribution.items())
        T = np.mean((-1.0) ** (X @ k))
        total += probability(k) * (
            G * G - 2 * G * T + (count * T * T - 1) / (count - 1)
        )
    return total


def test_permanent_distribution_matches_merlin(haar_setup):
    _, _, U = haar_setup
    ours = exact.output_distribution(U, INPUT_MODES)
    merlin = exact.merlin_output_distribution(U, INPUT_MODES)
    assert sum(ours.values()) == pytest.approx(1.0, abs=1e-12)
    assert max(abs(ours[s] - merlin[s]) for s in ours) < 1e-12


def test_proposition_1_walsh_form_equals_gaussian_mmd():
    rng = np.random.default_rng(1)
    support = [s for s in exact.fock_states(M, N) if max(s) <= 1]
    p, q = rng.random(len(support)), rng.random(len(support))
    p, q = (
        dict(zip(support, p / p.sum(), strict=True)),
        dict(zip(support, q / q.sum(), strict=True)),
    )
    walsh = exact.walsh_mmd(p, q, exact.gaussian_operator_probability(SIGMA))
    direct = exact.kernel_mmd(p, q, exact.gaussian_kernel(SIGMA))
    assert walsh == pytest.approx(direct, rel=1e-10)


def test_column_only_operator_matrix_equals_full_slice(haar_setup):
    _, _, U = haar_setup
    Uj = jnp.asarray(U)
    modes = jnp.asarray(INPUT_MODES)
    k = jnp.array([1.0, 0, 1, 1, 0, 0])
    full = (jnp.conj(Uj.T) @ ((1 - 2 * k)[:, None] * Uj))[modes[:, None], modes]
    assert (
        float(jnp.max(jnp.abs(estimator.walsh_operator_matrix(Uj[:, modes], k) - full)))
        < 1e-14
    )


def test_glynn_estimator_converges_to_permanent(haar_setup):
    _, _, U = haar_setup
    A = np.asarray(
        estimator.walsh_operator_matrix(
            jnp.asarray(U)[:, jnp.asarray(INPUT_MODES)], jnp.zeros(M)
        )
    )
    Z = jnp.asarray(np.array(list(itertools.product((-1.0, 1.0), repeat=N))))
    # Averaging over all 2^n sign vectors gives the permanent exactly.
    assert complex(
        jnp.mean(estimator.glynn_samples(jnp.asarray(A), Z))
    ) == pytest.approx(exact.permanent(A), abs=1e-12)


@pytest.mark.parametrize("photon_model", ["indistinguishable", "distinguishable"])
def test_estimator_is_unbiased(haar_setup, target_samples, photon_model):
    unitary_fn, params, U = haar_setup
    distribution = exact.output_distribution(
        U, INPUT_MODES, distinguishable=photon_model == "distinguishable"
    )
    target = _conditional_target(distribution, target_samples)
    sampler = kernels.make_operator_sampler({"type": "gaussian", "sigma": SIGMA}, M, N)
    loss = estimator.make_mmd_loss(
        unitary_fn, INPUT_MODES, sampler, 3000, 3000, photon_model
    )
    values = np.array(
        [
            float(loss(params, jax.random.PRNGKey(i), jnp.asarray(target_samples)))
            for i in range(30)
        ]
    )
    standard_error = values.std(ddof=1) / np.sqrt(len(values))
    assert abs(values.mean() - target) < 4 * standard_error + 1e-6


def test_sample_mmd_estimators_agree_with_exact():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 2, size=(3000, 5)).astype(float)
    Y = (rng.random((3000, 5)) < 0.3).astype(float)
    p, q = exact.empirical_distribution(X), exact.empirical_distribution(Y)
    exact_value = exact.kernel_mmd(p, q, exact.gaussian_kernel(1.0))
    # Unbiased U-statistic differs from the plug-in value by O(1/N).
    assert estimator.gaussian_mmd_unbiased(X, Y, 1.0) == pytest.approx(
        exact_value, abs=2e-3
    )
    K = kernels.make_operator_sampler({"type": "gaussian", "sigma": 1.0}, 5, 2)(
        jax.random.PRNGKey(0), 20000
    )
    assert estimator.walsh_mmd_unbiased(
        jnp.asarray(X), jnp.asarray(Y), K
    ) == pytest.approx(exact_value, abs=3e-3)
