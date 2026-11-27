"""V2: App. B quantities (marginal formula, block initializations) against exact simulation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401
from photonic_MMD_generative.lib import circuits, exact, initialization


def test_marginal_formula_equals_exact_occupation_probability():
    # App. B.2: p_j = 1 - perm(I - u_j u_j^dagger) is the exact probability that mode j is occupied.
    m, modes = 6, [0, 2, 4]
    U = np.asarray(
        circuits.make_unitary_fn(m, "haar")(
            jax.random.normal(jax.random.PRNGKey(5), (2 * m * m,))
        )
    )
    formula = np.asarray(
        initialization.model_marginals(
            jnp.abs(jnp.asarray(U)[:, jnp.asarray(modes)]) ** 2
        )
    )
    distribution = exact.output_distribution(U, modes)
    occupied = np.array(
        [sum(p for s, p in distribution.items() if s[j] > 0) for j in range(m)]
    )
    assert np.max(np.abs(formula - occupied)) < 1e-12


def test_pairwise_uniform_init_is_uniform_on_pair_configurations():
    m, pairs = 6, 2
    params, _ = initialization.initial_parameters(
        {"strategy": "pairwise_uniform", "n_pairs": pairs},
        jax.random.PRNGKey(0),
        m,
        "haar",
    )
    U = np.asarray(circuits.make_unitary_fn(m, "haar")(params))
    distribution = exact.output_distribution(U, [0, 2])
    support = {s: p for s, p in distribution.items() if p > 1e-12}
    assert len(support) == 2**pairs
    assert all(p == pytest.approx(0.25) for p in support.values())


def test_block_uniform_init_is_uniform_on_block_configurations():
    m, sizes = 6, [3, 2]
    params, _ = initialization.initial_parameters(
        {"strategy": "block_uniform", "block_sizes": sizes},
        jax.random.PRNGKey(0),
        m,
        "haar",
    )
    U = np.asarray(circuits.make_unitary_fn(m, "haar")(params))
    distribution = exact.output_distribution(U, [0, 3])
    support = {s: p for s, p in distribution.items() if p > 1e-12}
    assert len(support) == 3 * 2
    assert all(p == pytest.approx(1 / 6) for p in support.values())


@pytest.mark.parametrize("ansatz", circuits.ANSATZE)
def test_identity_init_is_close_to_identity(ansatz):
    m = 8
    params, _ = initialization.initial_parameters(
        {"strategy": "identity", "perturbation": 0.01}, jax.random.PRNGKey(0), m, ansatz
    )
    U = circuits.make_unitary_fn(m, ansatz)(params)
    if ansatz == "mzi3":
        # The 3-MZI near-identity point is theta, phi ~ pi/2: a fixed permutation-phase, not I.
        assert float(jnp.max(jnp.abs(jnp.abs(U) - jnp.round(jnp.abs(U))))) < 0.05
    else:
        assert float(jnp.max(jnp.abs(U - jnp.eye(m)))) < 0.05


def test_data_dependent_init_reduces_marginal_loss():
    m, modes = 8, [0, 2]
    rng = np.random.default_rng(0)
    X = np.zeros((200, m))
    X[
        np.arange(200)[:, None],
        np.stack([rng.choice(4, 2, replace=False) for _ in range(200)]),
    ] = 1
    unitary_fn = circuits.make_unitary_fn(m, "haar")
    key = jax.random.PRNGKey(0)
    start = initialization.random_init(key, m, "haar")
    target = jnp.asarray(X.mean(axis=0))

    def marginal_loss(params):
        U_inputs = unitary_fn(params)[:, jnp.asarray(modes)]
        return float(
            0.5
            * jnp.sum(
                (initialization.model_marginals(jnp.abs(U_inputs) ** 2) - target) ** 2
            )
        )

    params, info = initialization.initial_parameters(
        {"strategy": "data_dependent", "steps": 300, "learning_rate": 0.5},
        key,
        m,
        "haar",
        unitary_fn,
        modes,
        jnp.asarray(X),
    )
    assert info["marginal_loss"] < 0.5 * marginal_loss(start)
