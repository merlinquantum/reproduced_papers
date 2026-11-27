"""Ansätze unitarity (incl. the butterfly D3 finding) and kernel samplers (App. A)."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401
from photonic_MMD_generative.lib import circuits, kernels


@pytest.mark.parametrize("ansatz", circuits.ANSATZE)
def test_ansatze_are_unitary(ansatz):
    m = 16
    unitary_fn = circuits.make_unitary_fn(m, ansatz)
    params = jax.random.uniform(
        jax.random.PRNGKey(0), (circuits.num_parameters(m, ansatz),), maxval=6.0
    )
    assert circuits.unitarity_error(unitary_fn(params)) < 1e-12


def test_parameter_counts():
    assert circuits.num_parameters(100, "haar") == 20000
    assert circuits.num_parameters(100, "clements") == 10000
    assert len(circuits.butterfly_pairs(100)) == 100 * 99 // 2


def test_reference_butterfly_is_not_unitary_for_m_100_and_corrected_is():
    # D3: the original pair pattern overflows for m not a power of two.
    pairs = circuits.butterfly_pairs(100)
    assert int((pairs >= 100).any(axis=1).sum()) == 84
    params = jax.random.uniform(
        jax.random.PRNGKey(0),
        (circuits.num_parameters(100, "butterfly"),),
        maxval=2 * np.pi,
    )
    reference = circuits.make_unitary_fn(100, "butterfly", "reference")(params)
    corrected = circuits.make_unitary_fn(100, "butterfly", "corrected")(params)
    assert circuits.unitarity_error(reference) > 0.1
    assert circuits.unitarity_error(corrected) < 1e-12


def test_reference_and_corrected_butterfly_agree_for_powers_of_two():
    params = jax.random.normal(
        jax.random.PRNGKey(1), (circuits.num_parameters(64, "butterfly"),)
    )
    a = circuits.make_unitary_fn(64, "butterfly", "reference")(params)
    b = circuits.make_unitary_fn(64, "butterfly", "corrected")(params)
    assert float(jnp.max(jnp.abs(a - b))) == 0.0


def _kernel_from_operators(K, x, y):
    return float(jnp.mean((-1.0) ** jnp.mod(K @ (x + y), 2.0)))


@pytest.mark.parametrize(
    "kernel, exact_kernel",
    [
        (
            {"type": "gaussian", "sigma": 1.3},
            lambda x, y, m, n: np.exp(-np.sum((x - y) ** 2) / (2 * 1.3**2)),
        ),
        (
            {"type": "polynomial_parity", "degree": 2, "c": 1.5},
            lambda x, y, m, n: ((1.5 + np.mean((-1.0) ** ((x + y) % 2))) / 2.5) ** 2,
        ),
        (
            {"type": "polynomial_standard", "degree": 2, "c": 2.0},
            lambda x, y, m, n: ((2.0 + x @ y) / (2.0 + n)) ** 2,
        ),
    ],
)
def test_kernel_samplers_match_kernel_formulas(kernel, exact_kernel):
    # App. A / Prop. 2: K(x, y) = E_{k ~ P}[(-1)^{k.(x + y)}] on fixed-weight bitstrings.
    m, n = 8, 2
    sampler = kernels.make_operator_sampler(kernel, m, n)
    K = sampler(jax.random.PRNGKey(0), 200_000)
    states = [
        np.array(s, dtype=float)
        for s in itertools.product((0, 1), repeat=m)
        if sum(s) == n
    ][:6]
    for x, y in itertools.combinations(states, 2):
        assert _kernel_from_operators(
            K, jnp.asarray(x), jnp.asarray(y)
        ) == pytest.approx(exact_kernel(x, y, m, n), abs=1e-2)


def test_standard_polynomial_validity_is_checked():
    with pytest.raises(ValueError, match="c \\+ n - m/4"):
        kernels.make_operator_sampler(
            {"type": "polynomial_standard", "degree": 2, "c": 1.0}, 100, 10
        )


def test_low_order_operators_have_bounded_weight():
    sampler = kernels.make_operator_sampler(
        {"type": "low_order", "max_weight": 2}, 10, 2
    )
    weights = np.asarray(sampler(jax.random.PRNGKey(0), 1000).sum(axis=1))
    assert set(np.unique(weights)) == {1.0, 2.0}


def test_weighted_gaussian_normalizations_differ_by_m():
    marginals = np.full(10, 0.2)
    mean_w = kernels.mode_weights(marginals, 0.0, "mean")
    sum_w = kernels.mode_weights(marginals, 0.0, "sum")
    assert float(mean_w[0]) == pytest.approx(1.0)
    assert float(sum_w[0]) == pytest.approx(0.1)
