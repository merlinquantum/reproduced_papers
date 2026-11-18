# Adapted from merlinquantum/generative_classical_training (src/helpers/kernels.py),
# commit 17a12c4, Copyright (c) 2026 merlinquantum, MIT License.
"""Samplers of Walsh operators k ~ P(k) for the MMD observable (paper §3, App. A).

For a stationary kernel on bitstrings, K(x, y) = sum_k P(k) (-1)^{k.x} (-1)^{k.y}
(Prop. 2), so MMD^2(p, q) = E_{k ~ P}[(<W_k>_p - <W_k>_q)^2] with
<W_k>_p = E_{x ~ p}[(-1)^{k.x}]. Each sampler returns K of shape (n_ops, m),
float64 entries in {0, 1}.

Kernels (``kernel["type"]``):
- ``gaussian``: exp(-||x-y||^2 / 2 sigma^2); k_i ~ Bernoulli(p_sigma) i.i.d.,
  p_sigma = (1 - exp(-1/(2 sigma^2))) / 2 (paper Eq. (11)-(12)).
- ``weighted_gaussian``: per-mode bandwidths sigma^2 / w_i with data weights
  (App. A, Eq. (17)). ``weight_normalization`` = "mean" (original code, weights
  average to 1) or "sum" (paper Eq. (17), weights sum to 1) — D4.
- ``polynomial_parity``: ((c + Phi(x).Phi(y)) / (c + 1))^d on parity features.
- ``polynomial_standard``: ((c + x.y) / (c + n))^d on fixed-weight data; valid
  only if c + n - m/4 >= 0 (checked, App. A).
- ``low_order``: |k| uniform in 1..r, support uniform over modes.
- ``data_biased_low_order``: |k| uniform in 1..r, support drawn without
  replacement with probabilities from the data marginals (App. A; D8).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

KERNEL_TYPES = (
    "gaussian",
    "weighted_gaussian",
    "polynomial_parity",
    "polynomial_standard",
    "low_order",
    "data_biased_low_order",
)


def gaussian_bernoulli_probability(sigma: float) -> jnp.ndarray:
    """Return p_sigma = (1 - exp(-1 / (2 sigma^2))) / 2."""
    return (1.0 - jnp.exp(-1.0 / (2.0 * sigma**2))) / 2.0


def mode_weights(mode_marginals, epsilon: float, normalization: str) -> jnp.ndarray:
    """Data weights w_i ∝ mu_i + epsilon for the weighted Gaussian kernel.

    Parameters
    ----------
    mode_marginals : array of shape (m,)
        Empirical occupation probability of each mode in the training set.
    epsilon : float
        Smoothing constant added to every marginal.
    normalization : str
        ``"mean"`` (original code: mean weight 1) or ``"sum"`` (paper Eq. (17):
        weights sum to 1).
    """
    raw = jnp.asarray(mode_marginals, dtype=jnp.float64) + epsilon
    if normalization == "mean":
        return raw / jnp.mean(raw)
    if normalization == "sum":
        return raw / jnp.sum(raw)
    raise ValueError("weight_normalization must be 'mean' or 'sum'.")


@partial(jax.jit, static_argnames=("m", "n_ops"))
def _bernoulli_operators(key, probabilities, m: int, n_ops: int):
    return jax.random.bernoulli(key, p=probabilities, shape=(n_ops, m)).astype(
        jnp.float64
    )


@partial(jax.jit, static_argnames=("m", "n_ops", "degree"))
def _parity_polynomial_operators(
    key, m: int, n_ops: int, degree: int, feature_probability
):
    # Each of the `degree` factors picks either the constant term or one
    # uniformly chosen parity feature; repeated features cancel modulo 2.
    def one(subkey):
        def body(k_vec, step_key):
            key_choice, key_index = jax.random.split(step_key)
            pick_feature = jax.random.bernoulli(key_choice, p=feature_probability)
            index = jax.random.randint(key_index, (), 0, m)
            flipped = k_vec.at[index].set(1.0 - k_vec[index])
            return jnp.where(pick_feature, flipped, k_vec), None

        k_final, _ = jax.lax.scan(body, jnp.zeros(m), jax.random.split(subkey, degree))
        return k_final

    return jax.vmap(one)(jax.random.split(key, n_ops))


@partial(jax.jit, static_argnames=("m", "n_ops", "max_weight"))
def _low_order_operators(key, mode_probabilities, m: int, n_ops: int, max_weight: int):
    def one(subkey):
        key_weight, key_modes = jax.random.split(subkey)
        weight = jax.random.randint(key_weight, (), 1, max_weight + 1)
        modes = jax.random.choice(
            key_modes, m, shape=(max_weight,), replace=False, p=mode_probabilities
        )
        mask = (jnp.arange(max_weight) < weight).astype(jnp.float64)
        return jnp.zeros(m).at[modes].set(mask)

    return jax.vmap(one)(jax.random.split(key, n_ops))


def make_operator_sampler(kernel: dict, m: int, n: int, mode_marginals=None):
    """Return ``(key, n_ops) -> K`` for one kernel configuration.

    Parameters
    ----------
    kernel : dict
        Kernel config with key ``type`` and the type's hyperparameters
        (``sigma``; ``degree``, ``c``; ``max_weight``; ``epsilon``,
        ``weight_normalization``).
    m, n : int
        Number of modes and photons.
    mode_marginals : array of shape (m,) | None
        Training-set mode marginals, required by the data-adaptive kernels.
        Default value is None.

    Returns
    -------
    Callable
        ``sample(key, n_ops)`` returning an (n_ops, m) float64 array.

    Raises
    ------
    ValueError
        If the kernel type is unknown, a required hyperparameter is missing, or
        the standard polynomial kernel is not a valid sampling distribution.
    """
    kernel_type = kernel["type"]
    if kernel_type == "gaussian":
        probability = gaussian_bernoulli_probability(kernel["sigma"])
        return lambda key, n_ops: _bernoulli_operators(key, probability, m, n_ops)

    if kernel_type == "weighted_gaussian":
        if mode_marginals is None:
            raise ValueError("weighted_gaussian needs the training mode marginals.")
        weights = mode_weights(
            mode_marginals, kernel["epsilon"], kernel["weight_normalization"]
        )
        probabilities = (1.0 - jnp.exp(-weights / (2.0 * kernel["sigma"] ** 2))) / 2.0
        return lambda key, n_ops: _bernoulli_operators(key, probabilities, m, n_ops)

    if kernel_type == "polynomial_parity":
        feature_probability = 1.0 / (kernel["c"] + 1.0)
        return lambda key, n_ops: _parity_polynomial_operators(
            key, m, n_ops, kernel["degree"], feature_probability
        )

    if kernel_type == "polynomial_standard":
        constant_weight = kernel["c"] + n - m / 4.0
        if constant_weight < 0:
            raise ValueError(
                f"polynomial_standard needs c + n - m/4 >= 0 (got {constant_weight}); increase c."
            )
        feature_probability = (m / 4.0) / (kernel["c"] + n)
        return lambda key, n_ops: _parity_polynomial_operators(
            key, m, n_ops, kernel["degree"], feature_probability
        )

    if kernel_type == "low_order":
        uniform = jnp.full(m, 1.0 / m)
        return lambda key, n_ops: _low_order_operators(
            key, uniform, m, n_ops, kernel["max_weight"]
        )

    if kernel_type == "data_biased_low_order":
        if mode_marginals is None:
            raise ValueError("data_biased_low_order needs the training mode marginals.")
        probabilities = (
            jnp.asarray(mode_marginals, dtype=jnp.float64) + kernel["epsilon"]
        )
        probabilities = probabilities / jnp.sum(probabilities)
        return lambda key, n_ops: _low_order_operators(
            key, probabilities, m, n_ops, kernel["max_weight"]
        )

    raise ValueError(
        f"Unknown kernel type {kernel_type!r}; expected one of {KERNEL_TYPES}."
    )


def gaussian_kernel_matrix(X: jnp.ndarray, Y: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Exact Gaussian kernel matrix exp(-||x - y||^2 / (2 sigma^2))."""
    sq = jnp.sum(X**2, 1)[:, None] + jnp.sum(Y**2, 1)[None, :] - 2.0 * X @ Y.T
    return jnp.exp(-jnp.maximum(sq, 0.0) / (2.0 * sigma**2))
