# Adapted from merlinquantum/generative_classical_training (src/helpers/initialization.py),
# commit 17a12c4, Copyright (c) 2026 merlinquantum, MIT License.
"""Initialization strategies for the QCBM parameters (paper §4.3, App. B).

All functions return a parameter vector in the layout of ``circuits``
(``haar``: 2 m^2 reals; meshes: packed (theta, phi) pairs then gammas).

Strategies (``init["strategy"]``), with the original code's distributions:
- ``identity``: haar = identity matrix + U[0, eps) on every real/imag entry;
  clements/butterfly = theta, phi, gamma ~ U[0, eps); mzi3 = theta, phi ~
  pi/2 ± eps/2, gamma ~ U[0, eps). eps = ``perturbation`` (code default 0.01).
- ``random``: haar entries ~ U[0, 1); mesh phases ~ U[0, 2 pi).
- ``weak_unbiased``: haar entries ~ N(0, 1) (Haar-distributed U, App. B.3);
  mesh phases ~ N(0, 1) (original code; the paper's "decompose a Haar U"
  is not implemented there, D2).
- ``data_dependent``: start from ``random`` and run plain gradient descent on
  L_marg (App. B.2) for ``steps`` steps at ``learning_rate``.
- ``pairwise_uniform`` / ``block_uniform`` (haar only): U = H on each pair or
  F_d on each block, identity elsewhere (§4.3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .circuits import MESH_ANSATZE, mesh_pairs, num_parameters, pack_mesh_params

jax.config.update("jax_enable_x64", True)

STRATEGIES = (
    "identity",
    "random",
    "weak_unbiased",
    "data_dependent",
    "pairwise_uniform",
    "block_uniform",
)
TWO_PI = 2.0 * np.pi


def _haar_params_from_unitary(U: jnp.ndarray) -> jnp.ndarray:
    # QR of a unitary with the diag(R) phase fix returns the unitary itself.
    return jnp.concatenate([jnp.real(U).ravel(), jnp.imag(U).ravel()])


def identity_init(key, m: int, ansatz: str, perturbation: float) -> jnp.ndarray:
    """Near-identity parameters (paper §4.3, App. B.1)."""
    if ansatz == "haar":
        base = jnp.zeros(2 * m * m).at[jnp.arange(m) * (m + 1)].set(1.0)
        return base + jax.random.uniform(key, (2 * m * m,), maxval=perturbation)
    num_mzi = len(mesh_pairs(m, ansatz))
    key_theta, key_phi, key_gamma = jax.random.split(key, 3)
    if ansatz == "mzi3":
        low, high = np.pi / 2 - perturbation / 2, np.pi / 2 + perturbation / 2
        theta = jax.random.uniform(key_theta, (num_mzi,), minval=low, maxval=high)
        phi = jax.random.uniform(key_phi, (num_mzi,), minval=low, maxval=high)
    else:
        theta = jax.random.uniform(key_theta, (num_mzi,), maxval=perturbation)
        phi = jax.random.uniform(key_phi, (num_mzi,), maxval=perturbation)
    gammas = jax.random.uniform(key_gamma, (m,), maxval=perturbation)
    return pack_mesh_params(jnp.stack([theta, phi], axis=1), gammas)


def random_init(key, m: int, ansatz: str) -> jnp.ndarray:
    """Uniformly random parameters in the ansatz's natural range."""
    if ansatz == "haar":
        return jax.random.uniform(key, (2 * m * m,))
    return jax.random.uniform(key, (num_parameters(m, ansatz),), maxval=TWO_PI)


def weak_unbiased_init(key, m: int, ansatz: str) -> jnp.ndarray:
    """Normally distributed parameters (Haar-distributed U for the haar ansatz)."""
    return jax.random.normal(key, (num_parameters(m, ansatz),))


def pairwise_uniform_unitary(m: int, n_pairs: int) -> jnp.ndarray:
    """50:50 (Hadamard) on modes (0,1), (2,3), ..., identity elsewhere."""
    U = jnp.eye(m, dtype=jnp.complex128)
    H = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex128) / jnp.sqrt(2.0)
    for pair in range(n_pairs):
        U = U.at[2 * pair : 2 * pair + 2, 2 * pair : 2 * pair + 2].set(H)
    return U


def block_uniform_unitary(m: int, block_sizes) -> jnp.ndarray:
    """F_{d_1} ⊕ F_{d_2} ⊕ ... ⊕ identity on the padding modes."""
    U = jnp.eye(m, dtype=jnp.complex128)
    start = 0
    for size in map(int, block_sizes):
        index = jnp.arange(size)
        F = jnp.exp(2j * jnp.pi * index[:, None] * index[None, :] / size) / jnp.sqrt(
            size
        )
        U = U.at[start : start + size, start : start + size].set(F)
        start += size
    return U


def model_marginals(U_inputs_abs2: jnp.ndarray) -> jnp.ndarray:
    """Single-mode occupation probabilities p_j = sum_{k>=1} (-1)^{k+1} k! e_k(|U_{j,S}|^2) (App. B.2)."""
    n = U_inputs_abs2.shape[1]
    orders = jnp.arange(1, n + 1)
    weights = (-1.0) ** (orders + 1) * jnp.exp(lax.lgamma(orders + 1.0))

    def one_row(row):
        e = jnp.zeros(n + 1).at[0].set(1.0)

        def step(e, value):
            # e_k <- e_k + value * e_{k-1}, k = n..1 (backward update).
            shifted = jnp.concatenate([jnp.zeros(1), e[:-1]])
            return e + value * shifted, None

        e, _ = lax.scan(step, e, row)
        return jnp.sum(weights * e[1:])

    return jax.vmap(one_row)(U_inputs_abs2)


def data_dependent_init(
    key, m, ansatz, unitary_fn, input_modes, X_train, steps, learning_rate
):
    """Warm start: gradient descent on L_marg = 1/2 sum_j (p_j(theta) - p_j^target)^2 (App. B.2).

    Returns
    -------
    tuple
        (parameters, final L_marg value).
    """
    target = jnp.mean(jnp.asarray(X_train, dtype=jnp.float64), axis=0)
    modes = jnp.asarray(input_modes)

    def marginal_loss(params):
        U_inputs = unitary_fn(params)[:, modes]
        return 0.5 * jnp.sum((model_marginals(jnp.abs(U_inputs) ** 2) - target) ** 2)

    grad_fn = jax.jit(jax.value_and_grad(marginal_loss))

    def step(params, _):
        value, grad = grad_fn(params)
        return params - learning_rate * grad, value

    params, _ = lax.scan(step, random_init(key, m, ansatz), None, length=steps)
    return params, float(marginal_loss(params))


def initial_parameters(
    init: dict,
    key,
    m: int,
    ansatz: str,
    unitary_fn=None,
    input_modes=None,
    X_train=None,
):
    """Dispatch on ``init["strategy"]`` and return (params, info dict).

    Raises
    ------
    ValueError
        If the strategy is unknown or not available for the ansatz, or a
        required key is missing.
    """
    strategy = init["strategy"]
    if strategy == "identity":
        return identity_init(key, m, ansatz, init["perturbation"]), {}
    if strategy == "random":
        return random_init(key, m, ansatz), {}
    if strategy == "weak_unbiased":
        return weak_unbiased_init(key, m, ansatz), {}
    if strategy == "data_dependent":
        params, final_loss = data_dependent_init(
            key,
            m,
            ansatz,
            unitary_fn,
            input_modes,
            X_train,
            init["steps"],
            init["learning_rate"],
        )
        return params, {"marginal_loss": final_loss}
    if strategy in ("pairwise_uniform", "block_uniform"):
        if ansatz != "haar":
            # TODO: mesh versions exist in the original code for pairwise only; add if Fig. 4 needs meshes.
            raise ValueError(f"{strategy} is implemented for the haar ansatz only.")
        if strategy == "pairwise_uniform":
            U = pairwise_uniform_unitary(m, init["n_pairs"])
        else:
            U = block_uniform_unitary(m, init["block_sizes"])
        return _haar_params_from_unitary(U), {}
    raise ValueError(
        f"Unknown init strategy {strategy!r}; expected one of {STRATEGIES}."
    )


__all__ = [
    "MESH_ANSATZE",
    "STRATEGIES",
    "initial_parameters",
    "model_marginals",
    "pairwise_uniform_unitary",
    "block_uniform_unitary",
]
