# The QCBM estimator follows merlinquantum/generative_classical_training
# (src/models/mmd_estimator.py::compute_single_op, src/helpers/gurvits.py),
# commit 17a12c4, Copyright (c) 2026 merlinquantum, MIT License.
"""MMD estimators for the photonic QCBM (paper §3, Fig. 1, Eq. (15)).

Per sampled Walsh operator k (W_k = diag((-1)^{k_i})):

- model term: <s|U~^dagger W~_k U~|s> = Perm(A_k) with
  A_k = U[:, S]^dagger diag(1 - 2k) U[:, S] (n x n, S = input modes, Eq. (3)),
  estimated by Gurvits/Glynn: Perm(A) = E_z[prod(z) prod(A z)], z ~ U{±1}^n;
- data term: T_k = mean_x (-1)^{k.x} over the target set X;
- unbiased combination: G^2-hat - 2 G T + T^2-hat, where G^2-hat and T^2-hat
  are U-statistics (no self-pairs), as in the original code.

The returned loss is the mean over k of the real part. Every model built
with the same kernel sampler targets the same MMD^2, so values are comparable
across ansätze.

Only the n input columns of U are used. The original code forms the full
m x m product U^dagger diag(1 - 2k) U and slices it; both give the same A_k
(``tests/test_estimator.py`` checks this) at O(m n^2) instead of O(m^3).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

PHOTON_MODELS = ("indistinguishable", "distinguishable")


def walsh_expectations(K: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """Return E_x[(-1)^{k.x}] for each row k of K.

    Parameters
    ----------
    K : array of shape (n_ops, m)
        Walsh operators with entries in {0, 1}.
    X : array of shape (N, m)
        Samples (occupation numbers; parity uses x mod 2).

    Returns
    -------
    array of shape (n_ops,)
    """
    parity = jnp.mod(X @ K.T, 2.0)
    return jnp.mean(1.0 - 2.0 * parity, axis=0)


def walsh_operator_matrix(U_inputs: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    """A_k = U_S^dagger diag(1 - 2k) U_S for input columns U_S of shape (m, n)."""
    signs = (1.0 - 2.0 * k).astype(U_inputs.dtype)
    return jnp.conj(U_inputs.T) @ (signs[:, None] * U_inputs)


def glynn_samples(A: jnp.ndarray, Z: jnp.ndarray) -> jnp.ndarray:
    """Glynn estimator samples prod(z) prod(A z) for each row z of Z (paper Eq. (16))."""
    return jnp.prod(Z, axis=1) * jnp.prod(Z @ A.T, axis=1)


def _unbiased_square(values: jnp.ndarray) -> jnp.ndarray:
    # Unbiased estimate of (E v)^2 from i.i.d. samples: (S^2 - sum v^2) / (N (N - 1)).
    count = values.shape[0]
    return (jnp.sum(values) ** 2 - jnp.sum(values**2)) / (count * (count - 1))


def _data_square_term(T: jnp.ndarray, count: int) -> jnp.ndarray:
    # Unbiased estimate of T_true^2 for ±1 variables: (N T^2 - 1) / (N - 1).
    return (count * T * T - 1.0) / (count - 1)


def qcbm_mmd_terms(U_inputs, K, Z, X) -> jnp.ndarray:
    """Per-operator unbiased MMD^2 contributions for indistinguishable photons.

    Parameters
    ----------
    U_inputs : array of shape (m, n), complex
        Input columns U[:, S] of the interferometer.
    K : array of shape (n_ops, m)
        Sampled Walsh operators.
    Z : array of shape (n_z, n)
        Gurvits sign vectors in {-1, +1}.
    X : array of shape (N, m)
        Target samples.

    Returns
    -------
    array of shape (n_ops,), complex
    """
    T = walsh_expectations(K, X)
    count = X.shape[0]

    def one(k, t):
        G = glynn_samples(walsh_operator_matrix(U_inputs, k), Z)
        return _unbiased_square(G) - 2.0 * jnp.mean(G) * t + _data_square_term(t, count)

    return jax.vmap(one)(K, T)


def distinguishable_mmd_terms(U_inputs, K, X) -> jnp.ndarray:
    """Per-operator MMD^2 contributions for distinguishable photons (baseline B5).

    With distinguishable photons the output is a product of independent
    single-photon distributions, so <W_k> = prod_{s in S} sum_j |U_js|^2 (-1)^{k_j}
    exactly; only the data term needs the unbiased correction.
    """
    T = walsh_expectations(K, X)
    single_photon = (
        jnp.abs(U_inputs) ** 2
    )  # (m, n): column s = output distribution of photon s
    model = jnp.prod((1.0 - 2.0 * K) @ single_photon, axis=1)
    return model**2 - 2.0 * model * T + _data_square_term(T, X.shape[0])


def make_mmd_loss(
    unitary_fn: Callable,
    input_modes,
    operator_sampler: Callable,
    n_operators: int,
    n_gurvits: int,
    photon_model: str = "indistinguishable",
) -> Callable:
    """Build ``loss(params, key, X) -> MMD^2 estimate`` (fresh K, Z per call).

    Parameters
    ----------
    unitary_fn : Callable
        ``params -> U`` from ``circuits.make_unitary_fn``.
    input_modes : sequence of int
        Occupied input modes S (one photon each).
    operator_sampler : Callable
        ``(key, n_ops) -> K`` from ``kernels.make_operator_sampler``.
    n_operators : int
        |K|, Walsh operators per estimate.
    n_gurvits : int
        |Z|, Glynn samples per operator (unused for distinguishable photons).
    photon_model : str
        ``"indistinguishable"`` (the QCBM) or ``"distinguishable"`` (B5).
        Default value is "indistinguishable".

    Returns
    -------
    Callable
        Jitted function of (params, key, X).
    """
    if photon_model not in PHOTON_MODELS:
        raise ValueError(f"photon_model must be one of {PHOTON_MODELS}.")
    modes = jnp.asarray(input_modes)
    n_photons = int(modes.shape[0])

    @jax.jit
    def loss(params, key, X):
        key_k, key_z = jax.random.split(key)
        U_inputs = unitary_fn(params)[:, modes]
        K = operator_sampler(key_k, n_operators)
        if photon_model == "distinguishable":
            terms = distinguishable_mmd_terms(U_inputs, K, X)
        else:
            Z = (
                2 * jax.random.randint(key_z, (n_gurvits, n_photons), 0, 2) - 1
            ).astype(jnp.float64)
            terms = qcbm_mmd_terms(U_inputs, K, Z, X)
        return jnp.real(jnp.mean(terms))

    return loss


def walsh_mmd_unbiased(X: jnp.ndarray, Y: jnp.ndarray, K: jnp.ndarray) -> float:
    """Unbiased sample MMD^2 between sample sets X and Y for the kernel defined by K ~ P(k).

    Per operator: U-statistics for (E_p f)^2 and (E_q f)^2, cross term from
    the two means, f = (-1)^{k.x}. Averaged over the sampled operators.
    """
    fx = 1.0 - 2.0 * jnp.mod(X @ K.T, 2.0)
    fy = 1.0 - 2.0 * jnp.mod(Y @ K.T, 2.0)
    nx, ny = X.shape[0], Y.shape[0]
    xx = (jnp.sum(fx, 0) ** 2 - nx) / (nx * (nx - 1))
    yy = (jnp.sum(fy, 0) ** 2 - ny) / (ny * (ny - 1))
    xy = jnp.mean(fx, 0) * jnp.mean(fy, 0)
    return float(jnp.mean(xx - 2.0 * xy + yy))


def gaussian_mmd_unbiased(X, Y, sigma: float, chunk: int = 2048) -> float:
    """Unbiased sample MMD^2 with the exact Gaussian kernel (Gretton et al. 2012).

    Same estimator as ``qml_benchmarks.model_utils.mmd_loss``, vectorized and
    chunked. Inputs are used as given (occupation numbers or bitstrings).
    """
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    def kernel_sum(A, B, exclude_diagonal):
        total = 0.0
        for start in range(0, A.shape[0], chunk):
            block = A[start : start + chunk]
            sq = (block**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * block @ B.T
            total += np.exp(-np.maximum(sq, 0.0) / (2.0 * sigma**2)).sum()
        if exclude_diagonal:
            total -= A.shape[0]  # k(x, x) = 1
        return total

    nx, ny = X.shape[0], Y.shape[0]
    xx = kernel_sum(X, X, True) / (nx * (nx - 1))
    yy = kernel_sum(Y, Y, True) / (ny * (ny - 1))
    xy = kernel_sum(X, Y, False) / (nx * ny)
    return float(xx - 2.0 * xy + yy)
