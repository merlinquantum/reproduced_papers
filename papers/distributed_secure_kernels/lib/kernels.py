"""Kernel matrices: exact references and finite-shot swap-test estimates.

Swap test (paper App. A): the ancilla reads 0 with probability
``p0 = 1/2 + 1/2 |<psi|phi>|^2``. The authors' estimator (``library.py``) is
``K = 1 - 2 * n_ones / shots = 2 * n_zeros / shots - 1``, clipped at 0, so the
quantity estimated is the *squared* overlap. For amplitude-encoded data this
is the squared cosine similarity ``(x.y)^2 / (||x||^2 ||y||^2)``, not the
linear kernel named in the paper's Sec. 3.1. The paper's Eq. 11 (App. A) takes
the square root, ``|<psi|phi>| = sqrt(2 Pr(0) - 1)``; the authors' code does not.

Without noise, teleportation is exact, so the distributed and centralised
circuits give the same ``p0`` and each kernel entry is
``Binomial(shots, p0)``-distributed; ``tests/test_protocol.py`` checks this
against per-shot Aer runs of the authors' circuits.
"""

from __future__ import annotations

import numpy as np

from .encoding import amplitude_encode


def squared_overlap_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Exact ``|<psi(x)|psi(y)>|^2`` for amplitude-encoded rows (infinite shots, no noise)."""

    return (amplitude_encode(left) @ amplitude_encode(right).T) ** 2


def cosine_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """``<psi(x)|psi(y)> = x.y / (||x|| ||y||)``: the "linear kernel" the paper names."""

    return amplitude_encode(left) @ amplitude_encode(right).T


def swap_test_zero_probability(overlap_squared: np.ndarray) -> np.ndarray:
    """Ideal swap-test probability of reading 0: ``1/2 + 1/2 |<psi|phi>|^2``."""

    return 0.5 + 0.5 * overlap_squared


def estimate_from_zero_probability(
    zero_probability: np.ndarray,
    shots: int,
    rng: np.random.Generator,
    symmetric: bool,
) -> np.ndarray:
    """Draw the authors' finite-shot kernel estimate from swap-test probabilities.

    Parameters
    ----------
    zero_probability : numpy.ndarray
        ``p0`` for every pair, shape ``(n_left, n_right)``.
    shots : int
        Circuit repetitions per kernel entry (the paper uses 1,024).
    rng : numpy.random.Generator
        Shot-noise generator.
    symmetric : bool
        If True (square training kernel), draw each unordered pair once, as the
        authors' ``compute_partial_kernel`` does for ``j <= i``, and mirror it.

    Returns
    -------
    numpy.ndarray
        ``max(0, 2 * n_zeros / shots - 1)`` per entry.
    """

    probability = np.clip(zero_probability, 0.0, 1.0)
    if symmetric:
        if probability.shape[0] != probability.shape[1]:
            raise ValueError("symmetric=True needs a square probability matrix")
        rows, cols = np.tril_indices(probability.shape[0])
        zeros = np.zeros(probability.shape)
        zeros[rows, cols] = rng.binomial(shots, probability[rows, cols])
        zeros[cols, rows] = zeros[rows, cols]
    else:
        zeros = rng.binomial(shots, probability)
    return np.maximum(0.0, 2.0 * zeros / shots - 1.0)


def finite_shot_kernel(
    left: np.ndarray,
    right: np.ndarray,
    shots: int,
    rng: np.random.Generator,
    symmetric: bool,
) -> np.ndarray:
    """Noiseless finite-shot swap-test kernel (centralised or distributed circuit)."""

    probability = swap_test_zero_probability(squared_overlap_kernel(left, right))
    return estimate_from_zero_probability(probability, shots, rng, symmetric)
