"""Compact swap-test cosine projection (paper Sec. 4.1, Algorithms 1--3).

The compact swap test estimates the cosine similarity between two real vectors
via the control-qubit measurement of a Fredkin (CSWAP) circuit:
``cos(x, theta) ~ sqrt(2 p0 - 1)`` where ``p0`` is the control-qubit measurement
probability. In the noiseless / infinite-shot limit the magnitude estimate equals
the exact classical cosine similarity ``x . theta / (||x|| ||theta||)``, but the
measurement yields no sign information. The sign is recovered from the exact
classical dot product ``np.sign(x . theta)``, so this is not a pure quantum
estimator (hybrid classical-quantum; see swap_test_cosine below).

We compute the cosine magnitude analytically and optionally add Gaussian
measurement noise whose scale follows the shot count (the paper's ``sigma`` /
``shots`` knobs), so the forward pass matches the paper's infinite-shot limit
exactly while still allowing a finite-shot robustness study.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(x: np.ndarray, theta: np.ndarray) -> float:
    nx = float(np.linalg.norm(x))
    nt = float(np.linalg.norm(theta))
    if nx == 0.0 or nt == 0.0:
        return 0.0
    return float(np.dot(x, theta) / (nx * nt))


def swap_test_cosine(
    x: np.ndarray,
    theta: np.ndarray,
    shots: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Compact swap-test cosine estimate (analytic magnitude; sign from classical dot product).

    With ``shots is None`` returns the exact cosine.  With a finite ``shots`` the
    control-qubit probability ``p0 = (1 + cos^2)/2`` is sampled from a binomial
    over ``shots`` measurements and inverted to estimate ``|cos|``. The sign
    ``np.sign(x . theta)`` comes from the exact classical dot product, not from
    the quantum measurement (which provides no sign information). This is a
    hybrid classical-quantum estimator, not a pure finite-shot swap-test.
    """
    c = cosine_similarity(x, theta)
    if shots is None or shots <= 0:
        return c
    rng = rng or np.random.default_rng()
    p0_true = 0.5 * (1.0 + c * c)
    p0 = rng.binomial(shots, p0_true) / shots
    est = np.sqrt(max(2.0 * p0 - 1.0, 0.0))
    return float(np.sign(c) * est)


def binary_entropy(p: float) -> float:
    """Binary Shannon entropy H(p) in bits (paper Eq. 18)."""
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))
