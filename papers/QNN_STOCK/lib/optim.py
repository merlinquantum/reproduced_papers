"""SPSA (paper Eq. 11, 19) and exact parameter-shift gradients (Eq. 10)."""

from __future__ import annotations

from typing import Callable

import numpy as np

LossFn = Callable[[np.ndarray], float]


def spsa_step(
    loss_fn: LossFn,
    params: np.ndarray,
    lr: float,
    perturbation: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """One SPSA iteration with constant gains (the paper states no gain decay).

    ``g = (L(theta + c Delta) - L(theta - c Delta)) / (2 c) * Delta`` with
    Rademacher ``Delta``; ``theta <- theta - lr * g``. Each call evaluates the
    (possibly shot-sampled) loss twice, which is one paper "epoch".

    Returns
    -------
    tuple
        Updated parameters and the mean of the two loss evaluations.
    """

    delta = rng.choice(np.array([-1.0, 1.0]), size=params.shape)
    loss_plus = loss_fn(params + perturbation * delta)
    loss_minus = loss_fn(params - perturbation * delta)
    gradient = (loss_plus - loss_minus) / (2.0 * perturbation) * delta
    return params - lr * gradient, 0.5 * (loss_plus + loss_minus)


def parameter_shift_gradient(loss_fn: LossFn, params: np.ndarray) -> np.ndarray:
    """Exact gradient for losses affine in an expectation value of the circuit.

    Valid for the SWAP-test loss ``(1 - F) / 2`` because ``F`` is the
    expectation of a projector and every parameter is a Pauli rotation angle.
    Not valid for the MSE loss, which is quadratic in the probabilities.
    """

    gradient = np.zeros_like(params, dtype=float)
    for index in range(len(params)):
        shift = np.zeros_like(params, dtype=float)
        shift[index] = 0.5 * np.pi
        gradient[index] = 0.5 * (loss_fn(params + shift) - loss_fn(params - shift))
    return gradient
