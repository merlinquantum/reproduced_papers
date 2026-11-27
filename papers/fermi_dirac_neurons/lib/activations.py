"""Scalar activations, their derivatives, and spectral (matrix) calculus.

Every activation is applied to a Hermitian matrix through its eigenvalues
(functional calculus). Gradients of ``Tr[phi(H(theta)) R]`` use the
Daleckii-Krein formula

    d/dtheta_j Tr[phi(H) R] = sum_ab F_ab (V^dag H_j V)_ab (V^dag R V)_ba,
    F_ab = (phi(l_a) - phi(l_b)) / (l_a - l_b),  F_aa = phi'(l_a),

with ``H = V diag(l) V^dag``. This is the exact gradient, the same method as
the authors' notebooks (``fdd_*_matrix`` / ``dfj``).

Paper definitions (temperature ``T > 0``):

* ``tanh``: g_T(x) = tanh(x / T), Eq. (18) (the paper's Fermi-Dirac neuron).
* ``fermi_dirac``: f_T(x) = 1 / (exp(-x / T) + 1), Eq. (2).
* ``softplus``: r_T(x) = T ln(1 + exp(x / T)), Eq. (68).
* ``silu``: SiLU_T(x) = x f_T(x), Eq. (74).
* ``linear``: x (the "quantum linear model" of Fig. 9).
* ``exp``: exp(x / T), the target used by ``softplus_nqubits.ipynb`` (D1).
* ``softplus_code``: T ln(1 + exp(x / T**2)), the model the softplus notebook
  actually trains, because it applies r_T to H / T (D2).
* ``logistic_pos`` / ``logistic_neg``: T ln(1 + exp(-y x / T)) for y = +1 / -1,
  the logistic-loss observable of Eq. (57).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, log_expit

# Below this eigenvalue gap the divided difference is replaced by phi' at the
# midpoint; the truncation error is O(gap**2 * phi''') < 1e-10 and the
# cancellation error of the direct quotient above it is < 1e-11.
_DIVIDED_DIFFERENCE_GAP = 1e-5


@dataclass(frozen=True)
class Activation:
    """A scalar activation and its first derivative.

    Parameters
    ----------
    name : str
        Registry name.
    value : callable
        Elementwise ``phi(x)`` on a NumPy array.
    derivative : callable
        Elementwise ``phi'(x)``.
    """

    name: str
    value: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray], np.ndarray]


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def make_activation(name: str, temperature: float) -> Activation:
    """Return the activation ``name`` at temperature ``T``.

    Raises
    ------
    ValueError
        If ``name`` is unknown or ``temperature <= 0``.
    """
    T = float(temperature)
    if T <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if name == "tanh":
        return Activation(
            name,
            lambda x: np.tanh(x / T),
            lambda x: (1.0 - np.tanh(x / T) ** 2) / T,
        )
    if name == "fermi_dirac":
        return Activation(
            name,
            lambda x: expit(x / T),
            lambda x: expit(x / T) * expit(-x / T) / T,
        )
    if name == "softplus":
        return Activation(name, lambda x: T * _softplus(x / T), lambda x: expit(x / T))
    if name == "softplus_code":
        return Activation(
            name,
            lambda x: T * _softplus(x / T**2),
            lambda x: expit(x / T**2) / T,
        )
    if name == "silu":
        return Activation(
            name,
            lambda x: x * expit(x / T),
            lambda x: expit(x / T) + (x / T) * expit(x / T) * expit(-x / T),
        )
    if name == "linear":
        return Activation(name, lambda x: x, np.ones_like)
    if name == "exp":
        return Activation(name, lambda x: np.exp(x / T), lambda x: np.exp(x / T) / T)
    if name in ("logistic_pos", "logistic_neg"):
        label = 1.0 if name == "logistic_pos" else -1.0
        return Activation(
            name,
            lambda x: -T * log_expit(label * x / T),
            lambda x: -label * expit(-label * x / T),
        )
    raise ValueError(f"Unknown activation {name!r}")


def divided_differences(activation: Activation, eigenvalues: np.ndarray) -> np.ndarray:
    """Return the first divided-difference matrix ``F`` of ``phi`` on ``eigenvalues``.

    Parameters
    ----------
    activation : Activation
        Scalar function and derivative.
    eigenvalues : numpy.ndarray
        Real eigenvalues, shape ``(d,)``.

    Returns
    -------
    numpy.ndarray
        Real matrix of shape ``(d, d)``.
    """
    left = eigenvalues[:, None]
    right = eigenvalues[None, :]
    gap = left - right
    close = np.abs(gap) < _DIVIDED_DIFFERENCE_GAP
    safe_gap = np.where(close, 1.0, gap)
    quotient = (activation.value(left) - activation.value(right)) / safe_gap
    midpoint_derivative = activation.derivative(0.5 * (left + right))
    return np.where(close, midpoint_derivative, quotient)


def matrix_function(
    activation: Activation, eigenvalues: np.ndarray, eigenvectors: np.ndarray
) -> np.ndarray:
    """Return ``phi(H) = V diag(phi(l)) V^dag``."""
    return (eigenvectors * activation.value(eigenvalues)) @ eigenvectors.conj().T
