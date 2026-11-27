"""Haar Monte Carlo estimates of the moments and variances in paper Eqs. 7-8.

For each sampled interferometer ``U`` the exact output distribution gives the
noise-free loss ``l(U) = <O>_U`` and the single-shot variance
``Var[O | U] = <O^2>_U - <O>_U^2``. Averaging over ``U`` estimates

* ``mu1 = E_U[l]``, ``mu2 = E_U[l^2]``,
* ``Var_circ = Var_U[l]`` (circuit variance),
* ``Var_samp = E_U[Var[O | U]]`` (sample variance).

Standard errors use batch means: the unitaries are split into equal batches,
each estimator is computed per batch, and the spread of the batch values gives
the error. This handles the variance estimators without extra assumptions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from .fock import fock_basis
from .photonic_sim import output_distribution

ObservableFunction = Callable[[np.ndarray], np.ndarray]


def per_unitary_moments(
    observable: ObservableFunction,
    occupation: Sequence[int],
    unitaries: Iterable[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``<O>_U`` and ``<O^2>_U`` for each interferometer.

    Parameters
    ----------
    observable : callable
        Maps ``(D, M)`` patterns to ``(D,)`` observable values.
    occupation : sequence of int
        Fock input.
    unitaries : iterable of numpy.ndarray
        Interferometers to evaluate.

    Returns
    -------
    means : numpy.ndarray
        ``<O>_U`` per unitary.
    second_moments : numpy.ndarray
        ``<O^2>_U`` per unitary.
    """
    basis = fock_basis(int(sum(occupation)), len(occupation))
    values = observable(basis)
    means, second_moments = [], []
    for unitary in unitaries:
        _, probabilities = output_distribution(unitary, occupation, basis)
        means.append(float(probabilities @ values))
        second_moments.append(float(probabilities @ values**2))
    return np.array(means), np.array(second_moments)


def _estimators(means: np.ndarray, second_moments: np.ndarray) -> dict[str, float]:
    return {
        "mu1": float(means.mean()),
        "mu2": float((means**2).mean()),
        "circuit_variance": float(means.var(ddof=1)),
        "sample_variance": float((second_moments - means**2).mean()),
        "mu1_of_square": float(second_moments.mean()),
    }


def moment_estimates(
    means: np.ndarray, second_moments: np.ndarray, batches: int
) -> dict[str, dict[str, float]]:
    """Point estimates and batch-means standard errors.

    Parameters
    ----------
    means, second_moments : numpy.ndarray
        Output of :func:`per_unitary_moments`.
    batches : int
        Number of equal batches; unitaries beyond a multiple of ``batches`` are
        dropped from the error estimate but kept in the point estimate.

    Returns
    -------
    dict
        ``{quantity: {"estimate": ..., "standard_error": ...}}``.
    """
    if batches < 2:
        raise ValueError("Need at least two batches for a standard error")
    point = _estimators(means, second_moments)
    batch_size = len(means) // batches
    if batch_size < 2:
        raise ValueError(f"{len(means)} samples are too few for {batches} batches")
    per_batch = [
        _estimators(
            means[index * batch_size : (index + 1) * batch_size],
            second_moments[index * batch_size : (index + 1) * batch_size],
        )
        for index in range(batches)
    ]
    return {
        name: {
            "estimate": value,
            "standard_error": float(
                np.std([batch[name] for batch in per_batch], ddof=1) / np.sqrt(batches)
            ),
        }
        for name, value in point.items()
    }
