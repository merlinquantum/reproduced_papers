"""Fourier analysis of model outputs and predicted frequency spectra.

Notation follows the paper (Section II): a model is ``f(x) = sum_w c_w exp(i w x)``
with spectrum ``Omega = {Lambda_k - Lambda_j}``, where ``Lambda_j`` is a sum of
``L`` eigenvalues of the encoding generator (Eq. 9). Spectrum size is
``K = (|Omega| - 1) / 2`` and degree is ``D = max Omega``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

import numpy as np

# Rounding used to de-duplicate real-valued frequencies computed from sums of
# eigenvalues. Eigenvalues in this project are multiples of 1/2 or of a base
# frequency, so 9 decimals separate distinct frequencies safely.
_FREQUENCY_DECIMALS = 9


def periodic_grid(n_points: int, period: float = 2 * np.pi) -> np.ndarray:
    """Return ``n_points`` equispaced points of ``[0, period)`` (endpoint excluded)."""
    return np.linspace(0.0, period, n_points, endpoint=False)


def fourier_coefficients(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return integer frequencies and coefficients of samples on a periodic grid.

    For samples ``values[k] = f(2 pi k / N)`` of ``f(x) = sum_n c_n exp(i n x)``
    with ``|n| < N / 2``, ``np.fft.fft(values) / N`` returns ``c_n`` exactly.

    Parameters
    ----------
    values : np.ndarray
        Samples on ``periodic_grid(N)``, shape ``(N,)``.

    Returns
    -------
    tuple of np.ndarray
        Integer frequencies ``n`` (FFT order) and complex ``c_n``.
    """
    n_points = values.shape[-1]
    coefficients = np.fft.fft(values, axis=-1) / n_points
    frequencies = np.rint(np.fft.fftfreq(n_points, d=1.0 / n_points)).astype(int)
    return frequencies, coefficients


def reference_coefficients(values: np.ndarray) -> np.ndarray:
    """Reproduce the reference notebook's coefficient extraction (Fig. 5).

    The notebook samples ``2K + 1`` points of ``[0, 2 pi)`` and returns
    ``np.fft.rfft(f(t)) / t.size``, i.e. ``c_0 ... c_K``.

    Parameters
    ----------
    values : np.ndarray
        Samples on ``periodic_grid(2K + 1)``.

    Returns
    -------
    np.ndarray
        Complex ``c_0 ... c_K``.
    """
    return np.fft.rfft(values, axis=-1) / values.shape[-1]


def spectrum_support(
    frequencies: np.ndarray, coefficients: np.ndarray, tolerance: float
) -> list[int]:
    """Return the sorted frequencies whose coefficient magnitude exceeds ``tolerance``."""
    return sorted(int(n) for n in frequencies[np.abs(coefficients) > tolerance])


def largest_coefficient_outside(
    frequencies: np.ndarray, coefficients: np.ndarray, allowed: Iterable[int]
) -> float:
    """Return ``max |c_n|`` over frequencies not in ``allowed`` (0 if none)."""
    allowed_set = {int(n) for n in allowed}
    outside = np.array([n not in allowed_set for n in frequencies])
    if not outside.any():
        return 0.0
    return float(np.abs(coefficients[outside]).max())


def predicted_spectrum(eigenvalues: Sequence[float], n_layers: int) -> np.ndarray:
    """Return ``Omega = {Lambda_k - Lambda_j}`` for ``L`` repetitions of one generator.

    Parameters
    ----------
    eigenvalues : Sequence[float]
        Eigenvalues ``lambda_1 ... lambda_d`` of the encoding generator.
    n_layers : int
        Number of sequential encoding layers ``L`` (paper Eq. 9).

    Returns
    -------
    np.ndarray
        Sorted unique frequencies.
    """
    sums = {
        round(sum(combination), _FREQUENCY_DECIMALS)
        for combination in itertools.product(eigenvalues, repeat=n_layers)
    }
    differences = {round(a - b, _FREQUENCY_DECIMALS) for a in sums for b in sums}
    return np.array(sorted(differences))


def spectrum_size(spectrum: np.ndarray) -> int:
    """Return ``K = (|Omega| - 1) / 2`` for a symmetric spectrum containing 0."""
    return (len(spectrum) - 1) // 2


def paper_size_bound(dimension: int, n_layers: int) -> float:
    """Return the paper's bound ``K <= d^(2L) / 2 - 1`` (Eq. 12)."""
    return dimension ** (2 * n_layers) / 2 - 1


def distinct_sum_size_bound(dimension: int, n_layers: int) -> int:
    """Return ``K <= m (m - 1) / 2`` with ``m = d^L`` distinct values of ``Lambda``.

    ``Lambda`` takes at most ``m = d^L`` values, so ``Omega`` has at most
    ``m (m - 1)`` nonzero differences (LOG.md ambiguity A5). This is never
    looser than Eq. 12.
    """
    distinct_sums = dimension**n_layers
    return distinct_sums * (distinct_sums - 1) // 2
