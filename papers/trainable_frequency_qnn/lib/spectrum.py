"""Frequency analysis of single-input models (paper Sec. IV-V, Fig. 2c)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch


def accessible_frequencies(multipliers: list[float], decimals: int = 9) -> list[float]:
    """Return the positive frequencies a QNN with these encoding gates can carry.

    Each ``R_y(w_m x)`` gate contributes eigenvalues ``+-w_m / 2``; a QNN output
    contains every gap of the summed generator (paper Sec. IV), i.e. every
    ``sum_m c_m w_m`` with ``c_m`` in ``{-1, 0, 1}``.

    Parameters
    ----------
    multipliers : list of float
        Effective gate multipliers ``gamma_m * w_m``.
    decimals : int
        Rounding used to merge degenerate frequencies. Default value is 9.

    Returns
    -------
    list of float
        Sorted unique positive frequencies.
    """
    values = set()
    for coefficients in itertools.product((-1, 0, 1), repeat=len(multipliers)):
        value = round(
            abs(sum(c * w for c, w in zip(coefficients, multipliers))), decimals
        )
        if value > 0:
            values.add(value)
    return sorted(values)


def prediction_spectrum(
    model: torch.nn.Module,
    half_width_pi: float,
    n_points: int,
    dtype: torch.dtype,
    device,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the DFT magnitude of a trained model's prediction.

    The training window [-4 pi, 4 pi] gives an angular-frequency resolution of
    only 0.25, too coarse to separate 1.0 from 1.2. The model is a function of
    every real x, so we evaluate it on a much wider window to resolve its own
    spectrum. This is a property of the model, not of the training data.

    Returns
    -------
    tuple of numpy.ndarray
        Angular frequencies and the normalised DFT magnitudes ``|F| / n``.
    """
    half_width = half_width_pi * math.pi
    grid = torch.linspace(-half_width, half_width, n_points, dtype=dtype, device=device)
    outputs = []
    with torch.no_grad():
        for start in range(0, n_points, chunk_size):
            outputs.append(model(grid[start : start + chunk_size, None]).cpu())
    signal = torch.cat(outputs).numpy()
    spacing = (2 * half_width) / (n_points - 1)
    magnitudes = np.abs(np.fft.rfft(signal)) / n_points
    angular_frequencies = 2 * math.pi * np.fft.rfftfreq(n_points, d=spacing)
    return angular_frequencies, magnitudes


def dominant_frequencies(
    angular_frequencies: np.ndarray, magnitudes: np.ndarray, count: int
) -> list[float]:
    """Return the ``count`` largest local maxima of a magnitude spectrum (excluding 0)."""
    interior = np.arange(1, len(magnitudes) - 1)
    is_peak = (magnitudes[interior] >= magnitudes[interior - 1]) & (
        magnitudes[interior] >= magnitudes[interior + 1]
    )
    peaks = interior[is_peak]
    strongest = peaks[np.argsort(magnitudes[peaks])[::-1][:count]]
    return sorted(float(angular_frequencies[i]) for i in strongest)
