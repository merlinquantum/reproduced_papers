"""Classical feature maps for the fair baselines (our addition; the paper has none)."""

from __future__ import annotations

import numpy as np


def raw_window_features(windows: np.ndarray) -> np.ndarray:
    """Flattened window ``(N, w * d)``: the same inputs QuaRK sees, without a reservoir."""
    return windows.reshape(windows.shape[0], -1)


def esn_features(
    windows: np.ndarray,
    *,
    units: int,
    spectral_radius: float,
    leak: float,
    input_scale: float,
    seed: int,
) -> np.ndarray:
    """Final state of a leaky echo state network driven by each window.

    ``h_t = (1 - leak) h_{t-1} + leak * tanh(W_in x_t + W h_{t-1})`` from ``h = 0`` at
    the start of every window, mirroring QuaRK's fixed initial state per window.
    ``W_in`` is uniform in ``(-input_scale, input_scale)``; ``W`` is Gaussian, rescaled
    to the given spectral radius.

    Returns
    -------
    numpy.ndarray
        Features of shape ``(N, units)``.
    """
    rng = np.random.default_rng(seed)
    input_dim = windows.shape[-1]
    input_weights = rng.uniform(-input_scale, input_scale, size=(input_dim, units))
    recurrent = rng.normal(size=(units, units))
    recurrent *= spectral_radius / np.max(np.abs(np.linalg.eigvals(recurrent)))
    state = np.zeros((windows.shape[0], units))
    for t in range(windows.shape[1]):
        update = np.tanh(windows[:, t] @ input_weights + state @ recurrent)
        state = (1.0 - leak) * state + leak * update
    return state
