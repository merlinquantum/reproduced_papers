"""Fréchet Distance metric used in the paper.

Implements FD on flattened image samples (matching the QPatchGAN/MosaiQ
convention also used in the LatentQGAN paper, equation 10).

FD = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 * sqrt(Sigma_r * Sigma_g))

The sqrt of the matrix product is computed via scipy.linalg.sqrtm.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def _gaussian_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x: (N, D) -> (mu, Sigma)."""
    mu = x.mean(axis=0)
    Sigma = np.cov(x, rowvar=False)
    return mu, Sigma


def frechet_distance(real: np.ndarray, fake: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the Fréchet Distance between two sample sets.

    Parameters
    ----------
    real, fake : (N, D) arrays of flattened image samples.
    """
    if real.ndim > 2:
        real = real.reshape(real.shape[0], -1)
    if fake.ndim > 2:
        fake = fake.reshape(fake.shape[0], -1)
    mu_r, Sigma_r = _gaussian_stats(real)
    mu_g, Sigma_g = _gaussian_stats(fake)
    diff = mu_r - mu_g
    # Stabilise Sigma to ensure positive semi-definite.
    offset = eps * np.eye(Sigma_r.shape[0])
    covmean = linalg.sqrtm((Sigma_r + offset) @ (Sigma_g + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = diff @ diff + np.trace(Sigma_r + Sigma_g - 2 * covmean)
    return float(fd)
