"""Evaluation metrics.

``l1_to_nearest_integer`` is the paper's Table-I metric for discrete-valued
datasets.  The mixture-of-Gaussians metrics quantify the paper's qualitative
claim that "the quantum distribution produces the model that interpolates the
least between the modes".
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "l1_to_nearest_integer",
    "mode_assignment",
    "mode_coverage",
    "interpolation_rate",
    "mmd_rbf",
]


def l1_to_nearest_integer(samples: torch.Tensor) -> float:
    """Mean absolute distance from each generated coordinate to the nearest integer.

    Lower is better: it measures how well the generator has learned the
    *discrete* nature of a photon-count dataset.
    """
    x = samples.detach()
    return float((x - x.round()).abs().mean())


def mode_assignment(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the nearest mixture component for every sample."""
    d = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return d.argmin(axis=1)


def mode_coverage(samples: np.ndarray, centers: np.ndarray, radius: float) -> dict:
    """Per-mode capture counts and the number of modes actually covered.

    A mode counts as covered when it captures at least ``1 / (10 * n_modes)``
    of the generated mass, the usual convention for mode-collapse studies.
    """
    assign = mode_assignment(samples, centers)
    dist = np.linalg.norm(samples - centers[assign], axis=1)
    inside = dist <= radius
    counts = np.bincount(assign[inside], minlength=len(centers))
    frac = counts / max(len(samples), 1)
    return {
        "counts": counts.tolist(),
        "fraction": frac.tolist(),
        "n_modes_covered": int((frac > 1.0 / (10 * len(centers))).sum()),
        "captured_fraction": float(inside.mean()),
    }


def interpolation_rate(
    samples: np.ndarray, centers: np.ndarray, radius: float
) -> float:
    """Fraction of generated points that lie in no mode's neighbourhood.

    These are the points "in between the modes" -- the artefact a factorisable
    latent distribution encourages and that the paper reports the quantum
    latent suppresses.
    """
    assign = mode_assignment(samples, centers)
    dist = np.linalg.norm(samples - centers[assign], axis=1)
    return float((dist > radius).mean())


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigmas=(0.25, 0.5, 1.0, 2.0)) -> float:
    """Unbiased multi-bandwidth RBF MMD^2 between two point clouds."""

    def k(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return sum(np.exp(-d2 / (2 * s**2)) for s in sigmas) / len(sigmas)

    n, m = len(x), len(y)
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())
