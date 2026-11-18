"""Target datasets used by the reproduction.

Two families, matching sections IV.A and Appendix C of the paper:

* a 2D mixture of Gaussians (Fig. 2), and
* discrete-valued synthetic datasets (Table I): photon-count patterns from a
  boson sampler, and uniform bit strings.

The paper does not specify the mixture-of-Gaussians parameters anywhere -- the
number of components, their positions and their width are absent from the text.
The authors' released code (``src/gaussians_utils.py`` in
`orcacomputing/quantum-latent-distributions
<https://github.com/orcacomputing/quantum-latent-distributions>`_) does specify
them, and we follow it: **7** blobs on a radius-5 circle with *polar* noise,
``r ~ N(5, 0.2)`` and ``theta ~ N(theta_k, 0.05)``, not isotropic noise on a
radius-2 ring as an earlier version of this reproduction assumed.
"""

from __future__ import annotations

import numpy as np
from lib.circuits import haar_unitary
from lib.latents import sample_boson

__all__ = [
    "DATA_MODES",
    "DATA_PHOTONS",
    "bernoulli_dataset",
    "mixture_centers",
    "mixture_samples",
    "quantum_dataset",
]

#: Section IV.A: "8 identical photons interfering in a 16-channel random optical
#: circuit".
DATA_MODES = 16
DATA_PHOTONS = 8


def mixture_centers(n_components: int, radius: float) -> np.ndarray:  # noqa: D401
    """Return the component means of a ring-shaped 2D Gaussian mixture.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    radius : float
        Radius of the ring the component means sit on.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_components, 2)``.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_components, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def mixture_samples(
    n_samples: int,
    n_components: int,
    radius: float,
    radial_std: float,
    tangential_std: float,
    seed: int,
) -> np.ndarray:
    """Draw samples from the blobs-on-a-circle dataset of the released code.

    Noise is applied in polar coordinates, matching ``SevenGaussians``:
    ``r ~ N(radius, radial_std)`` and ``theta ~ N(theta_k, tangential_std)``.
    This makes the blobs slightly banana-shaped rather than round, which matters
    for the "does the model interpolate between modes" metric.

    Parameters
    ----------
    n_samples : int
        Number of points to draw.
    n_components : int
        Number of blobs.
    radius : float
        Radius of the circle the blob centres sit on.
    radial_std : float
        Standard deviation of the radial noise, in distance units.
    tangential_std : float
        Standard deviation of the angular noise, in radians.
    seed : int
        Seed for the numpy generator.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_samples, 2)``.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_components, size=n_samples)
    theta_k = np.linspace(0.0, 2.0 * np.pi, n_components, endpoint=False)[idx]
    theta = theta_k + rng.normal(scale=tangential_std, size=n_samples)
    r = radius + rng.normal(scale=radial_std, size=n_samples)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1).astype(np.float32)


def quantum_dataset(n_samples: int, seed: int) -> np.ndarray:
    """Photon-count patterns from an independently drawn boson sampler.

    The circuit seed is offset far from the latent seeds so that the latent and
    the data are never produced by the same unitary -- Appendix D: *"For each of
    the 12 experimental runs, we independently drew 3 random unitary matrices:
    one for the quantum latent distribution, one for the non-interfering
    photons, and one for the data."*

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    seed : int
        Run seed; the data circuit uses ``90000 + seed``.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(n_samples, 16)`` holding integer photon counts.
    """
    unitary = haar_unitary(DATA_MODES, np.random.default_rng(90_000 + seed))
    samples = sample_boson(unitary, DATA_PHOTONS, n_samples, seed=90_000 + seed)
    return samples.astype(np.float32)


def bernoulli_dataset(n_samples: int, seed: int) -> np.ndarray:
    """Uniform bit strings on ``{0, 1}^16`` -- the factorisable discrete control.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    seed : int
        Run seed; the generator uses ``91000 + seed``.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(n_samples, 16)``.
    """
    rng = np.random.default_rng(91_000 + seed)
    return rng.integers(0, 2, size=(n_samples, DATA_MODES)).astype(np.float32)
