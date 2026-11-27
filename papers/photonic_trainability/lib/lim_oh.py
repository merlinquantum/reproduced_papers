"""Baseline B1: Lim-Oh classical estimator of product observables (arXiv:2502.12882).

Lim and Oh, Theorem 1 (proof in their SM Sec. S3): for a product input state
and an observable acting on a subsystem ``A``,

    <O_A> = integral d^{2l} alpha q(alpha) X(alpha),
    q(alpha) = |chi_O(alpha)|^2 / (pi^l ||O_A||_2^2),
    X(alpha) = ||O_A||_2^2 chi_rho_A(alpha)^* / chi_O(alpha)^*,

with characteristic functions ``chi_Q(alpha) = Tr[D(alpha) Q]`` and
``chi_rho_A(alpha) = prod_j chi_psi_j((V^dagger alpha')_j)``, where ``alpha'`` pads
``alpha`` with zeros outside ``A`` and ``V`` is their circuit matrix
(``U a_i^+ U^dagger = sum_j V_ji a_j^+``, the transpose of this repository's
paper convention). Sampling ``alpha`` from ``q`` and averaging ``X`` is unbiased;
``Var(X) <= ||O_A||_2^2 ||rho_A||_2^2``.

Here every observable is a photon-number monomial ``prod_i n_i^{a_i}``, a
product of diagonal single-mode operators. ``n^a`` is not Hilbert-Schmidt on the
full Fock space, so each factor is truncated to photon numbers ``0..n_max``
(``n_max`` = total photons, exact on the state). For a diagonal operator
``O = sum_k o_k |k><k|``, ``chi_O(alpha) = e^{-|alpha|^2 / 2} sum_k o_k L_k(|alpha|^2)``
depends only on ``x = |alpha|^2``; with a uniform phase, ``x`` has density
``f(x)^2 / ||O||^2`` (Laguerre orthonormality gives ``int f^2 dx = sum o_k^2``),
sampled here by inverse CDF on a fine grid. Fock inputs have real
characteristic functions, ``chi_|1>(beta) = e^{-|beta|^2/2} (1 - |beta|^2)`` and
``chi_|0>(beta) = e^{-|beta|^2/2}``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .observables import Monomial

GRID_POINTS = 200_001


def laguerre_series(coefficients: Sequence[float], x: np.ndarray) -> np.ndarray:
    """``e^{-x/2} sum_k c_k L_k(x)`` by the three-term Laguerre recurrence."""
    x = np.asarray(x, dtype=np.float64)
    previous, current = np.zeros_like(x), np.ones_like(x)
    total = coefficients[0] * current
    for k in range(1, len(coefficients)):
        previous, current = (
            current,
            ((2 * k - 1 - x) * current - (k - 1) * previous) / k,
        )
        total = total + coefficients[k] * current
    return np.exp(-x / 2) * total


class DiagonalModeSampler:
    """Samples ``alpha`` from ``q`` for one diagonal single-mode operator.

    Parameters
    ----------
    diagonal : sequence of float
        ``o_k`` for ``k = 0..K``.
    """

    def __init__(self, diagonal: Sequence[float]) -> None:
        self.diagonal = np.asarray(diagonal, dtype=np.float64)
        self.norm_squared = float(np.sum(self.diagonal**2))
        degree = len(self.diagonal) - 1
        self.grid = np.linspace(0.0, 4.0 * degree + 80.0, GRID_POINTS)
        density = laguerre_series(self.diagonal, self.grid) ** 2
        cumulative = np.concatenate(
            [[0.0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(self.grid))]
        )
        self.grid_norm_squared = float(cumulative[-1])
        self.cdf = cumulative / cumulative[-1]

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """``size`` complex samples ``alpha = sqrt(x) e^{i phi}``."""
        x = np.interp(rng.random(size), self.cdf, self.grid)
        return np.sqrt(x) * np.exp(2j * np.pi * rng.random(size))

    def characteristic(self, alpha: np.ndarray) -> np.ndarray:
        """Real ``chi_O(alpha)`` of the diagonal operator."""
        return laguerre_series(self.diagonal, np.abs(alpha) ** 2)


def _fock_characteristic(occupation: Sequence[int], beta: np.ndarray) -> np.ndarray:
    """``prod_j chi_{|s_j>}(beta_j)`` for 0/1 occupations; ``beta`` is ``(S, M)``."""
    squared = np.abs(beta) ** 2
    factors = np.exp(-squared / 2)
    occupied = np.asarray(occupation, dtype=bool)
    factors[:, occupied] *= 1.0 - squared[:, occupied]
    return np.prod(factors, axis=1)


def monomial_samples(
    unitary: np.ndarray,
    occupation: Sequence[int],
    monomial: Monomial,
    samples: int,
    rng: np.random.Generator,
    samplers: dict[int, DiagonalModeSampler] | None = None,
) -> np.ndarray:
    """``samples`` unbiased Lim-Oh draws of ``X`` for ``<prod_i n_i^{a_i}>``.

    Parameters
    ----------
    unitary : numpy.ndarray
        Interferometer in this repository's paper convention.
    occupation : sequence of int
        Collision-free Fock input (entries 0 or 1).
    monomial : Monomial
        Photon-number monomial.
    samples : int
        Number of estimator draws.
    rng : numpy.random.Generator
        Source of randomness.
    samplers : dict | None
        Cache ``exponent -> DiagonalModeSampler`` reused across calls.
        Default value is None.

    Returns
    -------
    numpy.ndarray
        ``(samples,)`` real estimator values with mean ``<M_alpha>``.

    Raises
    ------
    ValueError
        If the input is not collision-free.
    """
    if max(occupation) > 1:
        raise ValueError("Lim-Oh estimator here supports 0/1 Fock inputs only")
    photons = int(sum(occupation))
    samplers = {} if samplers is None else samplers
    modes = unitary.shape[0]
    padded = np.zeros((samples, modes), dtype=np.complex128)
    weight = np.ones(samples)
    for mode, exponent in monomial:
        if exponent not in samplers:
            samplers[exponent] = DiagonalModeSampler(
                [float(k**exponent) for k in range(photons + 1)]
            )
        sampler = samplers[exponent]
        alpha = sampler.sample(samples, rng)
        padded[:, mode] = alpha
        weight = weight * sampler.norm_squared / sampler.characteristic(alpha)
    # V = U^T, so (V^dagger alpha')_j = sum_k conj(U_jk) alpha'_k.
    beta = padded @ np.conj(unitary).T
    return weight * _fock_characteristic(occupation, beta)
