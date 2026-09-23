"""Photonic circuits used as latent-distribution sources.

Two families are provided, matching Bacarreza et al., arXiv:2508.19857:

* ``haar_circuit`` -- an unstructured Haar-random interferometer on ``m`` modes.
  This is the "random optical circuit" used for the synthetic quantum dataset
  and for most of the QM9 latents.
* ``delay_line_circuit`` -- an experimentally realistic time-bin architecture
  built from sequential fibre loops with delays in a fixed ratio ("1-1" or
  "1-3-9").  This is the abstraction of the ORCA PT-series hardware used in the
  paper: a loop of delay ``d`` couples time-bin ``i`` with time-bin ``i + d``
  through a programmable variable beam splitter.
"""

from __future__ import annotations

import numpy as np
import perceval as pcvl

__all__ = [
    "haar_unitary",
    "delay_line_unitary",
    "haar_circuit",
    "delay_line_circuit",
    "to_circuit",
    "DELAY_CONFIGS",
]

#: Loop configurations reported in the paper.
DELAY_CONFIGS: dict[str, tuple[int, ...]] = {
    "1-1": (1, 1),
    "1-3-9": (1, 3, 9),
}


def haar_unitary(m: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an ``m x m`` unitary from the Haar measure (QR of a Ginibre matrix)."""
    z = (rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _bs_embed(m: int, i: int, j: int, theta: float, phi: float) -> np.ndarray:
    """Unitary of a variable beam splitter (+ phase) acting on modes ``i``/``j``."""
    u = np.eye(m, dtype=complex)
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    u[i, i] = c
    u[j, j] = c
    u[i, j] = 1j * s * np.exp(-1j * phi)
    u[j, i] = 1j * s * np.exp(1j * phi)
    return u


def delay_line_unitary(
    m: int,
    rng: np.random.Generator,
    config: str | tuple[int, ...] = "1-3-9",
) -> np.ndarray:
    """Unitary of a sequential-loop (time-bin) interferometer on ``m`` bins.

    Each loop of delay ``d`` applies programmable couplings between bins
    ``(i, i + d)`` for ``i = 0 .. m - 1 - d``, applied in increasing ``i`` so
    that the loop is traversed once per time bin.  Coupling angles are drawn
    uniformly, which is what "randomly initialised" means in the paper.

    Parameters
    ----------
    m : int
        Number of time bins (== latent dimension).
    rng : numpy.random.Generator
        Source of randomness for the coupling angles.
    config : str | tuple[int, ...]
        Either a key of :data:`DELAY_CONFIGS` or an explicit tuple of delays.

    Returns
    -------
    numpy.ndarray
        The ``m x m`` unitary implemented by the loop cascade.
    """
    delays = DELAY_CONFIGS[config] if isinstance(config, str) else tuple(config)
    u = np.eye(m, dtype=complex)
    for d in delays:
        if d >= m:
            continue
        for i in range(m - d):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            phi = rng.uniform(0.0, 2.0 * np.pi)
            u = _bs_embed(m, i, i + d, theta, phi) @ u
    return u


def to_circuit(u: np.ndarray) -> pcvl.Circuit:
    """Wrap a numpy unitary into a Perceval circuit usable by MerLin."""
    m = u.shape[0]
    return pcvl.Circuit(m).add(0, pcvl.Unitary(pcvl.Matrix(u)))


def haar_circuit(m: int, seed: int) -> pcvl.Circuit:
    """Perceval circuit implementing a Haar-random ``m``-mode interferometer."""
    return to_circuit(haar_unitary(m, np.random.default_rng(seed)))


def delay_line_circuit(m: int, seed: int, config: str = "1-3-9") -> pcvl.Circuit:
    """Perceval circuit implementing a randomly-set time-bin loop cascade."""
    return to_circuit(delay_line_unitary(m, np.random.default_rng(seed), config))
