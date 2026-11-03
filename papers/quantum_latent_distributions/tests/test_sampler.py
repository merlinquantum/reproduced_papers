"""The boson sampler against MerLin's exact Fock-space simulation."""

from __future__ import annotations

from collections import Counter

import numpy as np
from common import PROJECT_DIR  # noqa: F401 - puts the project on sys.path
from lib.circuits import haar_unitary
from lib.latents import exact_distribution, sample_boson


def _empirical(samples, keys):
    counts = Counter(map(tuple, samples.tolist()))
    hist = [counts.get(tuple(int(x) for x in key), 0) for key in keys]
    return np.asarray(hist, dtype=float) / len(samples)


def test_exact_distribution_covers_the_full_fock_space():
    """C(m + n - 1, n) = C(7, 3) = 35 states for 3 photons in 5 modes.

    MerLin's default computation space is UNBUNCHED, which would return only the
    C(5, 3) = 10 collision-free states, renormalised to sum to 1.
    """
    unitary = haar_unitary(5, np.random.default_rng(0))
    keys, probs = exact_distribution(unitary, 3)
    assert len(keys) == 35
    assert abs(probs.sum() - 1.0) < 1e-5


def test_sampler_converges_to_the_exact_distribution():
    unitary = haar_unitary(5, np.random.default_rng(0))
    keys, exact = exact_distribution(unitary, 3)
    coarse = (
        0.5
        * np.abs(
            _empirical(sample_boson(unitary, 3, 5_000, seed=1), keys) - exact
        ).sum()
    )
    fine = (
        0.5
        * np.abs(
            _empirical(sample_boson(unitary, 3, 80_000, seed=1), keys) - exact
        ).sum()
    )
    assert fine < coarse
    assert fine < 0.02
