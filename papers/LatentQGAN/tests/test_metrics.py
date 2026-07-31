"""Sanity tests for the Fréchet Distance implementation."""

from __future__ import annotations

import numpy as np

from lib.metrics import frechet_distance


def test_zero_distance_same_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 16))
    fd = frechet_distance(x, x)
    assert abs(fd) < 1e-3


def test_positive_distance_different_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 16))
    y = rng.normal(size=(64, 16)) + 2.0  # shifted
    fd = frechet_distance(x, y)
    assert fd > 0
