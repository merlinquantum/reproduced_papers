from __future__ import annotations

import numpy as np
from q_random_kitchen_sinks.lib.approx_kernel import (  # noqa: E402
    classical_features,
    sample_random_features,
    transform_inputs,
)


def test_sample_random_features_shapes() -> None:
    w, b = sample_random_features(r=3, seed=123)
    assert w.shape == (3, 2)
    assert b.shape == (3,)


def test_transform_inputs_shape() -> None:
    points = np.array([[0.0, 1.0], [1.0, 0.0]])
    w = np.ones((2, 2))
    b = np.zeros((2,))
    proj = transform_inputs(points, w, b, r=2, gamma=1.5)
    assert proj.shape == (2, 2)


def test_classical_features_shape() -> None:
    proj = np.zeros((4, 5))
    feats = classical_features(proj)
    assert feats.shape == (4, 5)
