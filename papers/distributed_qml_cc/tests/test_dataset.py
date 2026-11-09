from __future__ import annotations

import numpy as np
from common import PROJECT_DIR  # noqa: F401  (sets up sys.path)
from lib.data import SyntheticDatasetConfig, build_synthetic_dataset


def test_synthetic_shapes_and_balance():
    x_tr, y_tr, x_va, y_va, info = build_synthetic_dataset()
    assert x_tr.shape == (1536, 8)
    assert x_va.shape == (512, 8)
    assert y_tr.shape == (1536,)
    assert y_va.shape == (512,)
    assert set(np.unique(y_tr)).issubset({-1, 1})
    # Labels should be roughly balanced.
    pos = float((y_tr == 1).mean())
    assert 0.45 < pos < 0.55
    # Maximum Pearson coefficient should remain modest.
    assert info["pearson_max_abs"] < 0.4


def test_synthetic_reproducible():
    cfg = SyntheticDatasetConfig(seed=123)
    a = build_synthetic_dataset(cfg)
    b = build_synthetic_dataset(cfg)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
