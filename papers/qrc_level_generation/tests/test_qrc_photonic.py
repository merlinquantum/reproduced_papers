from __future__ import annotations

import math
import sys

import numpy as np
import pytest
from common import PROJECT_DIR

# The paper package lives under papers/, which pytest does not reliably put
# on sys.path at collection time; insert it explicitly.
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))

pytest.importorskip("merlin", reason="merlinquantum not installed")

from qrc_level_generation.lib.qrc_photonic import PhotonicQRC  # noqa: E402


def make_reservoir(**kwargs):
    defaults = {"num_features": 4, "n_modes": 6, "n_photons": 3, "seed": 0}
    defaults.update(kwargs)
    return PhotonicQRC(**defaults)


def test_output_dim_matches_unbunched_subspace():
    res = make_reservoir()
    # UNBUNCHED with n photons in m modes has C(m, n) outcomes.
    assert res.output_dim == math.comb(6, 3)


def test_reservoir_is_frozen():
    res = make_reservoir()
    assert all(not p.requires_grad for p in res._layer.parameters())


def test_step_returns_probability_vector():
    res = make_reservoir()
    h = res.initial_hidden()
    probs = res.step(1, h)
    assert probs.shape == (res.output_dim,)
    assert (probs >= 0).all()
    assert probs.sum() == pytest.approx(1.0)


def test_same_seed_same_reservoir():
    h = make_reservoir(seed=7).step(2, make_reservoir(seed=7).initial_hidden())
    h2 = make_reservoir(seed=7).step(2, make_reservoir(seed=7).initial_hidden())
    np.testing.assert_allclose(h, h2)
