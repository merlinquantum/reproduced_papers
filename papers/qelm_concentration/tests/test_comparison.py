"""Phase 5 fair-comparison checks: matched dimensions, input scaling and seeding."""

from __future__ import annotations

import common  # noqa: F401  (puts the paper package on sys.path)
import numpy as np
from qelm_concentration.lib.baselines import (
    random_fourier_features,
    random_tanh_features,
)
from qelm_concentration.lib.data import make_landsat_split
from qelm_concentration.lib.qelm_merlin import PhotonicArchitecture
from test_smoke import _config, _skip_without_landsat


def _readout_dimensions(arch: PhotonicArchitecture) -> tuple[int, int]:
    # w1: one click probability per mode; w2: 3 non-trivial outcomes per disjoint pair.
    return arch.n_modes, 3 * len(arch.disjoint_pairs())


def test_matched_dimensions_cover_every_compared_readout():
    cfg = _config("landsat_classical_standard.json")
    matched = set(cfg["landsat_classical"]["matched_elm"]["n_features"])
    qubit_dims = {3 * n for n in (16, 24)} | {15 * n // 2 for n in (16, 24)}
    photonic_dims = set()
    for name in (
        "landsat_readouts_merlin_m16.json",
        "landsat_readouts_merlin_m24.json",
    ):
        arch = PhotonicArchitecture.from_config(_config(name)["merlin_reservoir"])
        photonic_dims |= set(_readout_dimensions(arch))
    assert photonic_dims == {16, 24, 36}
    assert qubit_dims | photonic_dims == matched


def test_standard_config_reaches_the_split_and_matches_merlin_scaling():
    baseline = _config("landsat_classical_standard.json")
    merlin = _config("landsat_readouts_merlin_m16.json")
    _skip_without_landsat(baseline)
    assert (
        baseline["dataset"]["landsat"]["input_scaling"]
        == merlin["dataset"]["landsat"]["input_scaling"]
        == "standard"
    )
    assert (
        baseline["landsat_classical"]["split_seeds"]
        == merlin["landsat_readouts"]["split_seeds"]
    )
    baseline_split = make_landsat_split(baseline, 0)
    merlin_split = make_landsat_split(merlin, 0)
    np.testing.assert_array_equal(baseline_split.train_y, merlin_split.train_y)
    np.testing.assert_allclose(baseline_split.train_x, merlin_split.train_x)
    np.testing.assert_allclose(baseline_split.train_x.mean(axis=0), 0.0, atol=1e-9)


def test_random_features_are_seeded_and_shaped():
    rng = np.random.default_rng(0)
    train, test = rng.normal(size=(20, 36)), rng.normal(size=(7, 36))
    for make, scale in ((random_fourier_features, 2.0), (random_tanh_features, 1.0)):
        first = make(train, test, 24, scale, seed=5)
        second = make(train, test, 24, scale, seed=5)
        other = make(train, test, 24, scale, seed=6)
        assert first[0].shape == (20, 24) and first[1].shape == (7, 24)
        np.testing.assert_array_equal(first[0], second[0])
        assert not np.allclose(first[0], other[0])
