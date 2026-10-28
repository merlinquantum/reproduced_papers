"""Data recipes, training-loop semantics and baseline samplers."""

from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401
from photonic_MMD_generative.lib import baselines, data, training


def test_input_modes_follow_original_code():
    assert data.input_modes(100, 10, "middle_alternating") == list(range(40, 59, 2))
    assert data.input_modes(256, 16, "middle_compact") == list(range(120, 136))
    assert data.input_modes(8, 2, "beginning") == [0, 1]


def test_recipe_rejects_missing_keys_and_drops_unused_ones():
    with pytest.raises(KeyError, match="seed"):
        data.recipe(
            {"name": "sushi_top10", "m": 100, "n": 10, "n_train": 1, "n_test": 1}
        )
    spec = data.recipe(
        {
            "name": "bioinformatics",
            "m": 100,
            "n": 10,
            "number_samples": 10,
            "frac_train": 0.7,
            "n_train": 400,
        }
    )
    assert "n_train" not in spec


def test_random_dataset_build_and_reload(tmp_path):
    cfg = {
        "data_root": str(tmp_path),
        "device": "cpu",
        "dataset": {
            "name": "random_fixed_weight",
            "m": 8,
            "n": 3,
            "n_train": 50,
            "n_test": 20,
            "seed": 1,
        },
    }
    first = data.load_dataset(cfg)
    second = data.load_dataset(cfg)
    assert first["train"].shape == (50, 8) and (first["train"].sum(axis=1) == 3).all()
    assert first["manifest"]["train_sha256"] == second["manifest"]["train_sha256"]
    batch = next(iter(data.build_dataloader(cfg, "test")))
    assert tuple(batch[0].shape) == (20, 8)


def test_convergence_rule_matches_original_window_logic():
    flat = [1.0] * 20
    assert not training.converged(flat[:10], 5)
    assert training.converged(flat, 5)
    decreasing = list(np.linspace(1.0, 0.0, 20))
    assert not training.converged(decreasing, 5)


def test_baseline_samplers_have_fixed_weight():
    X_train = baselines.uniform_fixed_weight(20, 4, 500, seed=0)
    for X in (
        baselines.uniform_fixed_weight(20, 4, 100, seed=1),
        baselines.independent_marginal_fixed_weight(X_train, 100, seed=2),
    ):
        assert (X.sum(axis=1) == 4).all() and X.max() == 1


def test_distinguishable_samples_follow_single_photon_marginals():
    U = np.eye(4)
    X = baselines.distinguishable_samples(U, [0, 2], 50, seed=0)
    assert (X == np.array([1, 0, 1, 0])).all()
