"""E4: RPS game rules, data generator, feature map and loaders (Sec. 5, App. F)."""

from __future__ import annotations

import itertools
import math
from copy import deepcopy

import numpy as np
import pytest
from common import load_runtime_ready_config
from contextuality_inductive_bias.lib import data as D
from contextuality_inductive_bias.lib import rps_game as G

BEATS = {"R": "S", "P": "R", "S": "P"}
SPECIAL = "RPS"  # special action of players 1, 2, 3


def _score_from_text(profile: str) -> list[int]:
    """l_k(a) written directly from the rules quoted in Sec. 5.1."""
    scores = []
    for k, mine in enumerate(profile):
        score = 0
        for j, theirs in enumerate(profile):
            if j == k:
                continue
            if mine != theirs:
                score += (
                    1 if BEATS[mine] == theirs else -1 if BEATS[theirs] == mine else 0
                )
            elif mine == SPECIAL[k]:
                score += 1
            elif theirs == SPECIAL[j]:
                score -= 1
        scores.append(score)
    return scores


@pytest.mark.parametrize(
    "profile", ["".join(p) for p in itertools.product("RPS", repeat=3)]
)
def test_deterministic_profiles_follow_rules(profile):
    strategy = G.deterministic_strategy(profile)
    expected = (1 + np.array(_score_from_text(profile)) / 2) / 2
    np.testing.assert_allclose(G.win_probabilities(strategy), expected, atol=0)
    assert G.score_differences(tuple("RPS".index(c) for c in profile)).sum() == 0


def test_extremal_profiles_give_hexagon_vertices():
    for profile, vertex in zip(G.EXTREMAL_PROFILES, G.HEXAGON_VERTICES):
        np.testing.assert_array_equal(
            G.win_probabilities(G.deterministic_strategy(profile)), vertex
        )
    # The paper's example (Sec. 5.2): RSS gives v_1 = (1, 0, 1/2).
    np.testing.assert_array_equal(
        G.win_probabilities(G.deterministic_strategy("RSS")), [1.0, 0.0, 0.5]
    )


def test_zero_sum_bias_holds_for_random_strategies():
    strategies = D.generate_strategies(np.random.RandomState(0), 5000)
    np.testing.assert_allclose(
        G.win_probabilities(strategies).sum(axis=1), 1.5, atol=1e-14
    )


def test_generator_reproduces_author_arrays():
    cfg = load_runtime_ready_config()
    strategies, labels, provenance = D.load_author_training(D.author_data_dir(cfg))
    regenerated_x, regenerated_y = D.generate_labelled(999, 1500)
    np.testing.assert_array_equal(regenerated_x, strategies)
    np.testing.assert_array_equal(regenerated_y, labels)
    assert provenance["sha256"]["Xdata.npy"].startswith("50a880fb")


def test_feature_map_matches_eq_93():
    strategy = np.array([[[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.4, 0.4, 0.2]]])
    features = D.compress_strategies(strategy, math.pi / 2)
    expected = (
        math.pi / 2 * np.array([[0.2, 0.6, 0.2, 0.3 - 0.5, 0.3 - 0.1, 0.4 - 0.4]])
    )
    np.testing.assert_allclose(features, expected)


@pytest.mark.parametrize(
    "protocol,train_scale,test_scale",
    [
        ("paper", math.pi / 2, math.pi / 2),
        ("code", math.pi / 2, 1.0),
        ("fork", 1.0, 1.0),
    ],
)
def test_protocol_scales(protocol, train_scale, test_scale):
    cfg = load_runtime_ready_config()
    cfg["data"]["protocol"] = protocol
    cfg["data"]["test"]["n"] = 50
    train = D.load_split(cfg, "train")
    test = D.load_split(cfg, "test")
    assert train.input_scale == train_scale and test.input_scale == test_scale
    np.testing.assert_allclose(
        test.features.numpy(), D.compress_strategies(test.strategies, test_scale)
    )


def test_build_dataloader_one_batch():
    cfg = load_runtime_ready_config()
    batch = next(iter(D.build_dataloader(cfg, split="train")))
    assert batch[0].shape == (1500, 6)
    assert batch[1].shape == (1500, 3)
    assert batch[2].shape == (1500, 3)
    cfg["data"]["test"]["n"] = 100
    test_batch = next(iter(D.build_dataloader(cfg, split="test")))
    assert test_batch[0].shape == (100, 6) and test_batch[1].shape == (100, 0)


def test_test_set_is_seeded_and_shared():
    cfg = load_runtime_ready_config()
    cfg["data"]["test"]["n"] = 200
    first, second = D.load_split(cfg, "test"), D.load_split(cfg, "test")
    np.testing.assert_array_equal(first.strategies, second.strategies)
    other = deepcopy(cfg)
    other["data"]["test"]["seed"] += 1
    assert not np.array_equal(first.strategies, D.load_split(other, "test").strategies)


def test_generated_training_requires_seed():
    cfg = load_runtime_ready_config()
    cfg["data"]["train"]["source"] = "generated"
    with pytest.raises(ValueError, match="train.seed"):
        D.load_split(cfg, "train")


def test_unknown_protocol_fails():
    cfg = load_runtime_ready_config()
    cfg["data"]["protocol"] = "other"
    with pytest.raises(ValueError, match="protocol"):
        D.load_split(cfg, "test")


def test_runtime_default_data_root_is_redirected(monkeypatch):
    """The shared runtime resolves its default to papers/<paper>/data (see TODO in lib/data.py)."""
    cfg = load_runtime_ready_config()
    cfg["data_root"] = str(D._PROJECT_DIR / "data")
    assert (
        D.author_data_dir(cfg)
        == D._REPO_ROOT / "data" / D.PAPER_NAME / D.DATASET_DIRNAME
    )


def test_explicit_data_root_is_used_as_given(tmp_path):
    cfg = load_runtime_ready_config()
    cfg["data_root"] = str(tmp_path)
    assert D.author_data_dir(cfg) == tmp_path / D.PAPER_NAME / D.DATASET_DIRNAME
    with pytest.raises(FileNotFoundError, match="SHA256SUMS"):
        D.load_split(cfg, "train")
