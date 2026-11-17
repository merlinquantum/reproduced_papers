from __future__ import annotations

import common  # noqa: F401
import numpy as np
import pytest
from lib.data import (
    build_dataset,
    make_nonlinear_1d,
    partition_indices,
)
from sklearn.svm import SVC

MOONS = {
    "name": "moons_kmeans",
    "n_samples": 200,
    "noise": 0.1,
    "data_seed": 0,
    "n_clusters": 15,
    "kmeans_seed": 0,
    "assignment": "arc_order",
    "assignment_seed": 0,
}


def test_nonlinear_1d_matches_fig5a():
    dataset = make_nonlinear_1d(15)
    assert dataset.outcome_index.shape == (15,)
    positives = set(np.flatnonzero(dataset.labels == 1).tolist())
    assert positives == {0, 1, 2, 3, 11, 12, 13, 14}
    assert np.sum(dataset.labels == 1) == 8 and np.sum(dataset.labels == -1) == 7


def test_nonlinear_1d_is_not_linearly_separable():
    dataset = make_nonlinear_1d(15)
    model = SVC(kernel="linear", C=1e6).fit(dataset.features, dataset.labels)
    assert model.score(dataset.features, dataset.labels) < 1.0


def test_moons_shapes_balance_and_determinism():
    first = build_dataset(MOONS)
    second = build_dataset(MOONS)
    assert first.features.shape == (200, 2)
    assert first.labels.shape == first.outcome_index.shape == (200,)
    assert np.sum(first.labels == 1) == 100
    assert np.array_equal(first.outcome_index, second.outcome_index)
    assert set(first.outcome_index.tolist()) == set(range(15))
    assert sorted(first.extras["assignment"].tolist()) == list(range(15))
    assert 0.5 <= first.extras["weighted_purity"] <= 1.0


def test_arc_order_is_invariant_to_kmeans_label_permutation():
    # Different K-means seeds relabel clusters; arc_order outcome indices of the
    # points must not depend on the labels when the partition is the same.
    reference = build_dataset(MOONS)
    relabelled = build_dataset({**MOONS, "kmeans_seed": 1})
    same_partition = all(
        len(set(relabelled.extras["cluster"][reference.extras["cluster"] == c])) == 1
        for c in range(15)
    )
    if same_partition:
        assert np.array_equal(reference.outcome_index, relabelled.outcome_index)


def test_missing_dataset_key_fails_clearly():
    config = dict(MOONS)
    del config["noise"]
    with pytest.raises(ValueError, match="missing keys"):
        build_dataset(config)


def test_partitions_sizes_and_reuse():
    train, test = partition_indices(15, 0.2, 7)
    assert len(train) == 12 and len(test) == 3
    again_train, again_test = partition_indices(15, 0.2, 7)
    assert np.array_equal(train, again_train) and np.array_equal(test, again_test)
    train, test = partition_indices(200, 0.2, 0)
    assert len(train) == 160 and len(test) == 40
    assert not set(train) & set(test)


def test_data_root_resolves_to_repository_data(monkeypatch):
    # A paper-local implementation.py would make the shared runtime treat the
    # paper folder as the repository root (regression found in Phase 02).
    from common import PROJECT_DIR, REPO_ROOT
    from lib.data import digitized_kernels_path

    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.chdir(PROJECT_DIR)
    assert not (PROJECT_DIR / "implementation.py").exists()
    path = digitized_kernels_path({"data_root": None})
    assert (
        path
        == REPO_ROOT
        / "data"
        / "adaptive_boson_sampling"
        / "digitized"
        / "figure_kernels.npz"
    )


def test_x_order_sorts_clusters_by_centroid_x():
    dataset = build_dataset({**MOONS, "assignment": "x_order"})
    centroids = dataset.extras["centroids"]
    ordered_x = centroids[np.argsort(dataset.extras["assignment"]), 0]
    assert np.all(np.diff(ordered_x) >= 0)


def test_centroid_rank_x_equals_x_order_and_unknown_rule_fails():
    x_order = build_dataset({**MOONS, "assignment": "x_order"})
    ranked = build_dataset({**MOONS, "assignment": "centroid_rank", "rank_rule": "x"})
    assert np.array_equal(x_order.outcome_index, ranked.outcome_index)
    with pytest.raises(ValueError, match="rank_rule"):
        build_dataset({**MOONS, "assignment": "centroid_rank", "rank_rule": "z"})
