"""Tests for the Phase 5 degree-feature control (claim C3)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

PAPER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from lib import runner  # noqa: E402
from lib.classical_baseline import degree_one_hot_features  # noqa: E402
from lib.data import LadderConcatDataset  # noqa: E402


def _load_config(name: str) -> dict:
    return json.loads((PAPER_ROOT / "configs" / f"{name}.json").read_text())


def _small_ladder_dataset(node_encoding: str) -> LadderConcatDataset:
    return LadderConcatDataset(
        per_class=4,
        length_range=(5, 9),
        seed=314159,
        node_encoding=node_encoding,
        pe_dim=20,
        crossing_range=(2, 2),
    )


def test_degree_one_hot_marks_each_node_degree():
    adjacency = nx.to_numpy_array(nx.star_graph(3))  # centre degree 3, leaves 1
    features = degree_one_hot_features(adjacency, 5)
    assert features.shape == (4, 5)
    np.testing.assert_array_equal(features.sum(axis=1), np.ones(4))
    assert features[0, 3] == 1.0
    assert np.all(features[1:, 1] == 1.0)


def test_degree_one_hot_rejects_degrees_that_do_not_fit():
    adjacency = nx.to_numpy_array(nx.star_graph(5))
    with pytest.raises(ValueError, match="does not fit"):
        degree_one_hot_features(adjacency, 5)


def test_degree_dataset_uses_the_quantum_graphs_and_carries_the_class_signal():
    degree_dataset = _small_ladder_dataset("degree")
    quantum_dataset = _small_ladder_dataset("quantum")
    assert [r.edges for r in degree_dataset.records] == [
        r.edges for r in quantum_dataset.records
    ]
    assert [r.label for r in degree_dataset.records] == [
        r.label for r in quantum_dataset.records
    ]
    assert degree_dataset.feature_schema["node_feature_dim"] == 20
    for record in degree_dataset.records:
        assert record.node_features.shape == (record.num_nodes, 20)
        degree_two_nodes = int(record.node_features[:, 2].sum())
        degree_four_nodes = int(record.node_features[:, 4].sum())
        # The confounder this control tests (Phase 3 gap diagnosis).
        expected = 4 if record.label == 0 else 3
        assert (degree_two_nodes, degree_four_nodes) == (expected, expected)


def test_degree_control_has_the_quantum_model_and_parameter_count():
    degree_config = _load_config("synthetic_degree_classical")
    quantum_config = _load_config("synthetic_quantum_original")
    runner._validate_config(degree_config)
    differing_keys = {
        key
        for key in set(degree_config) | set(quantum_config)
        if degree_config.get(key) != quantum_config.get(key)
    }
    assert differing_keys == {"description", "dataset_kwargs", "encoding"}
    schema = _small_ladder_dataset("degree").feature_schema
    degree_model = runner._build_model(degree_config, edge_dim=0, feature_schema=schema)
    quantum_model = runner._build_model(
        quantum_config,
        edge_dim=0,
        feature_schema=_small_ladder_dataset("quantum").feature_schema,
    )
    count = sum(p.numel() for p in degree_model.parameters() if p.requires_grad)
    assert count == 3906
    assert count == sum(
        p.numel() for p in quantum_model.parameters() if p.requires_grad
    )


def test_degree_control_trains_end_to_end(tmp_path):
    config = copy.deepcopy(_load_config("synthetic_degree_classical"))
    # 20 graphs, so the 0.1 validation and test fractions are non-empty.
    config["dataset_kwargs"].update(per_class=10, length_range=[5, 9])
    del config["dataset_kwargs"]["cache_path"]
    config.update(epochs=1, seed=0)
    config.pop("seeds")
    output = runner.train_and_evaluate(config, tmp_path)
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    assert output["num_params"] == 3906
    assert 0.0 <= output["test_metrics"]["acc"] <= 1.0
