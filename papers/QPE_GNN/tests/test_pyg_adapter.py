"""Tests for the PyTorch Geometric benchmark adapter.

These tests don't require network access: we monkey-patch ``PyGBenchmarkAdapter``
internals with a small in-memory list of fake ``Data`` objects and verify that
the (num_nodes, edges, label) tuple-shape interface produces the right values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch_geometric.datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import runner  # noqa: E402
from lib.data import GraphRecord, PyGBenchmarkAdapter, collate_pad  # noqa: E402
from lib.model import GRITLite  # noqa: E402


class _FakeData:
    def __init__(self, x, edge_index, y, edge_attr=None):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.edge_attr = edge_attr
        self.num_nodes = x.shape[0]


def _make_fake_dataset(graphs, mode):
    return graphs  # PyG datasets are list-like; we use the same API


def _fake_zinc(monkey):
    # Two tiny ZINC-like graphs: x = node feature, edge_index = (2, E), y = scalar.
    g1 = _FakeData(
        x=torch.tensor([[1], [2], [3]], dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=torch.long),
        y=torch.tensor([1.7]),
        edge_attr=torch.tensor([[0], [0], [2], [2]], dtype=torch.long),
    )
    g2 = _FakeData(
        x=torch.zeros((4, 1)),
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 1, 2], [1, 2, 3, 0, 0, 1]], dtype=torch.long
        ),
        y=torch.tensor([0.42]),
        edge_attr=torch.tensor([[0], [1], [2], [0], [0], [1]], dtype=torch.long),
    )
    return [g1, g2]


def _fake_mnist_class(monkey):
    g = _FakeData(
        x=torch.zeros((5, 1)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        y=torch.tensor([7]),
    )
    return [g]


def _fake_pattern_node(monkey):
    g = _FakeData(
        x=torch.zeros((6, 1)),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
        y=torch.tensor([0, 1, 1, 1, 0, 1]),  # majority = 1
    )
    return [g]


def test_zinc_adapter_returns_scalar_label(monkeypatch):
    # We patch PyGBenchmarkAdapter to use our fake ZINC list without
    # downloading anything.
    fake = _fake_zinc(monkeypatch)

    class _Wrap:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    ad = PyGBenchmarkAdapter.__new__(PyGBenchmarkAdapter)
    ad._ds = _Wrap(fake)
    ad._limit = 2
    ad._mode = "graph_reg"
    ad.name = "zinc"
    ad.mode = "graph_reg"
    ad.directed = False
    record = ad[0]
    assert record.num_nodes == 3
    assert record.edges == [(0, 1), (1, 0), (2, 1), (1, 2)]
    assert record.node_features.tolist() == [[1], [2], [3]]
    assert record.edge_features.tolist() == [[0], [0], [2], [2]]
    assert isinstance(record.label, float)
    assert abs(record.label - 1.7) < 1e-6


def test_mnist_adapter_returns_int_class(monkeypatch):
    fake = _fake_mnist_class(monkeypatch)

    class _Wrap:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    ad = PyGBenchmarkAdapter.__new__(PyGBenchmarkAdapter)
    ad._ds = _Wrap(fake)
    ad._limit = 1
    ad._mode = "graph_class"
    ad.name = "mnist"
    ad.mode = "graph_class"
    ad.directed = True
    record = ad[0]
    assert record.num_nodes == 5
    assert record.edges == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert record.label == 7
    assert isinstance(record.label, int)
    assert record.directed


def test_pattern_adapter_preserves_node_labels(monkeypatch):
    fake = _fake_pattern_node(monkeypatch)

    class _Wrap:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    ad = PyGBenchmarkAdapter.__new__(PyGBenchmarkAdapter)
    ad._ds = _Wrap(fake)
    ad._limit = 1
    ad._mode = "node_class"
    ad.name = "pattern"
    ad.mode = "node_class"
    ad.directed = False
    record = ad[0]
    assert record.num_nodes == 6
    assert record.label == [0, 1, 1, 1, 0, 1]


def test_collate_pad_pads_node_labels_with_ignore_index():
    batch = collate_pad(
        [
            (3, [(0, 1), (1, 2)], [0, 1, 0]),
            (2, [(0, 1)], [1, 1]),
        ]
    )

    assert batch["label"].shape == (2, 3)
    assert batch["label"].tolist() == [[0, 1, 0], [1, 1, -1]]


def test_pyg_adapter_collates_with_existing_pipeline(monkeypatch):
    """The (n, edges, label) output of the adapter must be consumable by
    ``collate_pad`` exactly like the synthetic datasets."""
    fake = _fake_zinc(monkeypatch)

    class _Wrap:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    ad = PyGBenchmarkAdapter.__new__(PyGBenchmarkAdapter)
    ad._ds = _Wrap(fake)
    ad._limit = 2
    ad._mode = "graph_reg"
    ad.name = "zinc"
    ad.mode = "graph_reg"
    ad.directed = False

    batch = collate_pad([ad[i] for i in range(2)])
    assert batch["A"].shape == (2, 4, 4)
    assert batch["mask"].shape == (2, 4)
    assert batch["label"].shape == (2,)
    assert batch["node_features"].shape == (2, 4, 1)
    assert batch["edge_features"].shape == (2, 4, 4, 1)
    assert batch["node_features"][0, :3].tolist() == [[1], [2], [3]]
    assert batch["edge_features"][0, 2, 1].tolist() == [2]


def test_categorical_features_survive_adapter_collate_and_model():
    adapter = PyGBenchmarkAdapter.__new__(PyGBenchmarkAdapter)
    adapter._ds = _fake_zinc(None)
    adapter._limit = 2
    adapter._mode = "graph_reg"
    adapter.name = "zinc"
    adapter.mode = "graph_reg"
    adapter.directed = False
    batch = collate_pad([adapter[index] for index in range(2)])
    positional_encoding = torch.zeros((2, 4, 4, 3))
    model = GRITLite(
        edge_dim=3,
        node_dim=8,
        depth=1,
        num_heads=2,
        head="graph_reg",
        node_in_dim=1,
        edge_in_dim=1,
        node_feature_type="categorical",
        edge_feature_type="categorical",
        node_vocab_sizes=(28,),
        edge_vocab_sizes=(4,),
    )

    output = model(
        positional_encoding,
        batch["mask"],
        node_features=batch["node_features"],
        edge_features=batch["edge_features"],
        edge_mask=batch["edge_mask"],
    )

    assert output.shape == (2,)


def test_collate_preserves_directed_edges_without_symmetrizing():
    record = GraphRecord(
        num_nodes=2,
        edges=[(0, 1)],
        label=0,
        node_features=torch.ones((2, 1)),
        directed=True,
    )

    batch = collate_pad([record])

    assert batch["A"][0, 0, 1] == 1
    assert batch["A"][0, 1, 0] == 0


def test_zinc_adapter_uses_shared_paper_data_directory(monkeypatch, tmp_path):
    captured_arguments = {}

    def fake_zinc(**kwargs):
        captured_arguments.update(kwargs)
        return _fake_zinc(monkeypatch)

    monkeypatch.setattr(torch_geometric.datasets, "ZINC", fake_zinc)

    adapter = PyGBenchmarkAdapter(name="zinc", data_root=tmp_path)

    expected_root = (tmp_path / "QPE_GNN" / "zinc").resolve()
    assert Path(captured_arguments["root"]) == expected_root
    assert adapter.feature_schema["node_vocab_sizes"] == (21,)


def test_gnn_benchmark_adapter_uses_shared_paper_data_directory(monkeypatch, tmp_path):
    captured_arguments = {}

    def fake_gnn_benchmark(**kwargs):
        captured_arguments.update(kwargs)
        return _fake_mnist_class(monkeypatch)

    monkeypatch.setattr(
        torch_geometric.datasets,
        "GNNBenchmarkDataset",
        fake_gnn_benchmark,
    )

    PyGBenchmarkAdapter(name="mnist", data_root=tmp_path, split="val")

    expected_root = (tmp_path / "QPE_GNN" / "mnist").resolve()
    assert Path(captured_arguments["root"]) == expected_root
    assert captured_arguments["name"] == "MNIST"
    assert captured_arguments["split"] == "val"


def test_benchmark_config_cannot_override_official_split(tmp_path):
    config = {
        "dataset": "zinc",
        "model": "grit_lite",
        "encoding": "rrwp",
        "pe_dim": 4,
        "data_root": str(tmp_path),
        "dataset_kwargs": {"split": "val", "limit": 5},
    }

    with pytest.raises(ValueError, match="official train, val, and test splits"):
        runner._validate_config(config)


def test_runner_builds_all_official_benchmark_splits(monkeypatch, tmp_path):
    captured_splits = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            captured_splits.append(kwargs["split"])

    monkeypatch.setattr(runner, "PyGBenchmarkAdapter", _FakeAdapter)
    config = {
        "dataset": "zinc",
        "data_root": str(tmp_path),
        "dataset_kwargs": {"subset": True, "limit": 5},
    }

    runner._build_datasets(config)

    assert captured_splits == ["train", "val", "test"]


def test_zinc_fails_when_bond_features_are_missing():
    adapter = PyGBenchmarkAdapter.__new__(PyGBenchmarkAdapter)
    adapter._ds = [
        _FakeData(
            x=torch.zeros((2, 1), dtype=torch.long),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            y=torch.tensor([0.0]),
        )
    ]
    adapter._limit = 1
    adapter._mode = "graph_reg"
    adapter.name = "zinc"
    adapter.mode = "graph_reg"
    adapter.directed = False

    with pytest.raises(ValueError, match="bond features"):
        adapter[0]


def test_adapter_fails_when_node_features_are_missing():
    adapter = PyGBenchmarkAdapter.__new__(PyGBenchmarkAdapter)
    adapter._ds = [
        _FakeData(
            x=torch.ones((2, 1)),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            y=torch.tensor([0]),
        )
    ]
    adapter._ds[0].x = None
    adapter._limit = 1
    adapter._mode = "graph_class"
    adapter.name = "mnist"
    adapter.mode = "graph_class"
    adapter.directed = True

    with pytest.raises(ValueError, match="missing data.x"):
        adapter[0]
