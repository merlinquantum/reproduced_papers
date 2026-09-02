"""Tests for the data loaders."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.data import (  # noqa: E402
    LadderConcatDataset,
    RandomGraphRegression,
    adj_from_record,
    collate_pad,
    rook_graph,
    srg_pairs,
)


def test_ladder_dataset_balanced():
    ds = LadderConcatDataset(per_class=8, length_range=(3, 4), seed=0)
    labels = [ds[i].label for i in range(len(ds))]
    assert sum(labels) == 8
    assert len(labels) == 16


def test_collate_pad_shapes():
    ds = LadderConcatDataset(per_class=4, length_range=(3, 4), seed=0)
    batch = [ds[i] for i in range(len(ds))]
    out = collate_pad(batch)
    B = len(batch)
    Nmax = max(item.num_nodes for item in batch)
    assert out["A"].shape == (B, Nmax, Nmax)
    assert out["mask"].shape == (B, Nmax)
    assert out["label"].shape == (B,)


def test_synthetic_encodings_use_identical_graphs_and_labels():
    common_arguments = {
        "per_class": 4,
        "length_range": (5, 9),
        "seed": 314159,
        "pe_dim": 20,
    }
    datasets = [
        LadderConcatDataset(node_encoding=encoding, **common_arguments)
        for encoding in ("quantum", "rrwp", "laplacian", "none")
    ]
    reference = datasets[0]
    for dataset in datasets[1:]:
        assert [record.label for record in dataset.records] == [
            record.label for record in reference.records
        ]
        assert [record.edges for record in dataset.records] == [
            record.edges for record in reference.records
        ]


def test_synthetic_quantum_features_scale_beyond_dense_hilbert_limit():
    dataset = LadderConcatDataset(
        per_class=1,
        length_range=(101, 101),
        seed=0,
        node_encoding="quantum",
        pe_dim=20,
    )
    assert all(record.num_nodes == 404 for record in dataset.records)
    assert all(record.node_features.shape == (404, 20) for record in dataset.records)


def test_rook_graph_is_srg():
    G = rook_graph(4)
    # Should be 6-regular.
    deg = list(dict(G.degree()).values())
    assert all(d == 6 for d in deg)
    # Should have 16 nodes.
    assert G.number_of_nodes() == 16


def test_srg_pair_non_isomorphic():
    pair = srg_pairs()[0]
    assert not nx.is_isomorphic(pair.g1, pair.g2)
    # Both should have the same (n, k, lambda, mu).
    n1, k1, lam1, mu1 = pair.params
    for G in (pair.g1, pair.g2):
        assert G.number_of_nodes() == n1
        deg = list(dict(G.degree()).values())
        assert all(d == k1 for d in deg), f"expected {k1}-regular, got {set(deg)}"


def test_random_graph_regression_valid():
    ds = RandomGraphRegression(num_graphs=4, n_range=(5, 8), p=0.5, seed=0)
    for n, edges, label in [ds[i] for i in range(len(ds))]:
        A = adj_from_record(n, edges)
        # Connected ⇒ second-smallest eigenvalue > 0.
        L = np.diag(A.sum(axis=1)) - A
        eigs = np.sort(np.linalg.eigvalsh(L))
        assert label > 0
        np.testing.assert_allclose(label, eigs[1], atol=1e-9)
