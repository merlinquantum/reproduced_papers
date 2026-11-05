"""Dataset loaders for the QPE-GNN reproduction.

Four families:

- ``LadderConcatDataset``: the synthetic binary classification task from
  appendix D.2 (ladder graphs with crossings).
- ``SRGFamily``: strongly regular graphs (Sec. 4.3). We ship srg(16,6,2,2)
  generated on the fly with networkx so we don't depend on a graph catalogue.
- ``RandomGraphRegression``: an Erdős-Rényi graph regression dataset used as
  a CPU-friendly proxy for ZINC.
- ``PyGBenchmarkAdapter``: thin adapter over PyTorch Geometric's ``ZINC``
  and ``GNNBenchmarkDataset`` (MNIST, CIFAR10, PATTERN, CLUSTER). Downloads
  on demand under ``data/QPE_GNN/<name>/``. Used by the ``zinc_*``, ``mnist_*``,
  ``cifar10_*``, ``pattern_*``, ``cluster_*`` configs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class GraphRecord:
    """Store one graph and all features required by the benchmark model.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    edges : list[tuple[int, int]]
        Ordered edges. Directed benchmark edges are not canonicalized or
        symmetrized.
    label : int | float | list[int]
        Graph-level target or node-level targets.
    node_features : torch.Tensor
        Node feature matrix with shape ``(num_nodes, node_feature_dim)``.
    edge_features : torch.Tensor | None
        Edge feature matrix aligned with ``edges``. Default value is None.
    directed : bool
        Whether edge direction is part of the dataset representation. Default
        value is False.
    categorical_node_features : bool
        Whether node features require categorical embeddings. Default value is
        False.
    categorical_edge_features : bool
        Whether edge features require categorical embeddings. Default value is
        False.
    """

    num_nodes: int
    edges: list[tuple[int, int]]
    label: int | float | list[int]
    node_features: torch.Tensor
    edge_features: torch.Tensor | None = None
    directed: bool = False
    categorical_node_features: bool = False
    categorical_edge_features: bool = False


# ---------------------------------------------------------------------------
# Synthetic ladder graphs from Appendix D.2.
# ---------------------------------------------------------------------------


def _ladder_pairs(length: int) -> list[tuple[int, int]]:
    """Plain ladder of `length` rungs (2 * length nodes, even index = bottom)."""
    edges: list[tuple[int, int]] = []
    for r in range(length):
        a = 2 * r
        b = 2 * r + 1
        edges.append((a, b))  # rung
        if r + 1 < length:
            edges.append((a, 2 * (r + 1)))  # bottom rail
            edges.append((b, 2 * (r + 1) + 1))  # top rail
    return edges


def make_type0(length: int) -> tuple[int, list[tuple[int, int]]]:
    """Plain ladder. Two possible Ising ground states (anti-ferromagnetic)."""
    return 2 * length, _ladder_pairs(length)


def make_type1(
    length: int, crossings: Sequence[int]
) -> tuple[int, list[tuple[int, int]]]:
    """Ladder with crossings at the given rung indices, separated by odd gaps.

    A crossing at rung ``r`` adds the diagonal from the bottom of rung ``r``
    to the top of rung ``r + 1``. Both horizontal rail edges remain. This is
    the construction drawn in Appendix D.2, Figure 7.
    """
    n, edges = make_type0(length)
    edges = list(edges)
    for r in crossings:
        if r < 0 or r + 1 >= length:
            raise ValueError(f"crossing at rung {r} needs rung {r + 1}")
        edges.append((2 * r, 2 * (r + 1) + 1))
    return n, edges


def make_type2(length: int) -> tuple[int, list[tuple[int, int]]]:
    """Ladder of odd length with crossings at both ends (many ground states)."""
    assert length % 2 == 1, "type-2 graphs require odd length"
    return make_type1(length, crossings=[0, length - 2])


def concat_pair(
    a_nodes: int, a_edges, b_nodes: int, b_edges
) -> tuple[int, list[tuple[int, int]]]:
    """Concatenate two graphs by gluing the right end of `a` to the left end
    of `b`. We add bridging rails (a's last rung) → (b's first rung).
    """
    shift = a_nodes
    new_edges = list(a_edges) + [(u + shift, v + shift) for u, v in b_edges]
    # Bridge: rightmost rung of A is nodes (a_nodes-2, a_nodes-1); leftmost of B
    # is (shift, shift+1).
    new_edges.append((a_nodes - 2, shift))
    new_edges.append((a_nodes - 1, shift + 1))
    return a_nodes + b_nodes, new_edges


class LadderConcatDataset(Dataset):
    """Binary classification on type-0 + type-1 vs type-0 + type-2 concatenations.

    Default sizes are intentionally small so smoke tests stay fast. The paper
    uses 400 graphs per class with lengths 100..400; we expose ``length_range``
    and ``per_class`` to scale up.
    """

    def __init__(
        self,
        per_class: int = 32,
        length_range: tuple[int, int] = (4, 10),
        seed: int = 0,
        node_encoding: str = "none",
        pe_dim: int = 20,
        crossing_range: tuple[int, int] = (2, 9),
        cache_path: str | Path | None = None,
    ):
        if length_range[0] < 3 or length_range[0] > length_range[1]:
            raise ValueError("length_range must be increasing and start at 3 or above")
        if crossing_range[0] < 1 or crossing_range[0] > crossing_range[1]:
            raise ValueError("crossing_range must contain positive increasing values")
        cache_metadata = {
            "per_class": per_class,
            "length_range": tuple(length_range),
            "seed": seed,
            "node_encoding": node_encoding,
            "pe_dim": pe_dim,
            "crossing_range": tuple(crossing_range),
        }
        if cache_path is not None and Path(cache_path).is_file():
            cached_dataset = torch.load(cache_path, weights_only=False)
            if cached_dataset.get("metadata") != cache_metadata:
                raise ValueError(f"synthetic cache metadata mismatch: {cache_path}")
            self.records = cached_dataset["records"]
            self.feature_schema = cached_dataset["feature_schema"]
            return
        rng = np.random.default_rng(seed)
        graph_labels: list[tuple[int, list[tuple[int, int]], int]] = []
        for _ in range(per_class):
            length = int(rng.integers(length_range[0], length_range[1] + 1))
            if length % 2 == 0:
                length = length + 1 if length < length_range[1] else length - 1
            # class 0: type0 + type1
            n0, e0 = make_type0(length)
            # Equal-parity crossing indices leave an odd number of intervening
            # ladder nodes, as required by Appendix D.2.
            possible = list(range(0, length - 1, 2))
            maximum_crossings = min(crossing_range[1], len(possible))
            minimum_crossings = min(crossing_range[0], maximum_crossings)
            num_cross = int(rng.integers(minimum_crossings, maximum_crossings + 1))
            picks = list(rng.choice(possible, size=num_cross, replace=False))
            picks = [int(p) for p in sorted(picks)]
            n1, e1 = make_type1(length, picks)
            n, edges = concat_pair(n0, e0, n1, e1)
            graph_labels.append((n, edges, 0))

            # class 1: type0 + type2
            # Both classes use exactly the same component length. Type 2 needs
            # odd length, so sample one common odd length for the pair.
            n2, e2 = make_type2(length)
            n, edges = concat_pair(n0, e0, n2, e2)
            graph_labels.append((n, edges, 1))
        permutation = rng.permutation(len(graph_labels))
        self.records = [
            self._make_record(*graph_labels[index], node_encoding, pe_dim)
            for index in permutation
        ]
        self.feature_schema = {
            "node_feature_dim": 1 if node_encoding in {"none", "rrwp"} else pe_dim,
            "edge_feature_dim": pe_dim if node_encoding == "rrwp" else 0,
            "node_feature_type": "continuous",
            "edge_feature_type": "continuous",
            "node_vocab_sizes": (),
            "edge_vocab_sizes": (),
        }
        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "metadata": cache_metadata,
                    "records": self.records,
                    "feature_schema": self.feature_schema,
                },
                cache_path,
            )

    @staticmethod
    def _make_record(
        num_nodes: int,
        edges: list[tuple[int, int]],
        label: int,
        node_encoding: str,
        pe_dim: int,
    ) -> GraphRecord:
        """Construct one encoded synthetic graph record."""
        from .qpe import (
            ladder_ground_state_correlation_eigvecs,
            laplacian_eigenvectors,
            rrwp_edge_features,
        )

        adjacency = adj_from_record(num_nodes, edges)
        if node_encoding == "quantum":
            features = ladder_ground_state_correlation_eigvecs(num_nodes, edges, pe_dim)
        elif node_encoding == "laplacian":
            features = laplacian_eigenvectors(adjacency, pe_dim)
        elif node_encoding == "rrwp":
            features = np.ones((num_nodes, 1), dtype=np.float64)
            edge_features = rrwp_edge_features(adjacency, edges, pe_dim)
        elif node_encoding == "none":
            features = np.ones((num_nodes, 1), dtype=np.float64)
        else:
            raise ValueError(f"unknown synthetic node encoding: {node_encoding}")
        if node_encoding != "rrwp":
            edge_features = None
        return GraphRecord(
            num_nodes=num_nodes,
            edges=edges,
            label=label,
            node_features=torch.from_numpy(features).float(),
            edge_features=(
                None
                if edge_features is None
                else torch.from_numpy(edge_features).float()
            ),
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


# ---------------------------------------------------------------------------
# Strongly regular graphs for Sec. 4.3.
# ---------------------------------------------------------------------------


def rook_graph(n: int) -> nx.Graph:
    """n x n Rook graph: K_n □ K_n. srg(n^2, 2(n-1), n-2, 2)."""
    G = nx.Graph()
    G.add_nodes_from(range(n * n))
    for i in range(n):
        for j in range(n):
            u = i * n + j
            for k in range(n):
                if k != j:
                    G.add_edge(u, i * n + k)
                if k != i:
                    G.add_edge(u, k * n + j)
    return G


def paley_graph_9() -> nx.Graph:
    """Paley graph of order 9. srg(9, 4, 1, 2) — the SRG companion of K_3 □ K_3.

    Built directly over GF(9) = Z_3[i] / (i^2 + 1). Elements are encoded as
    integers a + 3 b for (a, b) ∈ Z_3 × Z_3.
    """

    def add(x, y):
        ax, bx = x % 3, x // 3
        ay, by = y % 3, y // 3
        return (ax + ay) % 3 + 3 * ((bx + by) % 3)

    def sub(x, y):
        ax, bx = x % 3, x // 3
        ay, by = y % 3, y // 3
        return (ax - ay) % 3 + 3 * ((bx - by) % 3)

    def mul(x, y):
        ax, bx = x % 3, x // 3
        ay, by = y % 3, y // 3
        # (ax + bx i)(ay + by i) = (ax ay - bx by) + (ax by + bx ay) i; i^2 = -1.
        return (ax * ay - bx * by) % 3 + 3 * ((ax * by + bx * ay) % 3)

    nonzero = list(range(1, 9))
    squares = {mul(x, x) for x in nonzero}
    G = nx.Graph()
    G.add_nodes_from(range(9))
    for i in range(9):
        for j in range(i + 1, 9):
            if sub(j, i) in squares:
                G.add_edge(i, j)
    return G


def paley_graph(q: int) -> nx.Graph:
    """Dispatch by `q`. We only need q = 9 here."""
    if q == 9:
        return paley_graph_9()
    # Fallback: prime q ≡ 1 mod 4.
    assert q % 4 == 1 and all(q % p != 0 for p in range(2, int(q**0.5) + 1))
    residues = {(x * x) % q for x in range(1, q)}
    G = nx.Graph()
    G.add_nodes_from(range(q))
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                G.add_edge(i, j)
    return G


def shrikhande_graph() -> nx.Graph:
    """Shrikhande graph: srg(16, 6, 2, 2), non-isomorphic to 4x4 Rook."""
    # 16 nodes arranged in a 4x4 toroidal grid; node (i, j) connected to
    # (i+1, j), (i, j+1), (i+1, j+1) (mod 4) and their inverses.
    G = nx.Graph()
    for i in range(4):
        for j in range(4):
            u = i * 4 + j
            G.add_node(u)
            for di, dj in [(1, 0), (0, 1), (1, 1)]:
                v = ((i + di) % 4) * 4 + (j + dj) % 4
                if u != v:
                    G.add_edge(u, v)
    return G


@dataclass
class SRGPair:
    """A pair of non-isomorphic SRGs sharing the same (ν, k, λ, μ) tuple."""

    name: str
    g1: nx.Graph
    g2: nx.Graph
    params: tuple[int, int, int, int]


def srg_pairs() -> list[SRGPair]:
    """Return non-isomorphic SRG pairs sharing the same (ν, k, λ, μ).

    srg(9, 4, 1, 2) is unique up to isomorphism (Rook(3) ≅ Paley(9)) so the
    smallest non-trivial family is srg(16, 6, 2, 2): 4x4 Rook vs Shrikhande.
    The paper uses srg(25, 12, 5, 6) (15 graphs) and srg(26, 10, 3, 4) (10
    graphs); we ship only the Rook/Shrikhande pair to keep the file size
    self-contained. Loading external SRG catalogues is left to extensions.
    """
    return [
        SRGPair(
            name="srg(16,6,2,2)",
            g1=rook_graph(4),  # 4x4 Rook
            g2=shrikhande_graph(),
            params=(16, 6, 2, 2),
        ),
    ]


# ---------------------------------------------------------------------------
# Tiny graph regression dataset (proxy for ZINC, runnable on CPU).
# ---------------------------------------------------------------------------


class RandomGraphRegression(Dataset):
    """Small graph regression dataset. Label = spectral gap of the laplacian.

    Each graph is a random connected graph drawn from an Erdős–Rényi model.
    Returns ``(num_nodes, edges, label)``.
    """

    def __init__(
        self,
        num_graphs: int = 64,
        n_range: tuple[int, int] = (6, 10),
        p: float = 0.4,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.items: list[tuple[int, list[tuple[int, int]], float]] = []
        for _ in range(num_graphs):
            n = int(rng.integers(n_range[0], n_range[1] + 1))
            # Build a connected random graph.
            while True:
                G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(2**30)))
                if nx.is_connected(G) and G.number_of_edges() > 0:
                    break
            edges = list(G.edges())
            L = nx.laplacian_matrix(G).toarray().astype(np.float64)
            eigs = np.sort(np.linalg.eigvalsh(L))
            gap = float(eigs[1])  # algebraic connectivity (Fiedler value)
            self.items.append((n, edges, gap))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        n, edges, gap = self.items[idx]
        return n, edges, gap


# ---------------------------------------------------------------------------
# A collate that turns a list of (n, edges, label) into batched tensors.
# ---------------------------------------------------------------------------


def adj_from_record(n: int, edges) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float64)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


# ---------------------------------------------------------------------------
# PyTorch Geometric adapters for the standard benchmarks (Sec. 5.1 / Table 1).
# ---------------------------------------------------------------------------


class PyGBenchmarkAdapter(Dataset):
    """Thin adapter that lets ``GNNBenchmarkDataset`` / ``ZINC`` PyG datasets
    behave like the rest of ``lib.data`` (``(num_nodes, edges, labels)`` tuples).

    Supports the five benchmarks used in the paper:

    - ``"zinc"``                   → graph-level regression (Mean Abs. Error).
    - ``"mnist"``, ``"cifar10"``   → graph-level 10-class classification.
    - ``"pattern"``, ``"cluster"`` → inductive node-level classification.
      These datasets retain one class label per node for the node-level head.

    Datasets are downloaded into ``data_root/QPE_GNN/<name>`` on first use
    (~hundreds of MB to a few GB each). Pass ``limit`` to truncate the split
    for smoke runs.
    """

    _GNN_BENCHMARKS = {
        "mnist": "MNIST",
        "cifar10": "CIFAR10",
        "pattern": "PATTERN",
        "cluster": "CLUSTER",
    }
    _DIRECTED_BENCHMARKS = {"mnist", "cifar10"}
    _CATEGORICAL_VOCABULARIES = {
        "zinc": {
            "node": (28,),
            "edge": (4,),
        }
    }

    def __init__(
        self,
        name: str,
        data_root: str | Path,
        split: str = "train",
        limit: int | None = None,
        subset: bool = True,
    ):
        from torch_geometric.datasets import ZINC, GNNBenchmarkDataset

        name = name.lower()
        dataset_root = Path(data_root).expanduser().resolve() / "QPE_GNN" / name
        if name == "zinc":
            self._ds = ZINC(root=str(dataset_root), subset=subset, split=split)
            self._mode = "graph_reg"
        elif name in self._GNN_BENCHMARKS:
            self._ds = GNNBenchmarkDataset(
                root=str(dataset_root),
                name=self._GNN_BENCHMARKS[name],
                split=split,
            )
            self._mode = "graph_class" if name in ("mnist", "cifar10") else "node_class"
        else:
            raise ValueError(f"unknown PyG benchmark: {name}")
        self._limit = limit if limit is not None else len(self._ds)
        self._limit = min(self._limit, len(self._ds))
        self.name = name
        self.mode = self._mode
        self.directed = name in self._DIRECTED_BENCHMARKS
        self.subset = subset
        self._validate_feature_schema()

    def _validate_feature_schema(self) -> None:
        if not self._limit:
            raise ValueError(f"{self.name} split is empty")
        sample = self._ds[0]
        if getattr(sample, "x", None) is None:
            raise ValueError(
                f"{self.name} requires node features, but data.x is missing"
            )
        node_feature_dim = 1 if sample.x.ndim == 1 else int(sample.x.shape[1])

        edge_attributes = getattr(sample, "edge_attr", None)
        if self.name == "zinc" and edge_attributes is None:
            raise ValueError(
                "zinc requires categorical bond features in data.edge_attr"
            )
        if edge_attributes is None:
            edge_feature_dim = 0
        elif edge_attributes.ndim == 1:
            edge_feature_dim = 1
        else:
            edge_feature_dim = int(edge_attributes.shape[1])

        vocabularies = self._CATEGORICAL_VOCABULARIES.get(self.name, {})
        if self.name == "zinc" and self.subset:
            vocabularies = {"node": (21,), "edge": (4,)}
        self.feature_schema = {
            "node_feature_dim": node_feature_dim,
            "edge_feature_dim": edge_feature_dim,
            "node_feature_type": "categorical"
            if "node" in vocabularies
            else "continuous",
            "edge_feature_type": "categorical"
            if "edge" in vocabularies
            else "continuous",
            "node_vocab_sizes": vocabularies.get("node", ()),
            "edge_vocab_sizes": vocabularies.get("edge", ()),
        }

    def __len__(self):
        return self._limit

    def __getitem__(self, idx):
        data = self._ds[idx]
        if getattr(data, "x", None) is None:
            raise ValueError(f"{self.name} record {idx} is missing data.x")
        n = int(data.num_nodes)
        ei = data.edge_index.numpy()
        edges = [(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])]
        node_features = data.x.detach().cpu()
        if node_features.ndim == 1:
            node_features = node_features.unsqueeze(-1)

        raw_edge_features = getattr(data, "edge_attr", None)
        if self.name == "zinc" and raw_edge_features is None:
            raise ValueError(
                f"zinc record {idx} is missing categorical bond features in data.edge_attr"
            )
        edge_features = None
        if raw_edge_features is not None:
            edge_features = raw_edge_features.detach().cpu()
            if edge_features.ndim == 1:
                edge_features = edge_features.unsqueeze(-1)
            if edge_features.shape[0] != len(edges):
                raise ValueError(
                    f"{self.name} record {idx} has {len(edges)} edges but "
                    f"{edge_features.shape[0]} edge feature rows"
                )
        if self._mode == "graph_reg":
            label = float(data.y.item())
        elif self._mode == "graph_class":
            label = int(data.y.item())
        else:  # node_class
            label = data.y.reshape(-1).long().tolist()
            if len(label) != n:
                raise ValueError(
                    f"expected {n} node labels for {self.name}, got {len(label)}"
                )
        categorical_features = self.name in self._CATEGORICAL_VOCABULARIES
        return GraphRecord(
            num_nodes=n,
            edges=edges,
            label=label,
            node_features=node_features,
            edge_features=edge_features,
            directed=self.directed,
            categorical_node_features=categorical_features,
            categorical_edge_features=categorical_features,
        )


def collate_pad(
    batch,
    max_nodes: int | None = None,
    include_dense_graph: bool = True,
) -> dict:
    """Pad and stack graph records without discarding benchmark features.

    Parameters
    ----------
    batch : list[GraphRecord | tuple]
        Graph records. Legacy three-tuples are accepted for generated datasets.
    max_nodes : int | None
        Explicit padded node count. If omitted, the largest graph in the batch
        determines the padded size. Default value is None.
    include_dense_graph : bool
        Whether to materialize dense adjacency and edge tensors. GCN training
        uses the returned sparse ``edge_index`` instead. Default value is True.

    Returns
    -------
    dict
        Padded adjacency, directed edge mask, node features, edge features,
        valid-node mask, labels, and original graph sizes.

    Raises
    ------
    ValueError
        If records mix incompatible node or edge feature schemas.
    """
    records = [
        item
        if isinstance(item, GraphRecord)
        else GraphRecord(
            num_nodes=item[0],
            edges=list(item[1]),
            label=item[2],
            node_features=torch.ones((item[0], 1), dtype=torch.float32),
        )
        for item in batch
    ]
    if max_nodes is None:
        max_nodes = max(record.num_nodes for record in records)
    B = len(batch)
    A_pad = (
        np.zeros((B, max_nodes, max_nodes), dtype=np.float32)
        if include_dense_graph
        else None
    )
    edge_mask = (
        np.zeros((B, max_nodes, max_nodes), dtype=bool) if include_dense_graph else None
    )
    mask = np.zeros((B, max_nodes), dtype=bool)
    node_feature_dim = records[0].node_features.shape[1]
    if any(record.node_features.shape[1] != node_feature_dim for record in records):
        raise ValueError(
            "all records in a batch must have the same node feature dimension"
        )
    categorical_node_features = records[0].categorical_node_features
    if any(
        record.categorical_node_features != categorical_node_features
        for record in records
    ):
        raise ValueError("cannot mix categorical and continuous node features")
    node_feature_dtype = np.int64 if categorical_node_features else np.float32
    node_features = np.zeros((B, max_nodes, node_feature_dim), dtype=node_feature_dtype)

    edge_feature_dimensions = {
        0 if record.edge_features is None else record.edge_features.shape[1]
        for record in records
    }
    if len(edge_feature_dimensions) != 1:
        raise ValueError(
            "all records in a batch must have the same edge feature dimension"
        )
    edge_feature_dim = edge_feature_dimensions.pop()
    categorical_edge_features = records[0].categorical_edge_features
    if any(
        record.categorical_edge_features != categorical_edge_features
        for record in records
    ):
        raise ValueError("cannot mix categorical and continuous edge features")
    edge_feature_dtype = np.int64 if categorical_edge_features else np.float32
    edge_features = (
        np.zeros((B, max_nodes, max_nodes, edge_feature_dim), dtype=edge_feature_dtype)
        if include_dense_graph
        else None
    )
    labels = []
    ns = []
    sparse_edges: list[tuple[int, int]] = []
    sparse_edge_features: list[np.ndarray] = []
    sparse_edge_arrays: list[np.ndarray] = []
    sparse_edge_feature_arrays: list[np.ndarray] = []
    for b, record in enumerate(records):
        n = record.num_nodes
        if not include_dense_graph:
            record_edges = np.asarray(record.edges, dtype=np.int64).T
            sparse_edge_arrays.append(record_edges + b * max_nodes)
            if record.edge_features is not None:
                sparse_edge_feature_arrays.append(record.edge_features.numpy())
            if not record.directed:
                sparse_edge_arrays.append(record_edges[::-1] + b * max_nodes)
                if record.edge_features is not None:
                    sparse_edge_feature_arrays.append(record.edge_features.numpy())
        else:
            for edge_index, (i, j) in enumerate(record.edges):
                sparse_edges.append((b * max_nodes + i, b * max_nodes + j))
                if record.edge_features is not None:
                    sparse_edge_features.append(
                        record.edge_features[edge_index].numpy()
                    )
                A_pad[b, i, j] = 1.0
                edge_mask[b, i, j] = True
                if record.edge_features is not None:
                    edge_features[b, i, j] = record.edge_features[edge_index].numpy()
                if not isinstance(batch[b], GraphRecord):
                    sparse_edges.append((b * max_nodes + j, b * max_nodes + i))
                    if record.edge_features is not None:
                        sparse_edge_features.append(
                            record.edge_features[edge_index].numpy()
                        )
                    A_pad[b, j, i] = 1.0
                    edge_mask[b, j, i] = True
                elif not record.directed:
                    sparse_edges.append((b * max_nodes + j, b * max_nodes + i))
                    if record.edge_features is not None:
                        sparse_edge_features.append(
                            record.edge_features[edge_index].numpy()
                        )
        mask[b, :n] = True
        node_features[b, :n] = record.node_features.numpy()
        labels.append(record.label)
        ns.append(n)
    node_level_labels = any(
        isinstance(label, (list, tuple, np.ndarray, torch.Tensor)) for label in labels
    )
    if node_level_labels:
        if not all(
            isinstance(label, (list, tuple, np.ndarray, torch.Tensor))
            for label in labels
        ):
            raise ValueError(
                "cannot mix graph-level and node-level labels in one batch"
            )
        padded_labels = np.full((B, max_nodes), -1, dtype=np.int64)
        for batch_index, (num_nodes, label) in enumerate(zip(ns, labels)):
            node_labels = np.asarray(label, dtype=np.int64).reshape(-1)
            if len(node_labels) != num_nodes:
                raise ValueError(
                    f"expected {num_nodes} node labels, got {len(node_labels)}"
                )
            padded_labels[batch_index, :num_nodes] = node_labels
        label_tensor = torch.from_numpy(padded_labels)
    else:
        label_tensor = torch.tensor(labels)

    if sparse_edge_arrays:
        sparse_edge_index = torch.from_numpy(np.concatenate(sparse_edge_arrays, axis=1))
        sparse_edge_feature_tensor = torch.from_numpy(
            np.concatenate(sparse_edge_feature_arrays, axis=0)
            if sparse_edge_feature_arrays
            else np.empty((sparse_edge_index.shape[1], 0), dtype=np.float32)
        )
    else:
        sparse_edge_index = torch.tensor(sparse_edges, dtype=torch.long).T.contiguous()
        sparse_edge_feature_tensor = torch.from_numpy(
            np.stack(sparse_edge_features)
            if sparse_edge_features
            else np.empty((len(sparse_edges), 0), dtype=np.float32)
        )

    return {
        "A": None if A_pad is None else torch.from_numpy(A_pad),
        "edge_mask": None if edge_mask is None else torch.from_numpy(edge_mask),
        "edge_features": (
            None if edge_features is None else torch.from_numpy(edge_features)
        ),
        "edge_index": sparse_edge_index,
        "sparse_edge_features": sparse_edge_feature_tensor,
        "mask": torch.from_numpy(mask),
        "node_features": torch.from_numpy(node_features),
        "label": label_tensor,
        "n": torch.tensor(ns),
    }
