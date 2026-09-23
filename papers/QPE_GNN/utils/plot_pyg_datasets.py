"""Visualise the PyG-benchmark datasets used in Table 1 of the paper.

For each of ZINC, MNIST-as-graphs, CIFAR10-as-graphs, PATTERN and CLUSTER:

1. Try to load a small slice from the real dataset via
   ``lib.data.PyGBenchmarkAdapter``. If the dataset is available (cached
   locally or downloadable), draw three example graphs and their RRWP and
   2-QiRW features side by side.

2. If the dataset is *not* available (no network, no cache, gated download,
   …), fall back to **shape-matched illustrative graphs** — synthetic graphs
   constructed from documented statistics in `Dwivedi et al., Benchmarking
   GNNs (2020)` (avg #nodes, avg #edges, prediction task, label types). The
   resulting figure is clearly labelled "illustrative — re-run with network
   for real examples".

Both branches save to ``results/figures/pyg_<name>.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[1]
SHARED_DATA_ROOT = REPOSITORY_ROOT / "data"
sys.path.insert(0, str(ROOT))

from lib.data import adj_from_record  # noqa: E402
from lib.qpe import (  # noqa: E402
    correlation_matrix_on_state,
    ising_ground_state,
    qirw_features,
    rrwp,
)

FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


# Documented statistics from Dwivedi et al. 2020 and the paper's Table 5.
BENCHMARK_STATS = {
    "zinc": {
        "avg_nodes": 23,
        "avg_edges": 25,
        "task": "graph-level regression (constrained solubility, MAE)",
        "label_type": "scalar in [-4, 4]",
        "n_classes": None,
        "directed": False,
        "comment": "Molecular graphs from ZINC. Nodes are atoms with categorical type; edges are bonds with bond-type features. We illustrate with random molecules of ~23 atoms generated from `nx.random_tree` plus a few extra cycle-closing edges to approximate ring structures.",
    },
    "mnist": {
        "avg_nodes": 70,
        "avg_edges": 564,
        "task": "graph-level 10-class classification (MNIST digits as superpixel graphs)",
        "label_type": "0..9",
        "n_classes": 10,
        "directed": True,
        "comment": "MNIST images converted to superpixel graphs via SLIC; ~70 superpixels per image, 8-NN edges. Each node has a (location, intensity) feature.",
    },
    "cifar10": {
        "avg_nodes": 117,
        "avg_edges": 941,
        "task": "graph-level 10-class classification (CIFAR10 as superpixel graphs)",
        "label_type": "0..9",
        "n_classes": 10,
        "directed": True,
        "comment": "CIFAR10 images converted to ~117-node superpixel graphs (similar pipeline to MNIST).",
    },
    "pattern": {
        "avg_nodes": 119,
        "avg_edges": 3039,
        "task": "inductive node-binary classification (subgraph detection)",
        "label_type": "binary per node",
        "n_classes": 2,
        "directed": False,
        "comment": "Synthetic stochastic block-model graphs with a planted pattern subgraph; each node is labelled 1 iff it belongs to the pattern.",
    },
    "cluster": {
        "avg_nodes": 117,
        "avg_edges": 2151,
        "task": "inductive node 6-class classification (community detection)",
        "label_type": "0..5 per node",
        "n_classes": 6,
        "directed": False,
        "comment": "Synthetic stochastic block-model graphs with 6 communities; each node is labelled with its community id.",
    },
}


def _data_already_cached(name: str) -> bool:
    """Cheap check: does the PyG processing artefact already exist on disk?"""
    base = SHARED_DATA_ROOT / "QPE_GNN" / name
    if not base.exists():
        return False
    # PyG stores processed tensors under <root>/<NAME>/processed/<split>.pt
    # or <root>/processed/<split>.pt depending on the dataset.
    for p in base.rglob("*.pt"):
        if p.stat().st_size > 1024:
            return True
    return False


def _network_reachable(timeout: float = 2.0) -> bool:
    """Cheap reachability probe.

    We avoid DNS resolution (which can take 5-10 s when blocked) and probe
    a single well-known IPv4 endpoint directly with a strict timeout. If
    the user explicitly sets ``QPE_FORCE_DOWNLOAD=1``, we skip the probe.
    """
    import os
    import socket

    if os.environ.get("QPE_FORCE_DOWNLOAD") == "1":
        return True
    try:
        # 1.1.1.1 is a stable public anycast IP that responds to TCP 443
        # within milliseconds when the network is up.
        with socket.create_connection(("1.1.1.1", 443), timeout=timeout):
            return True
    except OSError:
        return False


def _try_real(name: str, n_examples: int = 3):
    """Attempt to load `n_examples` graphs from the real PyG dataset. Return
    a list of ``(num_nodes, edges, label)`` tuples on success, or ``None``
    on any error (including network unreachable)."""
    if not _data_already_cached(name) and not _network_reachable():
        print(
            f"[{name}] no cached data and network unreachable → illustrative fallback"
        )
        return None
    try:
        from lib.data import PyGBenchmarkAdapter

        split = "val" if name in ("mnist", "cifar10", "pattern", "cluster") else "train"
        ad = PyGBenchmarkAdapter(
            name=name,
            data_root=SHARED_DATA_ROOT,
            split=split,
            limit=n_examples,
            subset=True,
        )
        return [ad[i] for i in range(min(n_examples, len(ad)))]
    except Exception as e:
        print(
            f"[{name}] PyG load failed → falling back to illustrative: "
            f"{type(e).__name__}: {str(e)[:140]}"
        )
        return None


def _illustrative_zinc(rng) -> list[tuple]:
    """Tree-plus-rings graphs with ~23 nodes, ~25 edges, random scalar label."""
    out = []
    for _ in range(3):
        n = int(rng.integers(18, 28))
        T = nx.random_labeled_tree(n, seed=int(rng.integers(2**30)))
        edges = {(min(u, v), max(u, v)) for u, v in T.edges()}
        # add a few extra edges to mimic ring closures
        extra = int(rng.integers(1, 4))
        for _ in range(extra):
            u, v = sorted(rng.choice(n, size=2, replace=False))
            edges.add((int(u), int(v)))
        edges = list(edges)
        label = float(rng.uniform(-4.0, 4.0))
        out.append((n, edges, label))
    return out


def _illustrative_mnist_cifar(rng, target_n: int) -> list[tuple]:
    """k-NN graphs on random 2D point clouds (k=8), mimicking SLIC superpixels."""
    out = []
    for _ in range(3):
        n = int(rng.integers(int(target_n * 0.7), int(target_n * 1.1)))
        pts = rng.normal(size=(n, 2))
        # 8-NN edges (undirected).
        from scipy.spatial import KDTree

        tree = KDTree(pts)
        _, idx = tree.query(pts, k=9)
        edges = set()
        for i in range(n):
            for j in idx[i, 1:]:
                edges.add((int(min(i, j)), int(max(i, j))))
        label = int(rng.integers(0, 10))
        out.append((n, list(edges), label))
    return out


def _illustrative_pattern(rng) -> list[tuple]:
    """SBM-like dense graphs with planted small dense subgraph."""
    out = []
    for _ in range(3):
        n = int(rng.integers(60, 90))  # smaller than 119 to keep figures readable
        p_dense = 0.5
        G = nx.erdos_renyi_graph(n, p_dense, seed=int(rng.integers(2**30)))
        if not nx.is_connected(G):
            G = nx.compose(G, nx.path_graph(n))
        # Pattern subgraph: take 10% of nodes, complete them.
        pat = list(rng.choice(n, size=max(3, n // 10), replace=False))
        for u in pat:
            for v in pat:
                if u != v:
                    G.add_edge(int(u), int(v))
        edges = list(G.edges())
        # Graph-level proxy = majority node label (most are 0 = not in pattern).
        label = 0
        out.append((n, edges, label))
    return out


def _illustrative_cluster(rng) -> list[tuple]:
    """SBM with 6 communities and intra-/inter-community probabilities."""
    out = []
    for _ in range(3):
        sizes = [int(rng.integers(8, 18)) for _ in range(6)]
        p_intra, p_inter = 0.6, 0.02
        p = [[p_intra if i == j else p_inter for j in range(6)] for i in range(6)]
        G = nx.stochastic_block_model(sizes, p, seed=int(rng.integers(2**30)))
        edges = list(G.edges())
        n = G.number_of_nodes()
        # Majority community as label proxy.
        label = int(np.argmax(sizes))
        out.append((n, edges, label))
    return out


ILLUSTRATIVE_FACTORIES = {
    "zinc": _illustrative_zinc,
    "mnist": lambda rng: _illustrative_mnist_cifar(rng, 70),
    "cifar10": lambda rng: _illustrative_mnist_cifar(rng, 117),
    "pattern": _illustrative_pattern,
    "cluster": _illustrative_cluster,
}


def _plot_one(name: str, examples: list[tuple], real: bool) -> None:
    """Three rows (graphs) × 4 columns (graph drawing + RRWP^3 + 2-QiRW(2) + Ising correlation)."""
    stats = BENCHMARK_STATS[name]
    fig, axes = plt.subplots(3, 4, figsize=(15.5, 11))
    for row, (n, edges, label) in enumerate(examples):
        A = adj_from_record(n, edges)
        # ── col 0: graph drawing ──────────────────────────────────────────
        ax = axes[row, 0]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        if n <= 30:
            pos = nx.spring_layout(G, seed=0)
            node_size = 200
            with_labels = True
        else:
            pos = nx.spring_layout(G, seed=0, k=1.5 / max(1, n**0.5))
            node_size = max(20, 600 // n)
            with_labels = False
        nx.draw(
            G,
            pos=pos,
            ax=ax,
            with_labels=with_labels,
            node_size=node_size,
            node_color="#1f77b4",
            font_size=6,
            width=0.6,
        )
        ax.set_title(
            f"example {row} — N={n}, |E|={len(edges)}\nlabel={label}", fontsize=9
        )

        # ── col 1: RRWP^3 ────────────────────────────────────────────────
        ax = axes[row, 1]
        K = 4
        P = rrwp(A, K)
        im = ax.imshow(P[3], cmap="Blues", aspect="auto")
        ax.set_title("RRWP^3 (classical)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.05)

        # ── col 2: 2-QiRW marginal step 2 ─────────────────────────────────
        ax = axes[row, 2]
        if n <= 14:  # k=2 subspace size = C(n, 2); explodes quickly
            Q = qirw_features(A, 2, 3)[-1]
            im = ax.imshow(Q, cmap="viridis", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.05)
            ax.set_title("2-QiRW step 2 (quantum)", fontsize=9)
        else:
            # Compute 1-QiRW instead (much cheaper).
            Q = qirw_features(A, 1, 4)[-1]
            im = ax.imshow(Q, cmap="viridis", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.05)
            ax.set_title(
                f"1-QiRW step 3 (quantum)\n(2-QiRW skipped: N={n} too large)",
                fontsize=8,
            )

        # ── col 3: Ising correlation matrix (or eigvec outer product) ─────
        ax = axes[row, 3]
        if n <= 16:
            psi = ising_ground_state(A)
            C = correlation_matrix_on_state(psi, n)
            im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.05)
            ax.set_title(r"Ising $\langle Z_i Z_j\rangle$ (quantum)", fontsize=9)
        else:
            # For larger graphs we can't compute the exact ground state;
            # show the QPE eigvec proxy from the |+>^N initial state (fast).
            # Use the laplacian eigvecs as a rough qualitative stand-in.
            L = np.diag(A.sum(axis=1)) - A
            w, V = np.linalg.eigh(L)
            top = V[:, :4]
            proxy = np.outer(top[:, 0], top[:, 0])
            im = ax.imshow(proxy, cmap="RdBu_r", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.05)
            ax.set_title(
                "Laplacian eigvec proxy\n(Ising ground state skipped: N too large)",
                fontsize=8,
            )

    source = (
        "real PyG download"
        if real
        else "illustrative (shape-matched, network unavailable)"
    )
    fig.suptitle(
        f"{name.upper()} — {stats['task']}\n"
        f"avg N={stats['avg_nodes']}, avg |E|={stats['avg_edges']}, labels: {stats['label_type']}\n"
        f"source: {source}",
        fontsize=10,
    )
    fig.tight_layout()
    out = FIGDIR / f"pyg_{name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} (source: {source})")


def main() -> None:
    rng = np.random.default_rng(0)
    for name in ["zinc", "mnist", "cifar10", "pattern", "cluster"]:
        examples = _try_real(name)
        real = examples is not None
        if examples is None:
            examples = ILLUSTRATIVE_FACTORIES[name](rng)
        _plot_one(name, examples, real)


if __name__ == "__main__":
    main()
