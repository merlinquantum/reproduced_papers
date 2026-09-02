"""Visualise the datasets used in the reproduction.

Run::

    python -m utils.plot_datasets

Produces, under ``results/figures/``:

- ``ladder_types.png`` — type-0, type-1, type-2 ladder building blocks with
  highlighted Ising ground-state configurations (right side of Fig. 7).
- ``ladder_concat_classes.png`` — one example graph per class with the RRWP
  feature and the ground-state correlation feature drawn side by side
  (analog of Fig. 8).
- ``srg_pair.png`` — the Rook(4) vs Shrikhande SRG family with the same
  permutation-invariant correlation distance plot as Fig. 3.
- ``graph_reg_examples.png`` — examples from the random-graph regression
  dataset coloured by label (Fiedler value).
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
sys.path.insert(0, str(ROOT))

from lib.data import (  # noqa: E402
    RandomGraphRegression,
    adj_from_record,
    concat_pair,
    make_type0,
    make_type1,
    make_type2,
    srg_pairs,
)
from lib.qpe import (  # noqa: E402
    correlation_matrix_on_state,
    ising_ground_state,
    rrwp,
)

FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def _graph_from(n, edges) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


def _ladder_pos(n: int, edges) -> dict[int, tuple[float, float]]:
    """Layout with even-indexed nodes on the bottom rail, odd on the top."""
    pos = {}
    for i in range(n):
        pos[i] = (i // 2, i % 2)
    return pos


def plot_ladder_types() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    cases = [
        ("type 0 (plain ladder)", *make_type0(5)),
        ("type 1 (one crossing)", *make_type1(5, [1])),
        ("type 2 (odd-length, end crossings)", *make_type2(5)),
    ]
    for ax, (title, n, edges) in zip(axes, cases):
        G = _graph_from(n, edges)
        pos = _ladder_pos(n, edges)
        # Mark one Ising ground-state colouring by computing the actual GS.
        A = adj_from_record(n, edges)
        psi = ising_ground_state(A)
        # Pick the basis state with the largest amplitude (any from the manifold).
        state = int(np.argmax(psi))
        bits = [(state >> i) & 1 for i in range(n)]
        colours = ["#1f77b4" if b == 0 else "#d62728" for b in bits]
        nx.draw(
            G,
            pos=pos,
            node_color=colours,
            with_labels=True,
            font_size=8,
            node_size=300,
            ax=ax,
        )
        ax.set_title(f"{title}\n|GS manifold| = {int((psi != 0).sum())}")
    fig.suptitle(
        "Ising ground states for the three ladder building blocks (cf. Fig. 7)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGDIR / "ladder_types.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_ladder_classes() -> None:
    # Class 0: type0 + type1; class 1: type0 + type2. Build small examples.
    n0, e0 = make_type0(3)
    n1, e1 = make_type1(3, [0])
    n2, e2 = make_type2(3)
    nA, eA = concat_pair(n0, e0, n1, e1)  # class 0
    nB, eB = concat_pair(n0, e0, n2, e2)  # class 1

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for row, (label, (n, edges)) in enumerate(
        [("class 0 (type0+type1)", (nA, eA)), ("class 1 (type0+type2)", (nB, eB))]
    ):
        G = _graph_from(n, edges)
        pos = _ladder_pos(n, edges)
        A = adj_from_record(n, edges)
        # Pure adjacency drawing.
        ax = axes[row, 0]
        nx.draw(
            G,
            pos=pos,
            with_labels=True,
            font_size=7,
            node_size=250,
            node_color="#cccccc",
            ax=ax,
        )
        ax.set_title(f"{label}\nN={n}, |E|={len(edges)}")
        # RRWP feature: show the 3-step distance matrix.
        ax = axes[row, 1]
        P = rrwp(A, 4)
        im = ax.imshow(P[3], cmap="Blues")
        ax.set_title("RRWP^3 (classical)")
        plt.colorbar(im, ax=ax, fraction=0.05)
        # Ising ground-state correlation matrix.
        ax = axes[row, 2]
        psi = ising_ground_state(A)
        C = correlation_matrix_on_state(psi, n)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(r"Ising $\langle Z_i Z_j\rangle$ (quantum)")
        plt.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle(
        "Classical RRWP features vs Ising correlations on the synthetic dataset (cf. Fig. 8)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGDIR / "ladder_concat_classes.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_srg_pair() -> None:
    pairs = srg_pairs()
    for pair in pairs:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for row, (label, G) in enumerate(
            [("Rook(4)", pair.g1), ("Shrikhande", pair.g2)]
        ):
            n = G.number_of_nodes()
            pos = nx.spring_layout(G, seed=0)
            A = nx.adjacency_matrix(G).toarray().astype(np.float64)
            ax = axes[row, 0]
            nx.draw(
                G, pos=pos, ax=ax, with_labels=False, node_size=90, node_color="#1f77b4"
            )
            ax.set_title(f"{pair.name} — {label}\nN={n}, deg={int(A.sum(axis=1)[0])}")
            ax = axes[row, 1]
            # Ising correlation matrix.
            psi = ising_ground_state(A)
            C = correlation_matrix_on_state(psi, n)
            im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title("Ising correlation (p=1)")
            plt.colorbar(im, ax=ax, fraction=0.05)
            # 2-particle XY hopping marginal as a proxy for Fig. 3's lower triangle.
            from lib.qpe import qirw_features

            QiRW2 = qirw_features(A, 2, 3)[-1]  # last step
            ax = axes[row, 2]
            im = ax.imshow(QiRW2, cmap="viridis")
            ax.set_title("2-QiRW marginal (step 2)")
            plt.colorbar(im, ax=ax, fraction=0.05)
        fig.suptitle(
            f"{pair.name}: two non-isomorphic SRGs and their quantum features",
            fontsize=11,
        )
        fig.tight_layout()
        out = (
            FIGDIR
            / f"srg_pair_{pair.name.replace('(', '_').replace(')', '').replace(',', '_')}.png"
        )
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


def plot_graph_reg_examples() -> None:
    ds = RandomGraphRegression(num_graphs=6, n_range=(6, 8), p=0.5, seed=1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, item in zip(axes.flat, ds.items):
        n, edges, label = item
        G = _graph_from(n, edges)
        pos = nx.spring_layout(G, seed=0)
        nx.draw(G, pos=pos, ax=ax, with_labels=True, font_size=7, node_color="#bbbbbb")
        ax.set_title(f"N={n}, |E|={len(edges)}\nFiedler value = {label:.3f}")
    fig.suptitle(
        "Random Erdős-Rényi graph regression — label = algebraic connectivity (Fiedler)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGDIR / "graph_reg_examples.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    plot_ladder_types()
    plot_ladder_classes()
    plot_srg_pair()
    plot_graph_reg_examples()


if __name__ == "__main__":
    main()
