"""Visualise the models used in the reproduction.

Two distinct kinds of visualisations:

1. **Feature visualisations** — same graph, four different positional-encoding
   tensors side by side. Lets the reader *see* what RRWP, 1-CQRW, 2-QiRW, and
   ground-state-correlation features each look like and how they differ.

2. **Architecture visualisations** — a forward-pass diagram of GRIT-lite and
   a small photonic-circuit diagram of the photonic adaptation. Both are
   produced as static matplotlib plots so they render in any environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.data import adj_from_record, make_type1  # noqa: E402
from lib.model import GRITLite  # noqa: E402
from lib.pe_factory import pe_batch  # noqa: E402
from lib.photonic import photonic_cqrw_features  # noqa: E402
from lib.qpe import (  # noqa: E402
    cqrw_features,
    ground_state_correlation_eigvecs,
    qirw_features,
    rrwp,
)

FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Feature visualisation: same graph, four encodings side by side.
# ---------------------------------------------------------------------------


def plot_feature_atlas() -> None:
    """RRWP, 1-CQRW, 2-QiRW, ground-state correlation — all on one graph."""
    n, edges = make_type1(5, [1])
    A = adj_from_record(n, edges)
    K = 4

    fig, axes = plt.subplots(4, K + 1, figsize=(3.2 * (K + 1), 11.5))
    times = list(np.linspace(0.3, 2.5, K))
    encodings = [
        ("RRWP (classical $M^k$)", rrwp(A, K)),
        (
            f"1-CQRW $|e^{{-iH_{{XY}}t}}|_{{ij}}^2$, t={times}",
            cqrw_features(A, 1, times),
        ),
        ("2-QiRW $(D^{-1}H_{XY}^{(2)})^k$", qirw_features(A, 2, K)),
        ("GS correlation eigenvector outer products", _gs_outer_products(A, K)),
    ]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = {i: (i // 2, i % 2) for i in range(n)}

    for row, (title, P) in enumerate(encodings):
        # Leftmost cell: the graph itself, decorated with title.
        ax = axes[row, 0]
        nx.draw(
            G,
            pos=pos,
            ax=ax,
            with_labels=True,
            node_color="#cccccc",
            font_size=7,
            node_size=200,
        )
        ax.set_title(title, fontsize=9, loc="left")
        for k in range(K):
            ax = axes[row, k + 1]
            im = ax.imshow(P[k], cmap="viridis")
            ax.set_title(f"k={k}")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle(
        "Atlas of positional encodings on a 12-node type-1 ladder.\n"
        "Each row is one encoding type; columns are the K successive 'depths' the model gets to see.",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGDIR / "feature_atlas.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def _gs_outer_products(A: np.ndarray, K: int) -> np.ndarray:
    feats = ground_state_correlation_eigvecs(A, K)
    N = feats.shape[0]
    out = np.zeros((K, N, N), dtype=np.float64)
    for k in range(K):
        out[k] = np.outer(feats[:, k], feats[:, k])
    return out


# ---------------------------------------------------------------------------
# 2. Architecture: GRITLite forward pass with labelled shapes.
# ---------------------------------------------------------------------------


def plot_grit_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    def box(x, y, w, h, text, color="#dfe9f7"):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="#3a4d70",
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": "#3a4d70", "lw": 1.4},
        )
        if label:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 0.15,
                label,
                ha="center",
                fontsize=8,
                color="#3a4d70",
            )

    box(0.2, 4.6, 2.0, 0.9, "Graph G = (V, E)\nN nodes", color="#f5f5f5")
    box(0.2, 2.6, 2.0, 0.9, "Adjacency A\n(B, N, N)", color="#f5f5f5")
    box(0.2, 0.6, 2.0, 0.9, "Mask\n(B, N)", color="#f5f5f5")

    box(
        3.0,
        3.4,
        2.3,
        1.6,
        "QPE module\nrrwp / 1-CQRW /\n2-QiRW / GS-corr",
        color="#dfe9f7",
    )
    arrow(2.2, 3.0, 3.0, 3.6, "A")

    box(5.8, 3.6, 1.7, 1.2, "PE tensor\n(B, N, N, K)", color="#cfe6cf")
    arrow(5.3, 4.2, 5.8, 4.2)
    box(5.8, 1.6, 1.7, 1.2, "Node embed\n(B, N, D)", color="#fde0a1")
    arrow(2.2, 1.0, 5.8, 2.0, "1·mask")

    box(
        8.0,
        1.6,
        2.2,
        3.2,
        "Edge-attention layer × depth\n\n"
        "scores = QKᵀ/√d + W_e·e\n"
        "gated_attn = softmax · σ(W_g·e)\n"
        "Layer-norm + FFN residual",
        color="#dfe9f7",
    )
    arrow(7.5, 2.2, 8.0, 2.6, "x")
    arrow(7.5, 4.2, 8.0, 4.0, "e")

    box(10.6, 2.9, 1.2, 0.9, "Mean-pool\nover nodes", color="#fde0a1")
    arrow(10.2, 3.3, 10.6, 3.3)

    box(10.6, 1.4, 1.2, 0.9, "Linear head\n→ logits / scalar", color="#cfe6cf")
    arrow(11.2, 2.9, 11.2, 2.3)

    ax.set_title(
        "GRIT-lite forward pass.  Same model is used for graph classification (logits) "
        "and graph regression (scalar) via the `head` flag.",
        fontsize=11,
    )
    out = FIGDIR / "grit_lite_architecture.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 3. Architecture: photonic interferometer for a 1-CQRW.
# ---------------------------------------------------------------------------


def plot_photonic_circuit() -> None:
    n_modes = 4
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left panel: schematic photonic chain.
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_modes + 1)
    # Draw the modes as horizontal lines.
    for m in range(n_modes):
        ax.plot([0.5, 9.5], [m + 0.5, m + 0.5], color="#3a4d70", lw=1.2)
        ax.text(0.05, m + 0.5, f"mode {m}", va="center", fontsize=9)
    # Input photon: |1, 0, ...>.
    ax.scatter([0.6], [0.5], color="#d62728", s=80, zorder=5)
    ax.text(
        0.6,
        0.05,
        "input photon\n|1, 0, 0, 0⟩",
        ha="center",
        fontsize=8,
        color="#d62728",
    )
    # Beam splitters realising U = e^{-i 2A t}.
    for m in range(n_modes - 1):
        x = 2.0 + m * 1.5
        ax.add_patch(
            mpatches.Rectangle(
                (x - 0.25, m + 0.45), 0.5, 1.1, facecolor="#dfe9f7", edgecolor="#3a4d70"
            )
        )
        ax.text(x, m + 1.0, "MZI", ha="center", va="center", fontsize=8)
    # Detectors at the output.
    for m in range(n_modes):
        ax.scatter([9.5], [m + 0.5], color="#1f77b4", marker="D", s=60, zorder=5)
        ax.text(9.7, m + 0.5, f"P_{m}(t)", va="center", fontsize=8, color="#1f77b4")
    ax.set_title(
        "Photonic interferometer for U(t) = exp(-2i A t)\n"
        "(1-CQRW = single-photon photonic walk)",
        fontsize=10,
    )

    # Right panel: photonic 1-CQRW probability matrix.
    A = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.float64,
    )
    P = photonic_cqrw_features(A, 1, [1.0])[0]
    im = axes[1].imshow(P, cmap="viridis")
    axes[1].set_title(
        "Output photon distribution P_{j}(t=1)\n(rows: input mode i, columns: output mode j)"
    )
    axes[1].set_xlabel("output mode j")
    axes[1].set_ylabel("input mode i")
    plt.colorbar(im, ax=axes[1], fraction=0.05)

    fig.suptitle(
        "Photonic MerLin adaptation: the XY hamiltonian evolution maps directly to a passive interferometer.",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGDIR / "photonic_circuit.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 4. Attention map heat-map: what does the model "see" via the edge bias?
# ---------------------------------------------------------------------------


def plot_attention_heatmap() -> None:
    n, edges = make_type1(5, [1])
    A = adj_from_record(n, edges)
    K = 4
    PE = pe_batch(
        torch.tensor(A).unsqueeze(0).float(),
        torch.ones((1, n), dtype=torch.bool),
        encoding="qirw2",
        K=K,
    )
    model = GRITLite(edge_dim=K, node_dim=16, depth=2, num_heads=2, num_classes=2)
    model.eval()
    # Capture the edge-bias contribution of each head from the first layer.
    layer0 = model.layers[0]
    with torch.no_grad():
        e_bias = layer0.edge_bias(PE).permute(0, 3, 1, 2)  # (1, H, N, N)
        gate = torch.sigmoid(layer0.edge_gate(PE)).permute(0, 3, 1, 2)
    H = e_bias.shape[1]
    fig, axes = plt.subplots(2, H, figsize=(4.5 * H, 8))
    for h in range(H):
        im = axes[0, h].imshow(e_bias[0, h].numpy(), cmap="RdBu_r")
        axes[0, h].set_title(f"head {h}: edge-bias term\n(added to QK/√d)")
        plt.colorbar(im, ax=axes[0, h], fraction=0.05)
        im = axes[1, h].imshow(gate[0, h].numpy(), cmap="viridis", vmin=0, vmax=1)
        axes[1, h].set_title(f"head {h}: σ(W_g·e) gate\n(multiplies attention)")
        plt.colorbar(im, ax=axes[1, h], fraction=0.05)
    fig.suptitle(
        "Per-head attention bias and gating computed from the QPE for the same graph.\n"
        "(Random initial weights — the goal is to show how PE flows into attention.)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGDIR / "attention_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# 5. Training curves: existing run directories.
# ---------------------------------------------------------------------------


def plot_training_curves(run_dirs: list[Path], labels: list[str], save_as: str) -> None:
    import json

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metric_key = None
    for run_dir, label in zip(run_dirs, labels):
        metrics = json.loads((run_dir / "metrics.json").read_text())
        hist = metrics["history"]
        axes[0].plot(hist["train_loss"], label=f"{label} train")
        axes[0].plot(hist["val_loss"], linestyle="--", label=f"{label} val")
        if hist["train_acc"]:
            metric_key = "acc"
            axes[1].plot(hist["train_acc"], label=f"{label} train")
            axes[1].plot(hist["val_acc"], linestyle="--", label=f"{label} val")
        else:
            metric_key = "mae"
            axes[1].plot(hist["train_mae"], label=f"{label} train")
            axes[1].plot(hist["val_mae"], linestyle="--", label=f"{label} val")
    axes[0].set_title("loss")
    axes[1].set_title(metric_key or "")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.suptitle("Training curves (reduced-compute smoke runs)", fontsize=11)
    fig.tight_layout()
    out = FIGDIR / save_as
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    plot_feature_atlas()
    plot_grit_architecture()
    plot_photonic_circuit()
    plot_attention_heatmap()


if __name__ == "__main__":
    main()
