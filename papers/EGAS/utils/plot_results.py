"""Plot EGAS reproduction figures from run metrics.

Usage:
    python utils/plot_results.py --wasserstein outdir/run_*/metrics.json
    python utils/plot_results.py --fig1 outdir/run_*/metrics.json
    python utils/plot_results.py --egas outdir/run_WC/metrics.json outdir/run_WQ/metrics.json
    python utils/plot_results.py --fig4-gate paths... --fig4-photonic paths...

Reads structured metrics.json and writes PNGs to both the run dir and ``results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ======================================================================= Constants

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

PAPER_W1 = {
    "PW": 5.2380,
    "WDGV1": 5.1570,
    "DB": 13.9108,
    "WC": 10.8562,
    "WQ": 3.0112,
    "MGT": 3.3036,
    "EGSSD": 3.5619,
}


# ======================================================================= Helpers


def _load(p):
    """Load metrics.json file."""
    return json.loads(Path(p).read_text())


# ======================================================================= Diagnostic Plots


def plot_wasserstein(path):
    """Plot Wasserstein distances (Table I) - diagnostic."""
    m = _load(path)["results"]
    names = [n for n in m if "w1" in m[n]]
    repro = [m[n]["w1"] for n in names]
    paper = [PAPER_W1.get(n, np.nan) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, repro, 0.4, label="reproduced")
    ax.bar(x + 0.2, paper, 0.4, label="paper (Table I)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("1-Wasserstein distance")
    ax.set_title("Table I — input-space class-conditional W1")
    ax.legend()
    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "table1_wasserstein.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote table1_wasserstein.png")


def plot_fig1(path):
    """Plot trace distance vs W1 (Fig 1) - diagnostic."""
    res = _load(path)["results"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, d in res.items():
        ax.plot(d["w1"], d["trace_dist"], "o-", label=f"ZZ {key}")
    ax.set_xlabel("input W1 distance")
    ax.set_ylabel("trace distance")
    ax.set_title("Fig 1 — trace distance vs input W1 (saturating)")
    ax.legend()
    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig1_tracedist_vs_w1.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote fig1_tracedist_vs_w1.png")


# ======================================================================= Fig 3: Bias-Refinement Energy Reduction


def plot_fig3_deltaE_per_candidate(path):
    """Fig 3 (Gate): Bias-refinement surrogate-energy reduction per candidate.

    Shows ΔE per candidate for G and B groups (PW dataset only).
    """
    m = _load(path)
    g = m["delta_E"]["G"]
    b = m["delta_E"]["B"]
    labels = [f"G{i + 1}" for i in range(len(g))] + [f"B{i + 1}" for i in range(len(b))]
    means = g + b

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        range(len(means)),
        means,
        color=["#1f77b4"] * len(g) + ["#d62728"] * len(b),
    )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel("ΔE  (E_before − E_after)")
    ax.set_title(
        "Fig 3 — bias-refinement surrogate-energy reduction\n(gate-based; single refinement run)"
    )
    plt.tight_layout()
    plt.savefig(RESULTS / "fig3_deltaE_per_candidate.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote fig3_deltaE_per_candidate.png")


def plot_fig3_deltaE_per_candidate_photonic(path):
    """Fig 3 (Photonic): Bias-refinement surrogate-energy reduction per candidate.

    Shows ΔE per candidate for G and B groups (PW dataset only).
    """
    m = _load(path)
    g = m["delta_E"]["G"]
    b = m["delta_E"]["B"]
    labels = [f"G{i + 1}" for i in range(len(g))] + [f"B{i + 1}" for i in range(len(b))]
    means = g + b

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        range(len(means)),
        means,
        color=["#1f77b4"] * len(g) + ["#d62728"] * len(b),
    )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel("ΔE  (E_before − E_after)")
    ax.set_title(
        "Fig 3 (Photonic) — bias-refinement surrogate-energy reduction\n(photonic; single refinement run)"
    )
    plt.tight_layout()
    plt.savefig(
        RESULTS / "fig3_deltaE_per_candidate_photonic.png", dpi=130, bbox_inches="tight"
    )
    plt.close()
    print("wrote fig3_deltaE_per_candidate_photonic.png")


# ======================================================================= Fig 4: Group-wise Energy Reduction


def plot_fig4_deltaE_groups(paths):
    """Fig 4 (Gate): Group-wise mean surrogate-energy reduction across datasets.

    Plots mean ΔE for G and B groups with error bars.
    """
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    gm = [np.mean(d["delta_E"]["G"]) for d in data]
    gs = [np.std(d["delta_E"]["G"]) for d in data]
    bm = [np.mean(d["delta_E"]["B"]) for d in data]
    bs = [np.std(d["delta_E"]["B"]) for d in data]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        x - 0.05, gm, yerr=gs, fmt="o", capsize=4, label="G group", color="#1f77b4"
    )
    ax.errorbar(
        x + 0.05, bm, yerr=bs, fmt="s", capsize=4, label="B group", color="#d62728"
    )
    ax.axhline(0, ls="--", color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean ΔE")
    ax.set_title("Fig 4 — group-wise mean surrogate-energy reduction across datasets")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "fig4_deltaE_groups.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote fig4_deltaE_groups.png")


def plot_fig4_deltaE_groups_photonic(paths):
    """Fig 4 (Photonic): Group-wise mean surrogate-energy reduction across datasets.

    Plots mean ΔE for G and B groups with error bars.
    """
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    gm = [np.mean(d["delta_E"]["G"]) for d in data]
    gs = [np.std(d["delta_E"]["G"]) for d in data]
    bm = [np.mean(d["delta_E"]["B"]) for d in data]
    bs = [np.std(d["delta_E"]["B"]) for d in data]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        x - 0.05, gm, yerr=gs, fmt="o", capsize=4, label="G group", color="#1f77b4"
    )
    ax.errorbar(
        x + 0.05, bm, yerr=bs, fmt="s", capsize=4, label="B group", color="#d62728"
    )
    ax.axhline(0, ls="--", color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean ΔE")
    ax.set_title(
        "Fig 4 (Photonic) — group-wise mean surrogate-energy reduction across datasets"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS / "fig4_deltaE_groups_photonic.png", dpi=130, bbox_inches="tight"
    )
    plt.close()
    print("wrote fig4_deltaE_groups_photonic.png")


# ======================================================================= Fig 5: Win/Tie/Loss Comparison


def plot_fig5_win_tie_loss(paths):
    """Fig 5 (Gate): Win/Tie/Loss vs classical linear SVM per dataset.

    Compares quantum kernels against classical baseline across splits.
    """
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]
    n = len(names)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        m = data[names.index(name)]
        models = [
            ("G*", m["G"][m["G_star_idx"]]["wtl_vs_linear"]),
            ("G*(Bias)", m["G_bias"][m["G_bias_star_idx"]]["wtl_vs_linear"]),
            ("B*", m["B"][m["B_star_idx"]]["wtl_vs_linear"]),
            ("B*(Bias)", m["B_bias"][m["B_bias_star_idx"]]["wtl_vs_linear"]),
            ("NQE", m["baselines"]["NQE"]["wtl_vs_linear"]),
            ("ZZ", m["baselines"]["ZZ"]["wtl_vs_linear"]),
        ]
        labels = [a for a, _ in models]
        wins = [w["win"] for _, w in models]
        ties = [w["tie"] for _, w in models]
        loss = [w["lose"] for _, w in models]
        x = np.arange(len(labels))
        ax.bar(x, wins, color="#2c7fb8", label="win")
        ax.bar(x, ties, bottom=wins, color="#f4d03f", label="tie")
        ax.bar(
            x,
            loss,
            bottom=np.array(wins) + np.array(ties),
            color="#d62728",
            label="lose",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_title(f"{name} (W1={m['w1']:.2f})")

    axes[0].set_ylabel("# splits (of {})".format(data[0]["n_splits"]))
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Fig 5 — Win/Tie/Loss vs classical linear SVM")
    plt.tight_layout()
    plt.savefig(RESULTS / "fig5_win_tie_loss.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote fig5_win_tie_loss.png")


def plot_fig5_win_tie_loss_photonic(paths):
    """Fig 5 (Photonic): Win/Tie/Loss vs classical linear SVM per dataset.

    Compares photonic quantum kernels against classical baseline across splits.
    """
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]
    n = len(names)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        m = data[names.index(name)]
        models = [
            ("G*", m["G"][m["G_star_idx"]]["wtl_vs_linear"]),
            ("G*(Bias)", m["G_bias"][m["G_bias_star_idx"]]["wtl_vs_linear"]),
            ("B*", m["B"][m["B_star_idx"]]["wtl_vs_linear"]),
            ("B*(Bias)", m["B_bias"][m["B_bias_star_idx"]]["wtl_vs_linear"]),
            ("NQE", m["baselines"]["NQE"]["wtl_vs_linear"]),
            ("ZZ", m["baselines"]["ZZ"]["wtl_vs_linear"]),
        ]
        labels = [a for a, _ in models]
        wins = [w["win"] for _, w in models]
        ties = [w["tie"] for _, w in models]
        loss = [w["lose"] for _, w in models]
        x = np.arange(len(labels))
        ax.bar(x, wins, color="#2c7fb8", label="win")
        ax.bar(x, ties, bottom=wins, color="#f4d03f", label="tie")
        ax.bar(
            x,
            loss,
            bottom=np.array(wins) + np.array(ties),
            color="#d62728",
            label="lose",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_title(f"{name} (W1={m['w1']:.2f})")

    axes[0].set_ylabel("# splits (of {})".format(data[0]["n_splits"]))
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Fig 5 (Photonic) — Win/Tie/Loss vs classical linear SVM")
    plt.tight_layout()
    plt.savefig(
        RESULTS / "fig5_win_tie_loss_photonic.png", dpi=130, bbox_inches="tight"
    )
    plt.close()
    print("wrote fig5_win_tie_loss_photonic.png")


# ======================================================================= Fig 6: Embedding-Sensitivity IQR


def plot_fig6_iqr(paths):
    """Fig 6 (Gate): Embedding-sensitivity IQR per dataset.

    Shows inter-quartile range of accuracy across embeddings.
    """
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    iqr = [d["embedding_sensitivity_IQR"] for d in data]
    w1 = [d["w1"] for d in data]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(np.arange(len(names)), iqr, color="#7fbf7b")
    for b, w in zip(bars, w1):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.002,
            f"W1={w:.2f}",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("IQR of mean test accuracy")
    ax.set_title("Fig 6 — embedding-sensitivity IQR per dataset (grows with W1)")
    plt.tight_layout()
    plt.savefig(RESULTS / "fig6_iqr.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote fig6_iqr.png")


def plot_fig6_iqr_photonic(paths):
    """Fig 6 (Photonic): Embedding-sensitivity IQR per dataset.

    Shows inter-quartile range of accuracy across embeddings (photonic).
    """
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    iqr = [d["embedding_sensitivity_IQR"] for d in data]
    w1 = [d["w1"] for d in data]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(np.arange(len(names)), iqr, color="#7fbf7b")
    for b, w in zip(bars, w1):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.002,
            f"W1={w:.2f}",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("IQR of mean test accuracy")
    ax.set_title(
        "Fig 6 (Photonic) — embedding-sensitivity IQR per dataset (grows with W1)"
    )
    plt.tight_layout()
    plt.savefig(RESULTS / "fig6_iqr_photonic.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote fig6_iqr_photonic.png")


# ======================================================================= Main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate EGAS reproduction figures from metrics.json files"
    )
    ap.add_argument(
        "--wasserstein", help="Wasserstein diagnostic (Table I)", metavar="PATH"
    )
    ap.add_argument("--fig1", help="Fig 1: Trace distance vs W1", metavar="PATH")
    ap.add_argument(
        "--fig3-gate",
        help="Fig 3 (gate): Energy reduction per candidate",
        metavar="PATH",
    )
    ap.add_argument(
        "--fig3-photonic",
        help="Fig 3 (photonic): Energy reduction per candidate",
        metavar="PATH",
    )
    ap.add_argument(
        "--fig4-gate",
        nargs="+",
        help="Fig 4 (gate): Energy reduction per group",
        metavar="PATHS",
    )
    ap.add_argument(
        "--fig4-photonic",
        nargs="+",
        help="Fig 4 (photonic): Energy reduction per group",
        metavar="PATHS",
    )
    ap.add_argument(
        "--fig5-gate",
        nargs="+",
        help="Fig 5 (gate): Win/Tie/Loss comparison",
        metavar="PATHS",
    )
    ap.add_argument(
        "--fig5-photonic",
        nargs="+",
        help="Fig 5 (photonic): Win/Tie/Loss comparison",
        metavar="PATHS",
    )
    ap.add_argument(
        "--fig6-gate",
        nargs="+",
        help="Fig 6 (gate): Embedding-sensitivity IQR",
        metavar="PATHS",
    )
    ap.add_argument(
        "--fig6-photonic",
        nargs="+",
        help="Fig 6 (photonic): Embedding-sensitivity IQR",
        metavar="PATHS",
    )
    a = ap.parse_args()

    try:
        if a.wasserstein:
            plot_wasserstein(a.wasserstein)
        if a.fig1:
            plot_fig1(a.fig1)
        if a.fig3_gate:
            plot_fig3_deltaE_per_candidate(a.fig3_gate)
        if a.fig3_photonic:
            plot_fig3_deltaE_per_candidate_photonic(a.fig3_photonic)
        if a.fig4_gate:
            plot_fig4_deltaE_groups(a.fig4_gate)
        if a.fig4_photonic:
            plot_fig4_deltaE_groups_photonic(a.fig4_photonic)
        if a.fig5_gate:
            plot_fig5_win_tie_loss(a.fig5_gate)
        if a.fig5_photonic:
            plot_fig5_win_tie_loss_photonic(a.fig5_photonic)
        if a.fig6_gate:
            plot_fig6_iqr(a.fig6_gate)
        if a.fig6_photonic:
            plot_fig6_iqr_photonic(a.fig6_photonic)
    except Exception as e:
        import traceback

        print(f"Error generating plots: {e}", file=__import__("sys").stderr)
        traceback.print_exc(file=__import__("sys").stderr)
        raise
