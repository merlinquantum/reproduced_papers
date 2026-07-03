"""Plot EGAS reproduction figures from run metrics.

Usage:
    python utils/plot_results.py --wasserstein outdir/run_*/metrics.json
    python utils/plot_results.py --fig1 outdir/run_*/metrics.json
    python utils/plot_results.py --egas outdir/run_WC/metrics.json outdir/run_WQ/metrics.json

Reads structured metrics.json (never hardcodes run dirs) and writes PNGs to both the run dir
and ``results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _load(p):
    return json.loads(Path(p).read_text())


def plot_wasserstein(path):
    m = _load(path)["results"]
    names = [n for n in m if "w1" in m[n]]
    repro = [m[n]["w1"] for n in names]
    paper = [PAPER_W1.get(n, np.nan) for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.2, repro, 0.4, label="reproduced")
    plt.bar(x + 0.2, paper, 0.4, label="paper (Table I)")
    plt.xticks(x, names)
    plt.ylabel("1-Wasserstein distance")
    plt.legend()
    plt.title("Table I: input-space class-conditional W1")
    plt.tight_layout()
    for d in (Path(path).parent, RESULTS):
        plt.savefig(d / "table1_wasserstein.png", dpi=120)
    plt.close()
    print("wrote table1_wasserstein.png")


def plot_fig1(path):
    res = _load(path)["results"]
    plt.figure(figsize=(6, 4))
    for key, d in res.items():
        plt.plot(d["w1"], d["trace_dist"], "o-", label=f"ZZ {key}")
    plt.xlabel("input W1 distance")
    plt.ylabel("trace distance")
    plt.title("Fig 1: trace distance vs input W1 (saturating)")
    plt.legend()
    plt.tight_layout()
    for d in (Path(path).parent, RESULTS):
        plt.savefig(d / "fig1_tracedist_vs_w1.png", dpi=120)
    plt.close()
    print("wrote fig1_tracedist_vs_w1.png")


def plot_fig4_energy_reduction(gate_paths, photonic_paths=None):
    """Fig 4: Energy reduction from EGAS search (E_before vs E_after).

    Shows effectiveness of EGAS in finding low-energy circuits.
    """
    fig, axes = plt.subplots(1, 2 if photonic_paths else 1, figsize=(12, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for idx, (paths, title_suffix) in enumerate(
        [(gate_paths, "Gate-based EGAS"), (photonic_paths or [], "Photonic EGAS")]
    ):
        if not paths:
            continue

        ax = axes[idx]
        data = [_load(p) for p in paths]
        names = [d["dataset"] for d in data]

        # Energy reduction from bias refinement
        e_before = [
            np.mean([g["E_before"] for g in d["G_bias"] if g["E_before"] is not None])
            for d in data
        ]
        e_after = [
            np.mean([g["E_after"] for g in d["G_bias"] if g["E_after"] is not None])
            for d in data
        ]

        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w / 2, e_before, w, label="before bias refinement", alpha=0.8)
        ax.bar(x + w / 2, e_after, w, label="after bias refinement", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("mean pairwise energy")
        ax.set_title(f"Fig 4: {title_suffix} Energy Reduction")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig4_energy_reduction.png", dpi=120)
    plt.close()
    print("wrote fig4_energy_reduction.png")


def plot_fig5_circuit_distributions(gate_paths, photonic_paths=None):
    """Fig 5: Distribution of circuit accuracies (G vs G_bias).

    Shows improvement from bias refinement across searched circuits.
    """
    fig, axes = plt.subplots(1, 2 if photonic_paths else 1, figsize=(12, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for idx, (paths, title_suffix) in enumerate(
        [(gate_paths, "Gate-based"), (photonic_paths or [], "Photonic")]
    ):
        if not paths:
            continue

        ax = axes[idx]
        data = [_load(p) for p in paths]
        names = [d["dataset"] for d in data]

        # Accuracy distributions
        g_accs = [np.mean([g["mean_acc"] for g in d["G"]]) for d in data]
        gb_accs = [np.mean([g["mean_acc"] for g in d["G_bias"]]) for d in data]

        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w / 2, g_accs, w, label="G (no bias)", alpha=0.8)
        ax.bar(x + w / 2, gb_accs, w, label="G_bias (refined)", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("mean accuracy")
        ax.set_ylim(0.4, 1.0)
        ax.set_title(f"Fig 5: {title_suffix} Circuit Accuracies")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig5_circuit_distributions.png", dpi=120)
    plt.close()
    print("wrote fig5_circuit_distributions.png")


def plot_egas(paths):
    """Plot accuracy comparison and IQR vs W1 diagnostic."""
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    # accuracy comparison (best G(bias), NQE, ZZ, classical linear)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(names))
    bestGb = [max(g["mean_acc"] for g in d["G_bias"]) for d in data]
    nqe = [d["baselines"]["NQE"]["mean_acc"] for d in data]
    zz = [d["baselines"]["ZZ"]["mean_acc"] for d in data]
    lin = [d["baselines"]["classical_linear"]["mean_acc"] for d in data]
    w = 0.2
    ax[0].bar(x - 1.5 * w, bestGb, w, label="best G*(bias)")
    ax[0].bar(x - 0.5 * w, nqe, w, label="NQE")
    ax[0].bar(x + 0.5 * w, zz, w, label="ZZ")
    ax[0].bar(x + 1.5 * w, lin, w, label="classical linear")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names)
    ax[0].set_ylabel("mean test acc")
    ax[0].set_ylim(0.4, 1.0)
    ax[0].legend()
    ax[0].set_title("QKSVM accuracy (Fig 7)")

    # IQR vs W1 (Fig 6 diagnostic)
    iqr = [d.get("embedding_sensitivity_IQR", 0) for d in data]
    w1 = [d["w1"] for d in data]
    ax[1].scatter(w1, iqr)
    for xi, yi, ni in zip(w1, iqr, names):
        ax[1].annotate(ni, (xi, yi))
    ax[1].set_xlabel("input W1 distance")
    ax[1].set_ylabel("embedding-sensitivity IQR")
    ax[1].set_title("Fig 6: IQR vs W1 (larger W1 -> larger IQR)")
    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "egas_summary.png", dpi=120)
    for p in paths:
        plt.savefig(Path(p).parent / "egas_summary.png", dpi=120)
    plt.close()
    print("wrote egas_summary.png")


def plot_egas_photonic(paths):
    """Plot photonic accuracy comparison and IQR vs W1 diagnostic."""
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    # accuracy comparison (best G(bias), NQE, ZZ, classical linear)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(names))
    bestGb = [max(g["mean_acc"] for g in d["G_bias"]) for d in data]
    nqe = [d["baselines"]["NQE"]["mean_acc"] for d in data]
    zz = [d["baselines"]["ZZ"]["mean_acc"] for d in data]
    lin = [d["baselines"]["classical_linear"]["mean_acc"] for d in data]
    w = 0.2
    ax[0].bar(x - 1.5 * w, bestGb, w, label="best G*(bias)")
    ax[0].bar(x - 0.5 * w, nqe, w, label="NQE")
    ax[0].bar(x + 0.5 * w, zz, w, label="ZZ")
    ax[0].bar(x + 1.5 * w, lin, w, label="classical linear")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names)
    ax[0].set_ylabel("mean test acc")
    ax[0].set_ylim(0.4, 1.0)
    ax[0].legend()
    ax[0].set_title("Photonic QKSVM accuracy")

    # IQR vs W1 (Fig 6 diagnostic)
    iqr = [d.get("embedding_sensitivity_IQR", 0) for d in data]
    w1 = [d["w1"] for d in data]
    ax[1].scatter(w1, iqr, color="orange")
    for xi, yi, ni in zip(w1, iqr, names):
        ax[1].annotate(ni, (xi, yi))
    ax[1].set_xlabel("input W1 distance")
    ax[1].set_ylabel("embedding-sensitivity IQR")
    ax[1].set_title("Photonic: IQR vs W1 diagnostic")
    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "egas_summary_photonic.png", dpi=120)
    for p in paths:
        plt.savefig(Path(p).parent / "egas_summary_photonic.png", dpi=120)
    plt.close()
    print("wrote egas_summary_photonic.png")


def plot_fig4_gate_only(paths):
    """Fig 4: Energy reduction from EGAS search (gate-based only)."""
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 4))
    e_before = [
        np.mean([g["E_before"] for g in d["G_bias"] if g["E_before"] is not None])
        for d in data
    ]
    e_after = [
        np.mean([g["E_after"] for g in d["G_bias"] if g["E_after"] is not None])
        for d in data
    ]

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, e_before, w, label="before bias refinement", alpha=0.8)
    ax.bar(x + w / 2, e_after, w, label="after bias refinement", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean pairwise energy")
    ax.set_title("Fig 4: Gate-based EGAS Energy Reduction")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig4_gate_energy_reduction.png", dpi=120)
    plt.close()
    print("wrote fig4_gate_energy_reduction.png")


def plot_fig4_photonic_only(paths):
    """Fig 4: Energy reduction from photonic EGAS search (photonic-based only)."""
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 4))
    e_before = [
        np.mean([g["E_before"] for g in d["G_bias"] if g["E_before"] is not None])
        for d in data
    ]
    e_after = [
        np.mean([g["E_after"] for g in d["G_bias"] if g["E_after"] is not None])
        for d in data
    ]

    x = np.arange(len(names))
    w = 0.35
    ax.bar(
        x - w / 2,
        e_before,
        w,
        label="before bias refinement",
        alpha=0.8,
        color="orange",
    )
    ax.bar(
        x + w / 2,
        e_after,
        w,
        label="after bias refinement",
        alpha=0.8,
        color="darkorange",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean pairwise energy")
    ax.set_title("Fig 4: Photonic EGAS Energy Reduction")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig4_photonic_energy_reduction.png", dpi=120)
    plt.close()
    print("wrote fig4_photonic_energy_reduction.png")


def plot_fig5_gate_only(paths):
    """Fig 5: Distribution of circuit accuracies (gate-based only)."""
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 4))
    g_accs = [np.mean([g["mean_acc"] for g in d["G"]]) for d in data]
    gb_accs = [np.mean([g["mean_acc"] for g in d["G_bias"]]) for d in data]

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, g_accs, w, label="G (no bias)", alpha=0.8)
    ax.bar(x + w / 2, gb_accs, w, label="G_bias (refined)", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Fig 5: Gate-based Circuit Accuracies")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig5_gate_circuit_distributions.png", dpi=120)
    plt.close()
    print("wrote fig5_gate_circuit_distributions.png")


def plot_fig5_photonic_only(paths):
    """Fig 5: Distribution of circuit accuracies (photonic-based only)."""
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 4))
    g_accs = [np.mean([g["mean_acc"] for g in d["G"]]) for d in data]
    gb_accs = [np.mean([g["mean_acc"] for g in d["G_bias"]]) for d in data]

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, g_accs, w, label="G (no bias)", alpha=0.8, color="orange")
    ax.bar(
        x + w / 2, gb_accs, w, label="G_bias (refined)", alpha=0.8, color="darkorange"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Fig 5: Photonic Circuit Accuracies")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig5_photonic_circuit_distributions.png", dpi=120)
    plt.close()
    print("wrote fig5_photonic_circuit_distributions.png")


def plot_fig6_gate_only(paths):
    """Fig 6: IQR vs W1 diagnostic (gate-based only)."""
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    iqr = [d.get("embedding_sensitivity_IQR", 0) for d in data]
    w1 = [d["w1"] for d in data]
    ax.scatter(w1, iqr, s=100, alpha=0.7)
    for xi, yi, ni in zip(w1, iqr, names):
        ax.annotate(ni, (xi, yi), fontsize=10)
    ax.set_xlabel("input W1 distance")
    ax.set_ylabel("embedding-sensitivity IQR")
    ax.set_title("Fig 6: Gate-based EGAS (IQR vs W1)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig6_gate_diagnostic.png", dpi=120)
    plt.close()
    print("wrote fig6_gate_diagnostic.png")


def plot_fig6_photonic_only(paths):
    """Fig 6: IQR vs W1 diagnostic (photonic-based only)."""
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    iqr = [d.get("embedding_sensitivity_IQR", 0) for d in data]
    w1 = [d["w1"] for d in data]
    ax.scatter(w1, iqr, s=100, alpha=0.7, color="orange")
    for xi, yi, ni in zip(w1, iqr, names):
        ax.annotate(ni, (xi, yi), fontsize=10)
    ax.set_xlabel("input W1 distance")
    ax.set_ylabel("embedding-sensitivity IQR")
    ax.set_title("Fig 6: Photonic EGAS (IQR vs W1)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig6_photonic_diagnostic.png", dpi=120)
    plt.close()
    print("wrote fig6_photonic_diagnostic.png")


def plot_fig7_gate_only(paths):
    """Fig 7: Accuracy comparison (gate-based only)."""
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    best_g = [max(g["mean_acc"] for g in d["G_bias"]) for d in data]
    nqe = [d["baselines"]["NQE"]["mean_acc"] for d in data]
    zz = [d["baselines"]["ZZ"]["mean_acc"] for d in data]
    lin = [d["baselines"]["classical_linear"]["mean_acc"] for d in data]

    x = np.arange(len(names))
    w = 0.2
    ax.bar(x - 1.5 * w, best_g, w, label="best G*(bias)", alpha=0.8)
    ax.bar(x - 0.5 * w, nqe, w, label="NQE", alpha=0.8)
    ax.bar(x + 0.5 * w, zz, w, label="ZZ", alpha=0.8)
    ax.bar(x + 1.5 * w, lin, w, label="classical linear", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean test accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Fig 7: Gate-based QKSVM Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig7_gate_accuracy.png", dpi=120)
    plt.close()
    print("wrote fig7_gate_accuracy.png")


def plot_fig7_photonic_only(paths):
    """Fig 7: Accuracy comparison (photonic-based only)."""
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    best_g_phot = [max(g["mean_acc"] for g in d["G_bias"]) for d in data]
    nqe_phot = [d["baselines"]["NQE"]["mean_acc"] for d in data]
    zz_phot = [d["baselines"]["ZZ"]["mean_acc"] for d in data]
    lin_phot = [d["baselines"]["classical_linear"]["mean_acc"] for d in data]

    x = np.arange(len(names))
    w = 0.2
    ax.bar(
        x - 1.5 * w, best_g_phot, w, label="best G*(bias)", alpha=0.8, color="orange"
    )
    ax.bar(x - 0.5 * w, nqe_phot, w, label="NQE", alpha=0.8, color="darkorange")
    ax.bar(x + 0.5 * w, zz_phot, w, label="ZZ", alpha=0.8, color="gold")
    ax.bar(x + 1.5 * w, lin_phot, w, label="classical linear", alpha=0.8, color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean test accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Fig 7: Photonic QKSVM Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig7_photonic_accuracy.png", dpi=120)
    plt.close()
    print("wrote fig7_photonic_accuracy.png")


def plot_fig4_deltaE_groups(paths):
    """Fig 4: Group-wise mean surrogate-energy reduction across datasets.

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
    for d in (RESULTS,):
        plt.savefig(d / "fig4_deltaE_groups.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote fig4_deltaE_groups.png")


def plot_fig4_deltaE_groups_photonic(paths):
    """Fig 4 Photonic: Group-wise mean surrogate-energy reduction across datasets."""
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
    for d in (RESULTS,):
        plt.savefig(d / "fig4_deltaE_groups_photonic.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote fig4_deltaE_groups_photonic.png")


def plot_fig5_win_tie_loss(paths):
    """Fig 5: Win/Tie/Loss vs classical linear SVM per dataset."""
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
    for d in (RESULTS,):
        plt.savefig(d / "fig5_win_tie_loss.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote fig5_win_tie_loss.png")


def plot_fig5_win_tie_loss_photonic(paths):
    """Fig 5 Photonic: Win/Tie/Loss vs classical linear SVM per dataset."""
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
    for d in (RESULTS,):
        plt.savefig(d / "fig5_win_tie_loss_photonic.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote fig5_win_tie_loss_photonic.png")


def plot_fig6_iqr(paths):
    """Fig 6: Embedding-sensitivity IQR per dataset."""
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
    for d in (RESULTS,):
        plt.savefig(d / "fig6_iqr.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote fig6_iqr.png")


def plot_fig6_iqr_photonic(paths):
    """Fig 6 Photonic: Embedding-sensitivity IQR per dataset."""
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
    for d in (RESULTS,):
        plt.savefig(d / "fig6_iqr_photonic.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote fig6_iqr_photonic.png")


def plot_fig6_diagnostic_photonic(gate_paths, photonic_paths):
    """Fig 6: IQR vs W1 comparison (gate vs photonic).

    Diagnostic showing relationship between input W1 and embedding sensitivity.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Gate-based
    gate_data = [_load(p) for p in gate_paths]
    gate_names = [d["dataset"] for d in gate_data]
    gate_w1 = [d["w1"] for d in gate_data]
    gate_iqr = [d.get("embedding_sensitivity_IQR", 0) for d in gate_data]

    ax1.scatter(gate_w1, gate_iqr, s=100, alpha=0.7, label="gate-based")
    for xi, yi, ni in zip(gate_w1, gate_iqr, gate_names):
        ax1.annotate(ni, (xi, yi), fontsize=9)
    ax1.set_xlabel("input W1 distance")
    ax1.set_ylabel("embedding-sensitivity IQR")
    ax1.set_title("Fig 6a: Gate-based EGAS (IQR vs W1)")
    ax1.grid(alpha=0.3)

    # Photonic
    photonic_data = [_load(p) for p in photonic_paths]
    photonic_names = [d["dataset"] for d in photonic_data]
    photonic_w1 = [d["w1"] for d in photonic_data]
    photonic_iqr = [d.get("embedding_sensitivity_IQR", 0) for d in photonic_data]

    ax2.scatter(
        photonic_w1, photonic_iqr, s=100, alpha=0.7, color="orange", label="photonic"
    )
    for xi, yi, ni in zip(photonic_w1, photonic_iqr, photonic_names):
        ax2.annotate(ni, (xi, yi), fontsize=9)
    ax2.set_xlabel("input W1 distance")
    ax2.set_ylabel("embedding-sensitivity IQR")
    ax2.set_title("Fig 6b: Photonic EGAS (IQR vs W1)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig6_diagnostic_comparison.png", dpi=120)
    plt.close()
    print("wrote fig6_diagnostic_comparison.png")


def plot_fig7_accuracy_photonic(gate_paths, photonic_paths):
    """Fig 7: Accuracy comparison (gate vs photonic implementations).

    Shows performance of EGAS-optimized embeddings vs baselines for both platforms.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gate-based
    gate_data = [_load(p) for p in gate_paths]
    gate_names = [d["dataset"] for d in gate_data]

    best_g = [max(g["mean_acc"] for g in d["G_bias"]) for d in gate_data]
    nqe = [d["baselines"]["NQE"]["mean_acc"] for d in gate_data]
    zz = [d["baselines"]["ZZ"]["mean_acc"] for d in gate_data]
    lin = [d["baselines"]["classical_linear"]["mean_acc"] for d in gate_data]

    x = np.arange(len(gate_names))
    w = 0.2
    ax1.bar(x - 1.5 * w, best_g, w, label="best G*(bias)", alpha=0.8)
    ax1.bar(x - 0.5 * w, nqe, w, label="NQE", alpha=0.8)
    ax1.bar(x + 0.5 * w, zz, w, label="ZZ", alpha=0.8)
    ax1.bar(x + 1.5 * w, lin, w, label="classical linear", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(gate_names)
    ax1.set_ylabel("mean test accuracy")
    ax1.set_ylim(0.4, 1.0)
    ax1.set_title("Fig 7a: Gate-based QKSVM")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Photonic
    photonic_data = [_load(p) for p in photonic_paths]
    photonic_names = [d["dataset"] for d in photonic_data]

    best_g_phot = [max(g["mean_acc"] for g in d["G_bias"]) for d in photonic_data]
    nqe_phot = [d["baselines"]["NQE"]["mean_acc"] for d in photonic_data]
    zz_phot = [d["baselines"]["ZZ"]["mean_acc"] for d in photonic_data]
    lin_phot = [d["baselines"]["classical_linear"]["mean_acc"] for d in photonic_data]

    x = np.arange(len(photonic_names))
    ax2.bar(x - 1.5 * w, best_g_phot, w, label="best G*(bias)", alpha=0.8)
    ax2.bar(x - 0.5 * w, nqe_phot, w, label="NQE", alpha=0.8)
    ax2.bar(x + 0.5 * w, zz_phot, w, label="ZZ", alpha=0.8)
    ax2.bar(x + 1.5 * w, lin_phot, w, label="classical linear", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(photonic_names)
    ax2.set_ylabel("mean test accuracy")
    ax2.set_ylim(0.4, 1.0)
    ax2.set_title("Fig 7b: Photonic QKSVM")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "fig7_accuracy_comparison.png", dpi=120)
    plt.close()
    print("wrote fig7_accuracy_comparison.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wasserstein")
    ap.add_argument("--fig1")
    ap.add_argument("--egas", nargs="+")
    ap.add_argument("--egas-photonic", nargs="+", help="photonic EGAS for summary plot")
    ap.add_argument("--gate-egas", nargs="+", help="gate-based EGAS metrics")
    ap.add_argument("--photonic-egas", nargs="+", help="photonic EGAS metrics")
    # Individual figures (gate-based)
    ap.add_argument("--fig4-gate", nargs="+", help="gate-based EGAS for Fig 4")
    ap.add_argument("--fig5-gate", nargs="+", help="gate-based EGAS for Fig 5")
    ap.add_argument("--fig6-gate", nargs="+", help="gate-based EGAS for Fig 6")
    ap.add_argument("--fig7-gate", nargs="+", help="gate-based EGAS for Fig 7")
    # Individual figures (photonic-based)
    ap.add_argument("--fig4-photonic", nargs="+", help="photonic EGAS for Fig 4")
    ap.add_argument("--fig5-photonic", nargs="+", help="photonic EGAS for Fig 5")
    ap.add_argument("--fig6-photonic", nargs="+", help="photonic EGAS for Fig 6")
    ap.add_argument("--fig7-photonic", nargs="+", help="photonic EGAS for Fig 7")
    # Comparison figures
    ap.add_argument("--fig4", nargs="+", help="gate-based EGAS for Fig 4 comparison")
    ap.add_argument("--fig5", nargs="+", help="gate-based EGAS for Fig 5 comparison")
    ap.add_argument("--fig6", nargs="+", help="gate-based EGAS for Fig 6 comparison")
    ap.add_argument("--fig7", nargs="+", help="gate-based EGAS for Fig 7 comparison")
    a = ap.parse_args()

    if a.wasserstein:
        plot_wasserstein(a.wasserstein)
    if a.fig1:
        plot_fig1(a.fig1)
    if a.egas:
        plot_egas(a.egas)
    if a.egas_photonic:
        plot_egas_photonic(a.egas_photonic)
    # Individual gate-based figures
    if a.fig4_gate:
        plot_fig4_gate_only(a.fig4_gate)
    if a.fig5_gate:
        plot_fig5_gate_only(a.fig5_gate)
    if a.fig6_gate:
        plot_fig6_gate_only(a.fig6_gate)
    if a.fig7_gate:
        plot_fig7_gate_only(a.fig7_gate)
    # Individual photonic figures
    if a.fig4_photonic:
        plot_fig4_photonic_only(a.fig4_photonic)
    if a.fig5_photonic:
        plot_fig5_photonic_only(a.fig5_photonic)
    if a.fig6_photonic:
        plot_fig6_photonic_only(a.fig6_photonic)
    if a.fig7_photonic:
        plot_fig7_photonic_only(a.fig7_photonic)
    # Comparison figures
    if a.fig4 and a.fig4_photonic:
        plot_fig4_energy_reduction(a.fig4, a.fig4_photonic)
    if a.fig5 and a.fig5_photonic:
        plot_fig5_circuit_distributions(a.fig5, a.fig5_photonic)
    if a.fig6 and a.fig6_photonic:
        plot_fig6_diagnostic_photonic(a.fig6, a.fig6_photonic)
    if a.fig7 and a.fig7_photonic:
        plot_fig7_accuracy_photonic(a.fig7, a.fig7_photonic)
