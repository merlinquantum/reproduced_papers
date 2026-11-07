"""Generate the curated figures from the results/ analysis JSONs.

All inputs are tracked files under results/ (written by
``utils/analyze_powered_runs.py`` and the latency benchmark), so every
figure regenerates without the raw outdir runs:

    python utils/plot_results.py            # writes the three fig_*.png

Figures:
  fig_fold_diff_distribution.png  per-fold paired AUCPR differences at
                                  gamma=0.5 — shows the left-skew that
                                  drives negative means despite
                                  majority-positive folds
  fig_mean_diff_vs_gamma.png      mean paired difference with 95% CI
                                  across the router-threshold sweep
  fig_latency.png                 Table 1 latency comparison, measured
                                  vs paper-reported (log scale)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"

# Fixed identity -> (label, color) assignment across all figures.
CONFIGS = [
    ("gate_powered", "gate", "#1f77b4"),
    ("gate_tuned_xgboost_powered", "gate + tuned XGB", "#17becf"),
    ("gate_validation_router_powered", "gate, val. router", "#9467bd"),
    ("classical_powered", "classical ablation", "#2ca02c"),
    ("photonic_fixed_powered", "photonic fixed", "#ff7f0e"),
    ("photonic_trainable_powered", "photonic trainable", "#d62728"),
]


def load(name: str) -> dict:
    return json.loads((RESULTS / f"analysis_{name}.json").read_text())


def fig_fold_diff_distribution(gamma: str = "0.5") -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rng = np.random.default_rng(0)
    for i, (name, _label, color) in enumerate(CONFIGS):
        stats = load(name)["per_threshold"][gamma]
        diffs = np.array(stats["fold_diffs"])
        x = i + rng.uniform(-0.16, 0.16, diffs.size)
        ax.scatter(x, diffs, s=10, alpha=0.45, color=color, edgecolors="none")
        ax.hlines(
            stats["mean_diff"],
            i - 0.28,
            i + 0.28,
            color=color,
            linewidth=2.5,
            label=None,
        )
        ax.hlines(
            stats["median_diff"],
            i - 0.28,
            i + 0.28,
            color=color,
            linewidth=1.2,
            linestyle="--",
        )
    ax.axhline(0.0, color="0.4", linewidth=1)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels(
        [label for _, label, _ in CONFIGS], rotation=18, ha="right", fontsize=9
    )
    ax.set_ylabel(f"AUCPR difference, MoE − XGBoost (γ={gamma})")
    ax.set_title(
        "Per-fold paired differences, n=100: a left-skewed minority of "
        "collapse folds drives negative means\n(solid = mean, "
        "dashed = median)",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = RESULTS / "fig_fold_diff_distribution.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def fig_mean_diff_vs_gamma() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for name, label, color in CONFIGS:
        per = load(name)["per_threshold"]
        gammas = sorted(per, key=float)
        means = [per[g]["mean_diff"] for g in gammas]
        lo = [per[g]["ci95"][0] for g in gammas]
        hi = [per[g]["ci95"][1] for g in gammas]
        x = [float(g) for g in gammas]
        ax.plot(x, means, marker="o", markersize=4, color=color, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, edgecolor="none")
    ax.axhline(0.0, color="0.4", linewidth=1)
    ax.set_xlabel("Router threshold γ")
    ax.set_ylabel("Mean AUCPR difference, MoE − XGBoost")
    ax.set_title(
        "Powered runs (n=100 folds): mean paired difference with 95% CI", fontsize=11
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = RESULTS / "fig_mean_diff_vs_gamma.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def fig_latency() -> None:
    measured = json.loads((RESULTS / "latency_benchmark.json").read_text())
    paper = {"QMKL": 123.9, "GFM": 48.4, "GQC": 0.089}  # ms/sample, Table 1
    methods = ["QMKL", "GFM", "GQC"]
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(
        x - width / 2,
        [measured[m]["ms_per_sample"] for m in methods],
        width,
        yerr=[measured[m]["std_ms_per_sample"] for m in methods],
        capsize=3,
        label="measured (this reproduction, CPU)",
        color="#1f77b4",
    )
    ax.bar(
        x + width / 2,
        [paper[m] for m in methods],
        width,
        label="paper Table 1",
        color="#ff7f0e",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Inference latency (ms / sample, log scale)")
    ax.set_title(
        "Table 1 latency: ranking reproduced, magnitude not\n"
        "(paper claims GQC 542–1387× faster; measured 3–10×)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = RESULTS / "fig_latency.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    fig_fold_diff_distribution()
    fig_mean_diff_vs_gamma()
    fig_latency()


if __name__ == "__main__":
    main()
