"""Landsat figures (paper Fig. 4, Fig. 3b/3c, Fig. A3/A4 analogues) from a summary JSON.

Usage (from the paper folder)::

    python utils/plot_landsat_readouts.py --summary results/landsat_n16_summary.json \
        --run-dir outdir/run_A

The summary comes from ``utils/summarize_landsat.py``. Figures are written to
``results/`` and to ``--run-dir`` (the readouts run). Every figure title carries the
result-scope label stored in the summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
READOUT_ORDER = ("x", "pauli", "et", "etcut")
# Paper Fig. 3b/3c read off the plot (124 qubits, hardware).
PAPER_SHOTS = {100: 0.22, 1000: 0.46, 10000: 0.72, 100000: 0.78}
PAPER_TRAIN_SIZE = {10: 0.48, 688: 0.79}


def _save(fig, name: str, outputs: list[Path]) -> None:
    for directory in outputs:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_readouts(summary: dict, outputs: list[Path]) -> None:
    readouts = summary["qelm"]["readouts"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, weight in zip(axes, ("w1", "w2"), strict=True):
        positions = range(len(READOUT_ORDER))
        for offset, scaling, hatch in ((-0.2, "unit", ""), (0.2, "scaled", "//")):
            names = [f"{weight}_{kind}_{scaling}" for kind in READOUT_ORDER]
            means = [readouts[name]["f1_macro"]["mean"] for name in names]
            stds = [readouts[name]["f1_macro"]["std"] for name in names]
            ax.bar(
                [p + offset for p in positions],
                means,
                0.4,
                yerr=stds,
                capsize=3,
                hatch=hatch,
                color="#4C78A8" if scaling == "unit" else "#9ECAE9",
                edgecolor="black",
                label=f"ours, {scaling}",
            )
            paper = [readouts[name]["paper_f1_macro"] for name in names]
            ax.scatter(
                [
                    p + offset
                    for p, v in zip(positions, paper, strict=True)
                    if v is not None
                ],
                [v for v in paper if v is not None],
                marker="*",
                s=90,
                color="#E45756",
                zorder=5,
                label="paper (124 q)" if scaling == "unit" else None,
            )
        if "classical" in summary:
            classical = summary["classical"]["paper_baselines"]
            for name, style in (("random_forest", "-"), ("knn", "--"), ("ridge", ":")):
                ax.axhline(
                    classical[name]["f1_macro"]["mean"],
                    color="gray",
                    linestyle=style,
                    linewidth=1,
                    label=f"{name} (ours)" if weight == "w1" else None,
                )
            for label, entry in summary["classical"]["matched_elm"].items():
                if label.startswith("fourier") and label.endswith("noise_std=0.0"):
                    n_features = int(label.split("n_features=")[1].split("|")[0])
                    matched = 3 * 16 if weight == "w1" else 15 * 8
                    if n_features == matched:
                        ax.axhline(
                            entry["f1_macro"]["mean"],
                            color="#54A24B",
                            linewidth=1.5,
                            label=f"RFF, {n_features} features (matched)",
                        )
        ax.set_xticks(list(positions), ["X", "Pauli", "ET", "ET-cut"])
        ax.set_title(f"weight-{weight[1]} readouts")
        ax.set_ylim(0.5, 0.9)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("Test F1 macro")
    fig.suptitle(
        f"Landsat QELM readouts, {summary['label']} ({summary['scope']}; mean ± std over seeds)"
    )
    _save(fig, f"landsat_{summary['label']}_readouts.png", outputs)


def plot_curves(summary: dict, outputs: list[Path]) -> None:
    qelm = summary["qelm"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for readout, curve in qelm["shots_sweep"].items():
        xs = [int(v) for v in curve]
        axes[0].errorbar(
            xs,
            [p["f1_macro"]["mean"] for p in curve.values()],
            yerr=[p["f1_macro"]["std"] for p in curve.values()],
            marker="o",
            label=f"ours {readout}",
        )
    axes[0].plot(
        list(PAPER_SHOTS),
        list(PAPER_SHOTS.values()),
        "r*--",
        label="paper (124 q, read off plot)",
    )
    axes[0].set(
        xscale="log",
        xlabel="shots per sample",
        ylabel="Test F1 macro",
        title="Fig. 3b analogue",
    )
    for readout, curve in qelm["train_size_sweep"].items():
        xs = [int(v) for v in curve]
        axes[1].errorbar(
            xs,
            [p["f1_macro"]["mean"] for p in curve.values()],
            yerr=[p["f1_macro"]["std"] for p in curve.values()],
            marker="o",
            label=f"ours {readout}",
        )
    axes[1].plot(
        list(PAPER_TRAIN_SIZE),
        list(PAPER_TRAIN_SIZE.values()),
        "r*",
        label="paper endpoints",
    )
    axes[1].set(xscale="log", xlabel="training-set size", title="Fig. 3c analogue")
    for readout, curve in qelm["cutoff_sweep"].items():
        xs = [float(v) for v in curve]
        axes[2].errorbar(
            xs,
            [p["f1_macro"]["mean"] for p in curve.values()],
            yerr=[p["f1_macro"]["std"] for p in curve.values()],
            marker="o",
            label=readout,
        )
        if readout.endswith("scaled"):
            axes[3].plot(
                xs,
                [p["n_eigentasks_retained"]["mean"] for p in curve.values()],
                "o-",
                label="retained",
            )
            axes[3].plot(
                xs,
                [p["rec_cut"]["mean"] for p in curve.values()],
                "s-",
                label="REC C_lambda",
            )
            axes[3].axhline(
                next(iter(curve.values()))["n_eigentasks_total"]["mean"],
                color="gray",
                linestyle="--",
                label="total eigentasks",
            )
    axes[2].axvline(1.0, color="k", linestyle=":")
    axes[2].set(
        xscale="log", xlabel="NSR cutoff lambda", title="Fig. A3 analogue (w2 ET)"
    )
    axes[3].axvline(1.0, color="k", linestyle=":")
    axes[3].set(xscale="log", xlabel="NSR cutoff lambda", title="Fig. A4 analogue")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(f"Landsat sensitivity curves, {summary['label']} ({summary['scope']})")
    _save(fig, f"landsat_{summary['label']}_curves.png", outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=PAPER_DIR / "results")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    outputs = [args.outdir, args.run_dir]
    plot_readouts(summary, outputs)
    plot_curves(summary, outputs)


if __name__ == "__main__":
    main()
