"""Plot the fair comparison from ``results/data/fair_comparison.json``.

Per architecture and configuration, seed-mean test MAE (normalized; error bars =
sample std over seeds 0-2) of the eval-selected quantum/hybrid cell (reproduction), the
eval-selected fair classical cell, the eval-selected MerLin photonic cell
(adaptation), and the paper baseline. The no-reservoir controls (linear ridge,
i.e. AR(4), and the paper's kernel on the raw lags) are horizontal lines. Log scale, because
The paper baseline is 2-5x higher than the other systems. Every value is locally executed and
simulated; none is a paper-reported value.

Saves ``results/fair_comparison.png`` and a copy in every ``outdir/sweep_E*-CLASSICAL``.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/plot_fair_comparison.py papers/distributed_qrc/outdir
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
ARCHS = {"srsr": "E1", "mrsr": "E2", "srmr": "E3", "mrmr": "E4"}
SELECTED = [
    ("quantum", "quantum/hybrid, eval-selected", "tab:blue"),
    ("fair_classical", "best classical, eval-selected", "tab:green"),
    ("merlin", "MerLin photonic, eval-selected (adaptation)", "tab:red"),
]


def total_neurons(config: str) -> int:
    """Total reservoir neurons of a configuration label: "10", "10/2" or "2x5"."""

    if "x" in config:
        k, n = config.split("x")
        return int(k) * int(n)
    return int(config.split("/")[0])


def main() -> int:
    outdir = Path(sys.argv[1])
    data = json.loads(
        (PAPER_DIR / "results" / "data" / "fair_comparison.json").read_text()
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, arch in zip(axes.flat, ARCHS):
        a = data["architectures"][arch]
        configs = sorted(a["eval_selected"], key=total_neurons)
        x = range(len(configs))
        for pool, label, color in SELECTED:
            values = [a["eval_selected"][c][pool]["test_mae"] for c in configs]
            ax.errorbar(
                x,
                [v["mean"] for v in values],
                yerr=[v["std"] for v in values],
                label=label,
                color=color,
                marker="o",
                markersize=3,
                capsize=2,
            )
        b0 = [a["per_configuration"][c]["classical|linear"] for c in configs]
        ax.errorbar(
            x,
            [v["mean"] for v in b0],
            yerr=[v["std"] for v in b0],
            label="paper baseline (classical | linear)",
            color="gray",
            linestyle=":",
            marker="s",
            markersize=3,
            capsize=2,
        )
        # raw_lags = no reservoir, the readout sees the 4-lag window directly: it has
        # no neurons, so one value (shared sweep) is drawn as a line at every size.
        for system, label, style in (
            (
                "raw_lags|quantum_kernel_ideal",
                "no reservoir: paper's kernel (same at every size)",
                "--",
            ),
            (
                "raw_lags|linear",
                "no reservoir: linear ridge, AR(4) (same at every size)",
                "-.",
            ),
        ):
            value = a["per_configuration"][configs[0]][system]["mean"]
            ax.axhline(
                value, color="black", linestyle=style, linewidth=0.9, label=label
            )
        ax.set_yscale("log")
        ax.set_xticks(list(x), configs, rotation=30, fontsize=8)
        ax.set_title(arch.upper())
        ax.set_xlabel("configuration (matched total neurons and readout instances)")
        ax.set_ylabel("test MAE (normalized, log scale)")
        ax.grid(alpha=0.3, which="both")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(
        "Quantum vs classical models: simulations, seeds 0-2, mean ± std; "
        "selection on the evaluation split\nMatching: same data, split, features/neurons, readout instances and λ protocol; MerLin is an adaptation",
        fontsize=10,
    )
    fig.tight_layout()
    targets = [PAPER_DIR / "results" / "fair_comparison.png"]
    targets += [
        outdir / f"sweep_{exp}-CLASSICAL" / "fair_comparison.png"
        for exp in ARCHS.values()
        if (outdir / f"sweep_{exp}-CLASSICAL").is_dir()
    ]
    for path in targets:
        fig.savefig(path, dpi=130)
        print(f"wrote {os.path.relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
