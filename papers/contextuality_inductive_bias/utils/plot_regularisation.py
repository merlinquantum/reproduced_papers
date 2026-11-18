"""Figs. 9 / 11-style curves: surrogate test KL and train objective per L2 strength.

Usage (from the repository root)::

    python3 papers/contextuality_inductive_bias/utils/plot_regularisation.py \\
        --sweep-dir <reg_sweep_dir> [--sweep-dir <extension_dir>] --protocol code \\
        --label "partial (1 seed per strength)" \\
        --out papers/contextuality_inductive_bias/results/fig11_surrogate_l1_reg_code.png

One line per L2 strength, darker meaning stronger regularisation (single-hue
blue ramp). The train panel shows the objective including the penalty, as the
author code plots it. Optional ``--reference-sweep`` adds the mean test KL of
a model (``--reference-select kind=biased_quantum,layers=2``) as a horizontal
dashed line at its final step.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_io import describe, load_sweep, matches  # noqa: E402

# Reference sequential blue, from step 250 (ordinal floor on light) to 700.
RAMP = LinearSegmentedColormap.from_list(
    "blue_250_700", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
TEXT, MUTED, GRID, REFERENCE = "#0b0b0b", "#52514e", "#e4e3df", "#eb6834"


def _parse_selector(text: str) -> dict:
    selector = {}
    for item in text.split(","):
        key, value = item.split("=")
        selector[key] = int(value) if value.isdigit() else value
    return selector


def plot(
    sweep_dirs, protocol, label, out, reference_dirs=(), reference_selector=None
) -> Path:
    """Draw the figure and return the written path."""
    records = [
        r
        for d in sweep_dirs
        for r in load_sweep(d)
        if r.candidate["protocol"] == protocol
    ]
    if not records:
        raise ValueError(f"No completed run for protocol {protocol}")
    strengths = sorted({r.candidate["l2"] for r in records})
    spectrum = records[0].candidate["model"]["spectrum"]
    fig, (ax_kl, ax_obj) = plt.subplots(1, 2, figsize=(12, 4.6))
    for index, strength in enumerate(strengths):
        colour = RAMP(index / max(1, len(strengths) - 1))
        for run in (r for r in records if r.candidate["l2"] == strength):
            history = run.history()
            ax_kl.plot(
                history["step"],
                history["test_kl"],
                color=colour,
                linewidth=1.6,
                label=f"L2 = {strength:g}",
            )
            ax_obj.plot(
                history["step"],
                history["train_objective"],
                color=colour,
                linewidth=1.6,
                label=f"L2 = {strength:g}",
            )
    if reference_dirs and reference_selector:
        ref = [
            r
            for d in reference_dirs
            for r in load_sweep(d)
            if r.candidate["protocol"] == protocol
            and matches(r.candidate, reference_selector)
        ]
        if not ref:
            raise ValueError(f"No reference run for {reference_selector}")
        value = float(np.mean([r.metrics["final"]["test_kl"] for r in ref]))
        ax_kl.axhline(
            value,
            color=REFERENCE,
            linestyle="--",
            linewidth=1.6,
            label=f"{describe(ref[0].candidate)}, final mean (n={len(ref)})",
        )
    for ax, ylabel in (
        (ax_kl, "KL divergence (test)"),
        (ax_obj, "train objective (NLL + L2 penalty)"),
    ):
        ax.set_xlabel("training step", color=TEXT)
        ax.set_ylabel(ylabel, color=TEXT)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    ax_kl.set_yscale("log")
    low, high = ax_kl.get_ylim()
    ax_kl.set_ylim(low * 0.9, high)
    handles, labels = ax_kl.get_legend_handles_labels()
    ax_obj.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7,
        labelcolor=TEXT,
        ncol=2,
        loc="upper right",
    )
    fig.suptitle(
        f"Surrogate L={spectrum} with L2 penalty, protocol '{protocol}' ({label}); darker = stronger penalty",
        color=TEXT,
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    shutil.copy(out, Path(sweep_dirs[0]) / out.name)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, action="append", required=True)
    parser.add_argument("--protocol", choices=["paper", "code", "fork"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-sweep", type=Path, action="append", default=[])
    parser.add_argument("--reference-select", type=_parse_selector, default=None)
    args = parser.parse_args(argv)
    plot(
        args.sweep_dir,
        args.protocol,
        args.label,
        args.out,
        args.reference_sweep,
        args.reference_select,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
