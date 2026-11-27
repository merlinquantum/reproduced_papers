"""Phase 5 figure: paired advantage of the tuned quantum neuron over each tuned baseline.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/plot_p5.py \
        papers/fermi_dirac_neurons/results/p5_comparison.json \
        papers/fermi_dirac_neurons/outdir/sweeps/p5_final_v1

Reads the comparison written by ``utils/analyze_p5.py`` (primary metric only)
and draws, per task, the median paired advantage over seeds 0-9 with its
min-max range: decades of test MSE (regression, MerLin) or points of test
accuracy (classification). Positive = the quantum (or photonic) neuron is
better. B4 is drawn hollow: it is matched on data, not on parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))

from fermi_dirac_neurons.lib.plotting import (  # noqa: E402
    COLOURS,
    GRID,
    apply_style,
    finish,
    save,
)

BASELINE_STYLE = {
    "B1": (COLOURS["commuting"], "o", "B1: paper's commuting neuron (IM / FCIM)"),
    "B2": (COLOURS["b2"], "o", "B2: classical neuron on the same Pauli features"),
    "B3": (COLOURS["linear"], "o", "B3: quantum linear Tr[H(θ)ρ]"),
    "B4": (
        "#7a7a7a",
        "D",
        "B4: linear model on all ≤2-local Pauli expectations (reference, more parameters)",
    ),
}


def _kind(model: str) -> str:
    if model.startswith("b4"):
        return "B4"
    if model.startswith("b2") or model == "correlation_tanh":
        return "B2"
    if model.startswith("classical") or model == "photon_count_tanh":
        return "B1"
    return "B3"


def main() -> None:
    """Draw the paired-advantage figure."""
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("comparison", type=Path)
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    apply_style()
    result = json.loads(args.comparison.read_text())
    primary = [
        row
        for row in result["comparisons"]
        if row["metric"] in ("validation_mse", "validation_accuracy_linear")
        and "median_advantage" in row
    ]
    panels = [
        ("C4", "Fig. 7 (C4): test MSE, decades"),
        ("C6", "Fig. 9 (C6): test MSE, decades"),
        ("C5", "Fig. 8 (C5): test accuracy, points"),
        ("MerLin", "MerLin adaptation: test MSE, decades"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(13, 4.2), layout="constrained")
    handles = {}
    for axis, (claim, title) in zip(axes, panels):
        rows = [row for row in primary if row["claim"] == claim]
        tasks = list(dict.fromkeys(row["task"] for row in rows))
        offsets = {"B1": -0.27, "B2": -0.09, "B3": 0.09, "B4": 0.27}
        for row in rows:
            kind = _kind(row["other_model"])
            colour, marker, label = BASELINE_STYLE[kind]
            y = tasks.index(row["task"]) + offsets[kind]
            axis.hlines(
                y,
                row["min_advantage"],
                row["max_advantage"],
                color=colour,
                linewidth=1.2,
            )
            handle = axis.scatter(
                row["median_advantage"],
                y,
                color=colour if kind != "B4" else "none",
                edgecolors=colour,
                marker=marker,
                s=26,
                zorder=3,
                label=label,
            )
            handles.setdefault(label, handle)
        axis.axvline(0.0, color=GRID, linewidth=1.4)
        axis.set_yticks(range(len(tasks)))
        axis.set_yticklabels(
            [
                task.replace("fig7_", "")
                .replace("fig8_logistic_", "n=")
                .replace("fig9_", "")
                for task in tasks
            ]
        )
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("quantum neuron better →")
    finish(
        figure,
        list(handles.values()),
        list(handles),
        "Phase 5: tuned quantum neuron vs tuned baselines (median paired advantage, min–max over seeds)",
        "Locally executed · learning rate tuned per model on a 20 % selection split (seeds 0–2) · 10 seeds at the selected rate · test = 500 held-out states · MerLin panel is an adaptation (simulation), not the paper's data",
        columns=2,
    )
    save(figure, "p5_paired_advantage", args.sweep_dir)


if __name__ == "__main__":
    main()
