"""Summary of paired per-seed advantages from ``utils/sweep_statistics.py``.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/plot_paired_advantage.py \
        papers/fermi_dirac_neurons/results/paper_seeds_v1.json \
        papers/fermi_dirac_neurons/outdir/sweeps/paper_seeds_v1

Reads the statistics JSON (not the runs) and draws two charts, one per unit,
never on a shared axis: decades of training squared loss (Figs. 7 and 9,
original protocol) and validation-accuracy points (Fig. 8, linear rule).
Each row is one comparison: dot = median paired advantage of the first model
over the comparator (positive favours the first model), whisker = min-max
over seeds; filled dot = ``supported`` under the declared rule, hollow = tie
or not supported. Colour = the comparator's model role.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from fermi_dirac_neurons.lib.plotting import (  # noqa: E402
    COLOURS,
    TEXT_SECONDARY,
    apply_style,
    finish,
    model_role,
    save,
)

SHORT = {
    "quantum_tfim": "TFIM neuron",
    "quantum_heisenberg": "Heisenberg neuron",
    "quantum_tfim_tanh": "tanh",
    "quantum_heis_tanh": "tanh",
    "quantum_tfim_linear": "linear",
    "quantum_heis_linear": "linear",
    "classical_im": "IM",
    "classical_ising": "IM",
    "classical_fcim": "FCIM",
    "b2_pauli_features": "B2",
    "b3_quantum_linear": "B3 linear",
}


def _rows(
    comparisons: list[dict], figure_prefix: tuple[str, ...], metric: str
) -> list[dict]:
    rows = [
        comparison
        for comparison in comparisons
        if comparison["metric"] == metric
        and comparison["better_config"].startswith(figure_prefix)
        and comparison["better_config"].endswith("_original")
    ]
    return rows


def _draw(axis, rows: list[dict], unit: str) -> None:
    for position, row in enumerate(rows):
        colour = COLOURS[model_role(row["other_model"])]
        y = len(rows) - 1 - position
        axis.plot(
            [row["min_advantage"], row["max_advantage"]],
            [y, y],
            color=colour,
            linewidth=1.4,
        )
        supported = row["verdict"] == "supported"
        axis.plot(
            row["median_advantage"],
            y,
            marker="o",
            markersize=7,
            color=colour,
            markerfacecolor=colour if supported else "white",
            markeredgewidth=1.5,
        )
    labels = [
        f"{row['better_config'].removesuffix('_original').replace('_', ' ')}: {SHORT[row['better_model']]} vs {SHORT[row['other_model']]}"
        for row in rows
    ]
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(labels[::-1], fontsize=7)
    axis.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    axis.set_xlabel(f"paired advantage of the first model ({unit}); > 0 favours it")
    axis.grid(axis="y", visible=False)


def main() -> None:
    """Draw the two summary charts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("statistics_json", type=Path)
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    report = json.loads(args.statistics_json.read_text())
    apply_style()
    comparator_labels = {
        "commuting": "vs commuting neuron (IM / FCIM)",
        "b2": "vs B2 (paper η, untuned)",
        "linear": "vs quantum linear (B3 / Fig. 9 linear)",
    }
    verdict_legend = [
        Line2D(
            [],
            [],
            color=TEXT_SECONDARY,
            marker="o",
            linestyle="none",
            label="filled: supported (p < 0.05, beyond tolerance)",
        ),
        Line2D(
            [],
            [],
            color=TEXT_SECONDARY,
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            label="hollow: tie or not supported",
        ),
    ]
    scope = f"paper-accurate, seeds {report['seeds'][0]}–{report['seeds'][-1]}, two-sided Wilcoxon signed-rank (n = 10); dot: median, whisker: min–max over seeds"
    for name, prefix, metric, unit, title in (
        (
            "paired_advantage_loss",
            ("fig7", "fig9"),
            "train_loss",
            "decades of training loss",
            "Figs. 7 and 9: paired advantage in training squared loss",
        ),
        (
            "paired_advantage_accuracy",
            ("fig8",),
            "validation_accuracy_linear",
            "accuracy points",
            "Fig. 8: paired advantage in validation accuracy (linear rule)",
        ),
    ):
        rows = _rows(report["comparisons"], prefix, metric)
        figure, axis = plt.subplots(
            figsize=(9, 0.32 * len(rows) + 2.2), constrained_layout=True
        )
        _draw(axis, rows, unit)
        roles = [
            role
            for role in comparator_labels
            if any(model_role(row["other_model"]) == role for row in rows)
        ]
        legend = [
            Line2D(
                [], [], color=COLOURS[role], marker="o", label=comparator_labels[role]
            )
            for role in roles
        ] + verdict_legend
        finish(
            figure,
            legend,
            [line.get_label() for line in legend],
            title,
            scope,
            columns=2,
        )
        for path in save(figure, name, args.sweep_dir):
            print(path)


if __name__ == "__main__":
    main()
