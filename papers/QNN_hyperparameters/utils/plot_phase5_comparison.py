"""Plot the Phase 5 system comparison from ``phase5_systems.csv``.

Usage (from the repository root)::

    python papers/QNN_hyperparameters/utils/plot_phase5_comparison.py \
        papers/QNN_hyperparameters/results/phase5_systems.csv

Writes ``phase5_comparison.png`` next to the input table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SHORT_NAMES = {
    "qubit QNN (E-B)": "qubit QNN\n(E-B)",
    "classical (E-F)": "classical\n(E-F)",
    "photonic MerLin (E-P, adaptation)": "photonic MerLin\n(E-P, adaptation)",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("systems_csv", type=Path)
    parser.add_argument("--scaling", default="minmax")
    args = parser.parse_args()

    table = pd.read_csv(args.systems_csv)
    table = table[table["scaling"] == args.scaling]
    datasets = list(dict.fromkeys(table["data"]))
    figure, axes = plt.subplots(1, len(datasets), figsize=(9.5, 4.2), sharey=False)
    figure.patch.set_facecolor(SURFACE)
    for axis, data in zip(axes, datasets):
        rows = table[table["data"] == data].reset_index(drop=True)
        axis.set_facecolor(SURFACE)
        for position, (row, color) in enumerate(zip(rows.itertuples(), SERIES_COLORS)):
            axis.errorbar(
                position,
                row.selected_test_mean,
                yerr=row.selected_test_sd,
                fmt="o",
                markersize=8,
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=4,
                zorder=3,
                label="selected configuration (mean ± sd over seeds)"
                if position == 0
                else None,
            )
            axis.plot(
                position + 0.18,
                row.grid_test_mean,
                marker="D",
                markersize=6,
                markerfacecolor=SURFACE,
                markeredgecolor=color,
                markeredgewidth=2,
                linestyle="none",
                zorder=3,
                label="mean over the whole grid" if position == 0 else None,
            )
            axis.annotate(
                f"{row.selected_test_mean:.3f}",
                (position, row.selected_test_mean),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=8,
                color=TEXT_PRIMARY,
            )
        axis.set_xticks(range(len(rows)))
        axis.set_xticklabels([SHORT_NAMES[name] for name in rows["system"]], fontsize=8)
        axis.set_xlim(-0.6, len(rows) - 0.4)
        test_rows = rows["preprocessing"].iloc[0]
        axis.set_title(
            f"{data.capitalize()} ({test_rows})", color=TEXT_PRIMARY, fontsize=11
        )
        axis.grid(axis="y", color="#e6e5e0", lw=0.6)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#d4d3cd")
    axes[0].set_ylabel("test accuracy", color=TEXT_SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        fontsize=8,
        frameon=False,
        labelcolor=TEXT_PRIMARY,
    )
    figure.suptitle(
        "Test accuracy by system, configuration selected on validation (MinMax inputs)",
        color=TEXT_PRIMARY,
        fontsize=11,
    )
    figure.text(
        0.5,
        0.005,
        "Locally executed, not paper-reported. Same rows, splits and features per dataset. "
        "Not capacity-matched: QNN 20-56, classical 8-570, photonic 246-576 parameters. "
        "Photonic is a simulated adaptation. QNN and photonic 3 seeds, classical 10.",
        ha="center",
        va="bottom",
        fontsize=7,
        color=TEXT_SECONDARY,
        wrap=True,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.88))
    output = args.systems_csv.parent / "phase5_comparison.png"
    figure.savefig(output, dpi=160, facecolor=SURFACE)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
