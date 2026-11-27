"""Plot the Phase 5 comparison (group G1) from the generated comparison JSON.

Usage, from the paper directory::

    python utils/plot_phase5.py --comparison results/phase5_comparison.json --min-degree 5

Heatmap of seeds reaching the B1 floor (out of the expected runs) for every
system of group G1 with degree >= ``--min-degree`` under protocols P0-P3.
Each cell is annotated with the median final MSE. Rows are grouped by
modality, and photonic rows are marked as the MerLin adaptation.
Saves ``phase5_comparison.png`` next to the input JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROTOCOLS = ("P0", "P1", "P2", "P3")
MODALITY_ORDER = {"classical": 0, "qubit": 1, "photonic": 2}


def _label(row: dict) -> str:
    if row["modality"] == "classical":
        init = "zero init" if "zero" in row["role"] else "uniform init"
        return f"B2 classical, degree {row['degree']} ({init}), {row['n_parameters']} params"
    prefix = "MerLin adaptation: " if row["modality"] == "photonic" else ""
    return f"{prefix}{row['system']}, readout {row['readout']}, degree {row['degree']}, {row['n_parameters']} params"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--min-degree", type=int, required=True)
    args = parser.parse_args()
    rows = [
        r
        for r in json.loads(args.comparison.read_text(encoding="utf-8"))["rows"]
        if r["group"] == "G1" and r["degree"] >= args.min_degree
    ]
    systems = sorted(
        {(MODALITY_ORDER[r["modality"]], r["degree"], _label(r)) for r in rows}
    )
    labels = [label for _, _, label in systems]
    success = np.full((len(labels), len(PROTOCOLS)), np.nan)
    annotations = [["" for _ in PROTOCOLS] for _ in labels]
    for r in rows:
        i, j = labels.index(_label(r)), PROTOCOLS.index(r["protocol"])
        success[i, j] = r["success_count"] / r["expected_runs"]
        annotations[i][j] = (
            f"{r['success_count']}/{r['expected_runs']}\n{r['median_mse']:.0e}"
        )

    figure, axis = plt.subplots(figsize=(10, 0.42 * len(labels) + 1.8))
    image = axis.imshow(success, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for i in range(len(labels)):
        for j in range(len(PROTOCOLS)):
            if annotations[i][j]:
                axis.text(j, i, annotations[i][j], ha="center", va="center", fontsize=6)
    axis.set_xticks(range(len(PROTOCOLS)))
    axis.set_xticklabels(
        [
            "P0 paper\nbatch 25\nlr 0.3\n200 st.",
            "P1\nfull\nlr 0.3\n1000 st.",
            "P2\nfull\nlr 0.05\n1000 st.",
            "P3\nfull\nlr 0.01\n1000 st.",
        ],
        fontsize=6,
    )
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=6)
    figure.colorbar(
        image,
        ax=axis,
        fraction=0.03,
        label="fraction of seeds reaching B1 (MSE − B1 ≤ 1e-4)",
    )
    axis.set_title(
        "Fig. 4 published target: seeds reaching the B1 floor (cell: count, median final MSE). "
        "Compare within a column only.",
        fontsize=8,
    )
    figure.text(
        0.01,
        0.005,
        "[6 seeds per cell; local CPU simulation; B2 = frequency-matched classical baseline; "
        "MerLin rows are an adaptation; source: results/phase5_comparison.json]",
        fontsize=6,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.02, 1, 1))
    output = args.comparison.with_name("phase5_comparison.png")
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
