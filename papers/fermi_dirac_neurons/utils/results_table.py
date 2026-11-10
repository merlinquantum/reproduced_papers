"""Print the README results table (paper value vs reproduced value) as Markdown.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/results_table.py

Reads ``results/paper_seeds_v1_paper_placement.csv`` (paper values against the
10-seed reproduction, original protocol) and ``results/p5_comparison.json``
(claim outcomes against tuned fair baselines). Losses are compared in decades
(log10 reproduced median − log10 paper value), accuracies in points. "Metric
agreement" is "within seed range" when the paper value lies inside the
reproduced min-max over 10 seeds; "trend agreement" compares the ordering of
the quantum neuron and the paper's commuting baseline.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
QUANTUM_PREFIX = "quantum"
CLAIM_OF = {"fig7": "C4", "fig8": "C5", "fig9": "C6"}
PAPER_ITEM = {"fig7": "Fig. 7", "fig8": "Table II", "fig9": "Fig. 9"}


def _format(value: float, accuracy: bool) -> str:
    return f"{100 * value:.1f} %" if accuracy else f"{value:.2g}"


def main() -> None:
    """Print one row per (task, model) with a paper value."""
    placement = list(
        csv.DictReader(
            (PROJECT_DIR / "results" / "paper_seeds_v1_paper_placement.csv").open()
        )
    )
    outcomes = json.loads((PROJECT_DIR / "results" / "p5_comparison.json").read_text())[
        "claim_outcomes"
    ]
    rows = [
        row
        for row in placement
        if (row["config"].startswith("fig8") and row["paper_source"] == "Table II")
        or (not row["config"].startswith("fig8") and row["metric"] == "train_loss")
    ]
    medians = {(row["config"], row["model"]): float(row["our_median"]) for row in rows}
    print(
        "| Paper item | Task | Model | Claim tested | Paper value | Reproduced median [min, max] | Delta | Metric agreement | Trend agreement | Claim support (tuned B2, Phase 5) | Seeds | Comment |"
    )
    print("|---|---|---|---|---:|---:|---:|---|---|---|---:|---|")
    for row in rows:
        figure = row["config"].split("_")[0]
        accuracy = "accuracy" in row["metric"]
        paper, median = float(row["paper_value"]), float(row["our_median"])
        delta = (
            f"{100 * (median - paper):+.1f} pts"
            if accuracy
            else f"{math.log10(median) - math.log10(paper):+.2f} dec"
        )
        partners = [
            value
            for (config, model), value in medians.items()
            if config == row["config"]
            and model != row["model"]
            and not model.endswith("_linear")
        ]
        quantum = row["model"].startswith(QUANTUM_PREFIX)
        if partners:
            other = partners[0]
            better = median > other if accuracy else median < other
            trend = "yes" if better == quantum else "no"
        else:
            trend = "n/a"
        if row["model"].endswith("_linear"):
            trend = "tanh < linear: yes" if figure == "fig9" else trend
        claim = CLAIM_OF[figure]
        support = outcomes[claim]["outcome"]
        comment = "paper value from " + row["paper_source"]
        if figure == "fig9":
            comment += "; settings inferred (D9)"
        print(
            f"| {PAPER_ITEM[figure]} | {row['config'].removesuffix('_original')} | {row['model']} | {claim} | "
            f"{_format(paper, accuracy)} | {_format(median, accuracy)} [{_format(float(row['our_min']), accuracy)}, {_format(float(row['our_max']), accuracy)}] | "
            f"{delta} | {'within seed range' if row['inside_our_range'] == 'True' else 'outside seed range'} | {trend} | {support} | 10 | {comment} |"
        )


if __name__ == "__main__":
    main()
