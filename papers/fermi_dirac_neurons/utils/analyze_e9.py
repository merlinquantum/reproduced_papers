"""E9 analysis: robustness of the quantum-vs-commuting ranking to eta, T and finite shots.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/analyze_e9.py \
        papers/fermi_dirac_neurons/outdir/sweeps/e9_robustness_v1 \
        papers/fermi_dirac_neurons/outdir/sweeps/paper_seeds_v1

(a) Every (task, eta, T) cell: paired quantum-minus-commuting advantage on the
declared metrics with the rule of ``utils/sweep_statistics.py``. (b) Finite
shots: final training loss per shot count, against the exact-gradient runs of
the same seeds in ``paper_seeds_v1`` (``fig7_tanh_2q_original``), and the
paired quantum-minus-commuting advantage at each shot count. Writes
``results/e9_robustness_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

from fermi_dirac_neurons.utils.sweep_statistics import (  # noqa: E402
    load_values,
    paired_comparison,
)

GRID_PATTERN = re.compile(r"e9_(fig[78])_(\w+?)_(\d)q_eta([\d.]+)_T([\d.]+)")
SHOTS_PATTERN = re.compile(r"e9b_fig7_tanh_2q_shots(\d+)")
MODELS = {
    "fig7": ("quantum_tfim", "classical_im", ["train_loss", "validation_mse"]),
    "fig8": ("quantum_heisenberg", "classical_fcim", ["validation_accuracy_linear"]),
}


def main() -> None:
    """Write the grid and finite-shot summaries."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("paper_sweep_dir", type=Path)
    args = parser.parse_args()
    values, seeds = load_values(args.sweep_dir)
    configs = sorted({config for config, _, _ in values})
    grid = []
    for config in configs:
        match = GRID_PATTERN.fullmatch(config)
        if match is None:
            continue
        figure, _, n_text, learning_rate, temperature = match.groups()
        quantum, commuting, metrics = MODELS[figure]
        for metric in metrics:
            grid.append(
                {
                    "config": config,
                    "task": f"{figure}_{n_text}q",
                    "learning_rate": float(learning_rate),
                    "temperature": float(temperature),
                    **paired_comparison(
                        values, seeds, (config, quantum), (config, commuting), metric
                    ),
                }
            )
    paper_values, paper_seeds = load_values(args.paper_sweep_dir)
    if paper_seeds != seeds:
        raise ValueError(f"Seed sets differ: {seeds} vs {paper_seeds}")
    shots = []
    reference = {
        model: [
            paper_values[("fig7_tanh_2q_original", model, "train_loss")][seed]
            for seed in seeds
        ]
        for model in ("quantum_tfim", "classical_im")
    }
    for config in configs:
        match = SHOTS_PATTERN.fullmatch(config)
        if match is None:
            continue
        row = {"config": config, "shots": int(match.group(1))}
        for model in ("quantum_tfim", "classical_im"):
            losses = np.array(
                [values[(config, model, "train_loss")][seed] for seed in seeds]
            )
            exact = np.array(reference[model])
            row[model] = {
                "train_loss_median": float(np.median(losses)),
                "exact_gradient_train_loss_median": float(np.median(exact)),
                "median_log10_ratio_to_exact": float(
                    np.median(np.log10(losses / exact))
                ),
            }
        row["advantage"] = paired_comparison(
            values,
            seeds,
            (config, "quantum_tfim"),
            (config, "classical_im"),
            "train_loss",
        )
        shots.append(row)
    shots.sort(key=lambda row: row["shots"])
    output = PROJECT_DIR / "results" / "e9_robustness_summary.json"
    output.write_text(
        json.dumps(
            {
                "sweep_dir": str(args.sweep_dir),
                "seeds": seeds,
                "grid": grid,
                "finite_shots": shots,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    if any(not math.isfinite(row["median_advantage"]) for row in grid):
        raise ValueError("Non-finite advantage in the grid")


if __name__ == "__main__":
    main()
