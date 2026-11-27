"""E12 analysis: Kirby-Love invariance counts and graded contextuality vs temperature.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/analyze_e12.py \
        papers/fermi_dirac_neurons/outdir/sweeps/e12_contextuality_v1

Reads the COMPLETED sweep's ``metrics.json`` files and writes
``results/e12_contextuality_summary.json`` plus the figure
``e12_contextuality_vs_temperature.png`` (sweep ``figures/`` and
``results/figures/``):

* invariance: per (n, set kind, activation), how many sets had a support whose
  Kirby-Love flag differed from the term set's flag at some T, and the
  temperatures of every change;
* graded: per (family, n, epsilon, T), the median over draws of the
  epsilon-support size, the number of noncontextual draws and the median
  greedy contextual fraction (an upper bound; None where the support exceeded
  the configured cap).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from fermi_dirac_neurons.lib.plotting import apply_style, finish, save  # noqa: E402

# Sequential single-hue ramp (blue steps 250, 400, 550, 700 of the reference
# palette): n is ordinal, so lighter = fewer qubits.
QUBIT_COLOURS = {2: "#86b6ef", 3: "#3987e5", 4: "#1c5cab", 5: "#0d366b"}
FIGURE_EPSILON = 1e-3


def load(sweep_dir: Path) -> dict:
    """Return ``{config: metrics}`` for a COMPLETED sweep."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"Sweep is {status['status']}, not COMPLETED")
    return {
        run["config"]: json.loads((sweep_dir / run["metrics_path"]).read_text())
        for run in status["runs"]
    }


def invariance_summary(metrics: dict) -> list[dict]:
    """Count flag changes per (n, set kind, activation)."""
    temperatures = metrics["temperatures"]
    groups: dict = defaultdict(lambda: {"sets": 0, "changed": 0, "examples": []})
    for record in metrics["invariance"]:
        key = (record["n_qubits"], record["set_kind"], record["activation"])
        groups[key]["sets"] += 1
        if record["flag_changed_at_some_T"]:
            groups[key]["changed"] += 1
            groups[key]["examples"].append(
                {
                    "labels": record["labels"],
                    "support_sizes": record["support_sizes"],
                    "temperatures": temperatures,
                }
            )
    return [
        {"n_qubits": n, "set_kind": kind, "activation": activation, **value}
        for (n, kind, activation), value in sorted(groups.items())
    ]


def graded_summary(records: list[dict]) -> list[dict]:
    """Median support size, noncontextual count and contextual fraction per cell."""
    groups: dict = defaultdict(list)
    for record in records:
        groups[
            (
                record["family"],
                record["n_qubits"],
                record["epsilon"],
                record["temperature"],
            )
        ].append(record)
    rows = []
    for (family, n_qubits, epsilon, temperature), cell in sorted(groups.items()):
        fractions = [
            record["contextual_fraction"]
            for record in cell
            if record["contextual_fraction"] is not None
        ]
        rows.append(
            {
                "family": family,
                "n_qubits": n_qubits,
                "epsilon": epsilon,
                "temperature": temperature,
                "draws": len(cell),
                "median_support_size": float(
                    np.median([record["support_size"] for record in cell])
                ),
                "noncontextual_draws": int(
                    sum(record["noncontextual"] for record in cell)
                ),
                "median_contextual_fraction": float(np.median(fractions))
                if len(fractions) == len(cell)
                else None,
            }
        )
    return rows


def plot(rows: list[dict], sweep_dir: Path) -> None:
    """Contextual fraction and support size vs T, one column per family."""
    apply_style()
    families = sorted({row["family"] for row in rows})
    figure, axes = plt.subplots(
        2, len(families), figsize=(10, 7), constrained_layout=True, sharex=True
    )
    for column, family in enumerate(families):
        for n_qubits, colour in QUBIT_COLOURS.items():
            cells = [
                row
                for row in rows
                if row["family"] == family
                and row["n_qubits"] == n_qubits
                and row["epsilon"] == FIGURE_EPSILON
            ]
            temperatures = [row["temperature"] for row in cells]
            fractions = [
                np.nan
                if row["median_contextual_fraction"] is None
                else row["median_contextual_fraction"]
                for row in cells
            ]
            axes[0, column].plot(
                temperatures, fractions, color=colour, marker="o", markersize=3
            )
            axes[1, column].plot(
                temperatures,
                [row["median_support_size"] for row in cells],
                color=colour,
                marker="o",
                markersize=3,
            )
        axes[0, column].set_title(f"{'TFIM' if family == 'tfim' else 'Heisenberg'}")
        axes[0, column].set_ylabel("greedy contextual fraction (upper bound)")
        axes[1, column].set_ylabel(f"ε-support size (ε = {FIGURE_EPSILON:g})")
        axes[1, column].set_yscale("log")
        for row_index in range(2):
            axes[row_index, column].set_xscale("log")
        axes[1, column].set_xlabel("temperature T")
    handles = [
        Line2D([], [], color=colour, marker="o", markersize=3, label=f"n = {n}")
        for n, colour in QUBIT_COLOURS.items()
    ]
    finish(
        figure,
        handles,
        [handle.get_label() for handle in handles],
        "E12: contextuality of the Pauli support of tanh(H/T) vs temperature (median over 20 draws, θ ~ U[−2, 2])",
        "extension (not in the paper); synthetic random Hamiltonians, seed 0; missing points: support above 300 strings, greedy fraction not computed",
        columns=4,
    )
    for path in save(figure, "e12_contextuality_vs_temperature", sweep_dir):
        print(path)


def main() -> None:
    """Write the summary JSON and the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    metrics = load(args.sweep_dir)
    graded = [
        record
        for config, run in sorted(metrics.items())
        if run["graded"]
        for record in run["graded"]
    ]
    summary = {
        "sweep_dir": str(args.sweep_dir),
        "invariance": invariance_summary(metrics["e12_invariance"]),
        "graded": graded_summary(graded),
    }
    output = PROJECT_DIR / "results" / "e12_contextuality_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output)
    plot(summary["graded"], args.sweep_dir)


if __name__ == "__main__":
    main()
