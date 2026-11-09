"""Plot the Phase 4 MerLin adaptation against the Phase 3 qubit models.

Usage, from the paper directory::

    python utils/plot_merlin.py --photonic-run outdir/<e2_fig4_merlin run> \
        --qubit-run outdir/<e2_fig4_scaling1 run> --reach-run outdir/<e7 run>

``merlin_mse_vs_degree.png``: final full-grid MSE per seed against the
model's predicted degree on the published Fig. 4 target, for the qubit
serial and parallel models (Phase 3) and the photonic photon-number and
re-uploading routes (Phase 4), with the B1 floor of each degree.
``merlin_coefficient_reach.png``: largest reachable ``|c_k|`` against the
target's ``|c_k|``. Both are saved to the photonic run directory and
``results/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _figures import load_completed_metrics, save_to_run_and_results, stamp_scope

ROUND_OFF_FLOOR = 1e-16
FAMILY_STYLE = {
    "qubit serial": ("tab:blue", "o"),
    "qubit parallel": ("tab:cyan", "s"),
    "photonic photon-number": ("tab:purple", "D"),
    "photonic re-uploading": ("tab:orange", "^"),
}


def _family_and_degree(model: dict) -> tuple[str, int]:
    if model["layout"] == "photonic":
        family = {
            "photon_number": "photonic photon-number",
            "reuploading": "photonic re-uploading",
        }
        return family[model["route"]], int(model["n_photons"]) * int(
            model["encoding_layers"]
        )
    return f"qubit {model['layout']}", int(model["repetitions"])


def plot_mse_against_degree(photonic_run: Path, qubit_run: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    floors: dict[int, float] = {}
    labelled = set()
    for run_dir in (qubit_run, photonic_run):
        metrics = load_completed_metrics(run_dir, "fit")
        # Only the Fig. 4 target is compared here; Fig. 3c cases are in the tables.
        fig4_cases = [c for c in metrics["cases"] if c["name"].startswith("fig4")]
        families = sorted({_family_and_degree(c["model"])[0] for c in fig4_cases})
        for case in fig4_cases:
            family, degree = _family_and_degree(case["model"])
            color, marker = FAMILY_STYLE[family]
            offset = (families.index(family) - (len(families) - 1) / 2) * 0.12
            offset += 0.06 if family.startswith("photonic") else -0.06
            values = [
                max(run["final_mse"], ROUND_OFF_FLOOR)
                for run in metrics["runs"]
                if run["case"] == case["name"]
            ]
            axis.scatter(
                np.full(len(values), degree + offset),
                values,
                color=color,
                marker=marker,
                s=22,
                alpha=0.75,
                label=family if family not in labelled else None,
            )
            labelled.add(family)
            floors[degree] = max(case["b1_mse"], ROUND_OFF_FLOOR)
    degrees = sorted(floors)
    axis.plot(
        degrees,
        [floors[d] for d in degrees],
        "k--",
        lw=1,
        marker="_",
        label="B1 floor (free coefficients)",
    )
    axis.axhline(1e-3, color="0.6", lw=0.8, ls=":", label="near-perfect threshold 1e-3")
    axis.set_yscale("log")
    axis.set_ylim(ROUND_OFF_FLOOR / 10, 1)
    axis.set_xticks(degrees)
    axis.set_xlabel("predicted degree of the model (r for qubits, n·L for photonic)")
    axis.set_ylabel("final full-grid MSE (per seed)")
    axis.set_title(
        "Fig. 4 target (published scaling): qubit models vs MerLin adaptation, 200 steps",
        fontsize=9,
    )
    axis.legend(fontsize=7, loc="lower left")
    stamp_scope(
        figure,
        "paper protocol, 6 seeds; photonic = MerLin adaptation (FOCK, threshold readout), simulation",
        f"{qubit_run}, {photonic_run}",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save_to_run_and_results(figure, photonic_run, "merlin_mse_vs_degree.png")


def plot_coefficient_reach(reach_run: Path, photonic_run: Path) -> None:
    metrics = load_completed_metrics(reach_run, "coefficient_reach")
    cases = metrics["cases"]
    positions = np.arange(len(cases))
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(
        positions - 0.18,
        [c["best_abs_coefficient"] for c in cases],
        width=0.36,
        color="tab:purple",
        label="largest |c_k| found",
    )
    axis.bar(
        positions + 0.18,
        [c["target_abs_coefficient"] for c in cases],
        width=0.36,
        color="0.6",
        label="target |c_k|",
    )
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{c['name']}\n(degree {c['model_degree']}, k = {c['frequency']})"
            for c in cases
        ],
        fontsize=7,
    )
    axis.set_ylabel("|c_k|")
    axis.set_title(
        "Reachable top coefficient vs target (necessary condition for fitting)",
        fontsize=9,
    )
    axis.legend(fontsize=7)
    stamp_scope(
        figure,
        f"diagnostic: Adam lr {metrics['lr']:g}, {metrics['steps']} steps, {len(metrics['init_seeds'])} inits; lower bound",
        str(reach_run),
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_to_run_and_results(figure, photonic_run, "merlin_coefficient_reach.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photonic-run", required=True, type=Path)
    parser.add_argument("--qubit-run", required=True, type=Path)
    parser.add_argument("--reach-run", required=True, type=Path)
    args = parser.parse_args()
    plot_mse_against_degree(args.photonic_run, args.qubit_run)
    plot_coefficient_reach(args.reach_run, args.photonic_run)


if __name__ == "__main__":
    main()
