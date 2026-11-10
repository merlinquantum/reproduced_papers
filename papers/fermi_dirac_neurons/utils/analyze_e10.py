"""E10 analysis: does the Fermi-Dirac neuron's gradient variance decay exponentially in n?

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/analyze_e10.py \
        papers/fermi_dirac_neurons/outdir/sweeps/e10_gradient_variance_v1

Applies the rule declared in the sweep spec before launch. For every cell
(term set, init range a, temperature rule, input ensemble, parameter
position, quantity d_j f or d_j L) with variances at n = 4, 5, ...:

* fit ``log10 Var = c - b n`` (exponential) and ``log10 Var = c - b log10 n``
  (polynomial) by least squares; both have two parameters, so
  ``AIC_poly - AIC_exp = m ln(RSS_poly / RSS_exp)`` over the m points;
* verdict ``exponential`` when that difference exceeds 2 and b > 0,
  ``polynomial`` when it is below -2, ``undecided`` otherwise; cells whose
  variance does not decrease (b <= 0 in the exponential fit) are ``no_decay``;
* report the per-qubit factor ``10**(-b)`` of the exponential fit and the
  local slope ``d log10 Var / dn`` over the last three n.

Only runs listed as COMPLETED are read, so a PARTIAL sweep is analysed over
the n it covers (the coverage is written into the output). Writes
``results/e10_gradient_variance_summary.json`` and the figure
``e10_gradient_variance_vs_n.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fermi_dirac_neurons.lib.plotting import TEXT_SECONDARY, apply_style, finish, save  # noqa: E402

MIN_QUBITS_IN_FIT = 4
AIC_THRESHOLD = 2.0
ZERO_VARIANCE = 1e-280
# Categorical slots 1-5 of the reference palette, in fixed order per term set.
TERM_SET_COLOURS = {
    "tfim": "#2a78d6",
    "heisenberg": "#eb6834",
    "ising": "#1baf7a",
    "fcim": "#eda100",
    "tfim_global": "#e87ba4",
}
INPUT_TITLES = {
    "zero": "|0…0⟩",
    "product": "random product states",
    "ghz": "GHZ",
    "haar": "Haar-random pure states",
    "ginibre": "random mixed (Ginibre)",
}


def _position(index: int, n_terms: int) -> str:
    if index == 0:
        return "first"
    if index == n_terms - 1:
        return "last"
    return "middle"


def load_cells(sweep_dir: Path) -> tuple[dict, list, dict]:
    """Return ``{cell: {n: record}}``, the completed run ids and the Haar checks."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    cells: dict = defaultdict(dict)
    completed, haar_checks = [], {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        completed.append(run["run_id"])
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        n_terms = len(metrics["term_labels"])
        for record in metrics["records"]:
            key = (
                record["term_set"],
                record["init_range"],
                record["temperature_rule"],
                record["input"],
                _position(record["parameter_index"], n_terms),
                record["quantity"],
            )
            cells[key][record["n_qubits"]] = record
        check = metrics["haar_data_variance_check"]
        if check is not None:
            haar_checks[(check["term_set"], check["n_qubits"])] = check
    return cells, completed, haar_checks


def classify(points: dict) -> dict:
    """Apply the declared exponential-vs-polynomial rule to one cell."""
    qubits = np.array(sorted(n for n in points if n >= MIN_QUBITS_IN_FIT))
    variances = np.array([points[n]["variance"] for n in qubits])
    result = {"qubits": qubits.tolist(), "variances": variances.tolist()}
    if len(qubits) < 3:
        return {**result, "verdict": "too_few_points"}
    if np.any(variances <= ZERO_VARIANCE):
        return {**result, "verdict": "zero_variance"}
    log_variance = np.log10(variances)
    exp_fit = np.polyfit(qubits, log_variance, 1)
    poly_fit = np.polyfit(np.log10(qubits), log_variance, 1)
    rss_exp = float(np.sum((np.polyval(exp_fit, qubits) - log_variance) ** 2))
    rss_poly = float(
        np.sum((np.polyval(poly_fit, np.log10(qubits)) - log_variance) ** 2)
    )
    floor = 1e-30
    delta_aic = len(qubits) * math.log((rss_poly + floor) / (rss_exp + floor))
    decay_rate = -float(exp_fit[0])
    tail = np.polyfit(qubits[-3:], log_variance[-3:], 1)
    if decay_rate <= 0:
        verdict = "no_decay"
    elif delta_aic > AIC_THRESHOLD:
        verdict = "exponential"
    elif delta_aic < -AIC_THRESHOLD:
        verdict = "polynomial"
    else:
        verdict = "undecided"
    return {
        **result,
        "verdict": verdict,
        "delta_aic_poly_minus_exp": delta_aic,
        "factor_per_qubit": 10 ** (-decay_rate),
        "polynomial_exponent": -float(poly_fit[0]),
        "local_slope_last3_log10_per_qubit": float(tail[0]),
    }


def plot(cells: dict, haar_checks: dict, sweep_dir: Path) -> None:
    """Var(d_j f) vs n at a = 0.5, T = 2, first parameter: one panel per input."""
    apply_style()
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for axis, input_kind in zip(axes.flat, INPUT_TITLES):
        for term_set, colour in TERM_SET_COLOURS.items():
            points = cells.get(
                (term_set, 0.5, "fixed_2", input_kind, "first", "output"), {}
            )
            if not points:
                continue
            qubits = sorted(points)
            variances = [points[n]["variance"] for n in qubits]
            lows = [points[n]["variance_ci95"][0] for n in qubits]
            highs = [points[n]["variance_ci95"][1] for n in qubits]
            axis.fill_between(
                qubits, lows, highs, color=colour, alpha=0.15, linewidth=0
            )
            axis.plot(
                qubits,
                variances,
                color=colour,
                marker="o",
                markersize=3,
                label=term_set,
            )
        reference_n = np.arange(2, 12)
        axis.plot(
            reference_n,
            0.5 * 2.0 ** (-reference_n.astype(float)),
            color=TEXT_SECONDARY,
            linestyle=":",
            linewidth=1.0,
            label="∝ 2⁻ⁿ (reference)",
        )
        axis.set_yscale("log")
        axis.set_title(INPUT_TITLES[input_kind])
        axis.set_xlabel("qubits n")
        axis.set_ylabel("Var over θ of ∂f/∂θ₁")
    axis = axes.flat[5]
    for term_set, colour in TERM_SET_COLOURS.items():
        qubits = sorted(n for (name, n) in haar_checks if name == term_set)
        if not qubits:
            continue
        axis.plot(
            qubits,
            [haar_checks[(term_set, n)]["empirical_variance"] for n in qubits],
            color=colour,
            marker="o",
            markersize=3,
            linestyle="none",
        )
        axis.plot(
            qubits,
            [haar_checks[(term_set, n)]["predicted_variance"] for n in qubits],
            color=colour,
            linewidth=1.0,
        )
    axis.set_yscale("log")
    axis.set_title("Haar data variance at fixed θ: formula (line) vs sampled (dots)")
    axis.set_xlabel("qubits n")
    axis.set_ylabel("Var over states of ∂f/∂θ₁")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    finish(
        figure,
        handles,
        labels,
        "E10: gradient variance of the Fermi–Dirac neuron vs qubit count (θ ~ U[−0.5, 0.5], T = 2, first coupling; band: 95 % bootstrap CI)",
        "extension (not in the paper); 200 θ draws per point (50 at n = 11), seed 0; 12-qubit cells not run (eigensolver failure, LOG.md)",
        columns=6,
    )
    for path in save(figure, "e10_gradient_variance_vs_n", sweep_dir):
        print(path)


def main() -> None:
    """Classify every cell and write the summary and figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    cells, completed, haar_checks = load_cells(args.sweep_dir)
    rows = []
    for key, points in sorted(cells.items(), key=lambda item: str(item[0])):
        term_set, init_range, rule, input_kind, position, quantity = key
        rows.append(
            {
                "term_set": term_set,
                "init_range": init_range,
                "temperature_rule": rule,
                "input": input_kind,
                "parameter_position": position,
                "quantity": quantity,
                **classify(points),
            }
        )
    summary = {
        "sweep_dir": str(args.sweep_dir),
        "completed_runs": len(completed),
        "rule": {"min_qubits": MIN_QUBITS_IN_FIT, "aic_threshold": AIC_THRESHOLD},
        "cells": rows,
        "haar_data_variance_checks": [
            {
                **check,
                "ratio_empirical_to_predicted": check["empirical_variance"]
                / check["predicted_variance"],
            }
            for _, check in sorted(haar_checks.items())
        ],
    }
    output = PROJECT_DIR / "results" / "e10_gradient_variance_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output)
    plot(cells, haar_checks, args.sweep_dir)


if __name__ == "__main__":
    main()
