"""Phase 5 fair comparison, generated from structured artifacts only.

``python utils/fair_comparison.py <sweeps_dir> <results_dir>`` checks that every sweep
listed in the frozen protocol (LOG.md, "Phase 5 Comparison Plan") has exactly its
expected number of COMPLETED runs with finite metrics, then writes
``fair_comparison.md``, ``fair_comparison.json`` and ``fair_comparison_params.png``.
A difference between two systems in one cell is meaningful only if it exceeds the
larger of their two standard deviations over seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_with_paper import (  # noqa: E402
    PAPER_ACCURACY,
    PAPER_E2,
    PAPER_KEY,
    SETTINGS,
    rounds_to_fraction,
    verdict,
)

TASKS = tuple(PAPER_ACCURACY)
EXPECTED_RUNS = {
    "e1_vqtn_hybrid": 75,
    "e2_vqtn_hybrid": 10,
    "control_classical_e1": 75,
    "control_classical_e2": 10,
    "control_iid_vqtn_0v9": 10,
    "sensitivity_faithful_0v9": 25,
    "e1_merlin": 75,
    "e2_merlin": 10,
    "sensitivity_merlin_lexgroup_0v9": 25,
}
# System -> (E1 sweep, role, provenance/scope label, matching basis)
SYSTEMS = {
    "Hybrid VQTN (reproduction, deviation: eight-<Z> head)": ("e1_vqtn_hybrid", "reproduced method", "local, 5 seeds, inferred circuit", "60 params; paper settings"),
    "Faithful VQTN (root-qubit read-out)": ("sensitivity_faithful_0v9", "reproduced method, sensitivity", "local, 5 seeds, 0v9 only", "42 params"),
    "Classical MLP 16-3-2 (fair baseline)": ("control_classical_e1", "fair classical baseline", "local, 5 seeds, untuned lr", "59 params, matched to hybrid VQTN"),
    "Photonic MerLin (adaptation)": ("e1_merlin", "MerLin adaptation", "local, 5 seeds, noiseless simulation", "130 params (not matched to classical)"),
    "Photonic MerLin, fixed LexGrouping (adaptation)": ("sensitivity_merlin_lexgroup_0v9", "MerLin adaptation, sensitivity", "local, 5 seeds, 0v9 only", "112 params"),
}  # fmt: skip


class ComparisonError(RuntimeError):
    """Raised when the reused evidence does not match the frozen protocol."""


def load_checked(sweeps: Path) -> dict[str, dict[str, list[dict]]]:
    """Load every protocol sweep; fail if counts, status or finiteness are off."""
    loaded: dict[str, dict[str, list[dict]]] = {}
    for sweep, expected in EXPECTED_RUNS.items():
        status = json.loads((sweeps / sweep / "sweep_status.json").read_text())
        completed = [run for run in status["runs"] if run["status"] == "COMPLETED"]
        if (
            status["status"] != "COMPLETED"
            or len(completed) != expected
            or len(status["runs"]) != expected
        ):
            raise ComparisonError(
                f"{sweep}: expected {expected} COMPLETED runs, found {len(completed)} of {len(status['runs'])}"
            )
        names = [run["run_id"] for run in completed]
        if len(set(names)) != len(names):
            raise ComparisonError(f"{sweep}: duplicate run ids")
        grouped: dict[str, list[dict]] = {}
        for run in completed:
            metrics = json.loads((sweeps / sweep / run["metrics_path"]).read_text())
            value = metrics["history"][-1]["test_accuracy"]
            if not math.isfinite(value):
                raise ComparisonError(f"{sweep}: non-finite metric in {run['run_id']}")
            grouped.setdefault(run["candidate_name"], []).append(metrics)
        loaded[sweep] = grouped
    return loaded


def cell(runs: list[dict]) -> tuple[float, float]:
    values = [100 * run["history"][-1]["test_accuracy"] for run in runs]
    return statistics.fmean(values), statistics.stdev(values)


def seconds(runs_by_cell: dict[str, list[dict]]) -> tuple[float, float, float]:
    values = [
        run["wall_clock_seconds"] for runs in runs_by_cell.values() for run in runs
    ]
    return statistics.fmean(values), min(values), max(values)


def pair_summary(
    cells_a: dict[str, tuple[float, float]], cells_b: dict[str, tuple[float, float]]
) -> dict:
    counts = {"above": 0, "below": 0, "within noise": 0}
    diffs = []
    for name in cells_a:
        (am, asd), (bm, bsd) = cells_a[name], cells_b[name]
        counts[verdict(am - bm, asd, bsd)] += 1
        diffs.append(am - bm)
    return {
        **counts,
        "cells": len(diffs),
        "mean_difference": statistics.fmean(diffs),
        "min_difference": min(diffs),
        "max_difference": max(diffs),
    }


def build(sweeps: Path) -> tuple[str, dict]:
    data = load_checked(sweeps)
    e1_cells = [f"task{t}_{s}" for t in TASKS for s in SETTINGS]
    zero_nine = [f"task0v9_{s}" for s in SETTINGS]
    per_system = {}
    system_rows = [
        "| System | Role | Provenance and scope | Successful / expected runs | Mean test accuracy % (0v9 / 2v7 / 3v6, mean over 5 settings) | Mean seed std (points) | Matching basis | Mean wall-clock per run s (min to max) | Status |",
        "|---|---|---|---:|---|---:|---|---|---|",
    ]
    paper_by_task = {
        t: statistics.fmean(PAPER_ACCURACY[t][PAPER_KEY[s]][0] for s in SETTINGS)
        for t in TASKS
    }
    system_rows.append(
        "| Paper QFL VQTN (Table II) | paper method | paper-reported, 5 runs, TFQ | n/a | "
        + " / ".join(f"{paper_by_task[t]:.2f}" for t in TASKS)
        + f" | {statistics.fmean(PAPER_ACCURACY[t][PAPER_KEY[s]][1] for t in TASKS for s in SETTINGS):.2f} | paper settings, parameter count not stated | n/a | reference only (not run locally) |"
    )
    for label, (sweep, role, scope, basis) in SYSTEMS.items():
        runs = data[sweep]
        cells = {name: cell(runs[name]) for name in runs}
        per_system[label] = cells
        tasks_present = [t for t in TASKS if f"task{t}_centralized" in cells]
        accuracy = " / ".join(
            f"{statistics.fmean(cells[f'task{t}_{s}'][0] for s in SETTINGS):.2f}"
            if t in tasks_present
            else "not run"
            for t in TASKS
        )
        mean_std = statistics.fmean(sd for _, sd in cells.values())
        mean_s, min_s, max_s = seconds(runs)
        parameters = runs[next(iter(runs))][0]["trainable_parameters"]
        if "Classical" in label:
            status = "fair baseline (reference)"
        elif "Hybrid" in label:
            status = "COMMENSURABLE with the classical baseline"
        else:
            status = "PARTIALLY_COMMENSURABLE (see pairs)"
        system_rows.append(
            f"| {label} | {role} | {scope} | {sum(len(v) for v in runs.values())} / {EXPECTED_RUNS[sweep]} | {accuracy} | {mean_std:.2f} | "
            f"{basis} ({parameters} counted) | {mean_s:.0f} ({min_s:.0f} to {max_s:.0f}) | {status} |"
        )

    hybrid, faithful, classical, photonic, lex = (per_system[k] for k in SYSTEMS)
    paper_cells = {
        f"task{t}_{s}": PAPER_ACCURACY[t][PAPER_KEY[s]] for t in TASKS for s in SETTINGS
    }

    def fed_vs_central(cells: dict, tasks: tuple) -> tuple[dict, dict]:
        fed = {
            f"task{t}_{s}": cells[f"task{t}_{s}"] for t in tasks for s in SETTINGS[1:]
        }
        central = {name: cells[f"{name.split('_')[0]}_centralized"] for name in fed}
        return fed, central

    pairs = []

    def add(label: str, status: str, a: dict, b: dict, note: str) -> None:
        summary = pair_summary(a, b)
        pairs.append((label, status, summary, note))

    add(
        "QFL hybrid VQTN vs its non-federated baseline (C1a, C1b)",
        "COMMENSURABLE",
        *fed_vs_central(hybrid, TASKS),
        "paper's own control, same model and budget",
    )
    add(
        "QFL faithful VQTN vs its non-federated baseline",
        "COMMENSURABLE",
        *fed_vs_central(faithful, ("0v9",)),
        "0v9 only",
    )
    add(
        "QFL photonic vs its non-federated baseline (adaptation)",
        "COMMENSURABLE",
        *fed_vs_central(photonic, TASKS),
        "MerLin adaptation",
    )
    add(
        "Federated classical MLP vs non-federated classical MLP",
        "COMMENSURABLE",
        *fed_vs_central(classical, TASKS),
        "context",
    )
    add(
        "Hybrid VQTN vs classical MLP",
        "COMMENSURABLE",
        {k: hybrid[k] for k in e1_cells},
        {k: classical[k] for k in e1_cells},
        "60 vs 59 params; classical lr untuned (handicap)",
    )
    add(
        "Faithful VQTN vs classical MLP",
        "PARTIALLY_COMMENSURABLE",
        {k: faithful[k] for k in zero_nine},
        {k: classical[k] for k in zero_nine},
        "42 vs 59 params, 0v9 only",
    )
    add(
        "Photonic MerLin vs classical MLP",
        "PARTIALLY_COMMENSURABLE",
        {k: photonic[k] for k in e1_cells},
        {k: classical[k] for k in e1_cells},
        "130 vs 59 params: only 'no photonic advantage' can be concluded",
    )
    add(
        "Photonic fixed LexGrouping vs classical MLP",
        "PARTIALLY_COMMENSURABLE",
        {k: lex[k] for k in zero_nine},
        {k: classical[k] for k in zero_nine},
        "112 vs 59 params, 0v9 only",
    )
    add(
        "Photonic MerLin vs hybrid VQTN",
        "PARTIALLY_COMMENSURABLE",
        {k: photonic[k] for k in e1_cells},
        {k: hybrid[k] for k in e1_cells},
        "130 vs 60 params, separate calibrations",
    )
    add(
        "Hybrid VQTN vs paper-reported values",
        "PARTIALLY_COMMENSURABLE",
        {k: hybrid[k] for k in e1_cells},
        paper_cells,
        "inferred circuit and partition; paper std from its 5 runs",
    )
    add(
        "Photonic MerLin vs paper-reported values",
        "PARTIALLY_COMMENSURABLE",
        {k: photonic[k] for k in e1_cells},
        paper_cells,
        "adaptation vs paper method",
    )

    pair_rows = [
        "| Pair (A vs B) | Status | Cells | A above | A below | Within noise | Mean A - B (points) | Range | Note |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for label, status, summary, note in pairs:
        pair_rows.append(
            f"| {label} | {status} | {summary['cells']} | {summary['above']} | {summary['below']} | {summary['within noise']} | "
            f"{summary['mean_difference']:+.2f} | {summary['min_difference']:+.2f} to {summary['max_difference']:+.2f} | {note} |"
        )

    rounds_rows = [
        "| System | Mean rounds to reach 99 % of final accuracy, federated cells (min to max over cells) |",
        "|---|---|",
    ]
    for label in (
        "Hybrid VQTN (reproduction, deviation: eight-<Z> head)",
        "Photonic MerLin (adaptation)",
        "Classical MLP 16-3-2 (fair baseline)",
    ):
        sweep = SYSTEMS[label][0]
        per_cell = [
            statistics.fmean(
                rounds_to_fraction(run) for run in data[sweep][f"task{t}_{s}"]
            )
            for t in TASKS
            for s in SETTINGS[1:]
        ]
        rounds_rows.append(
            f"| {label} | {statistics.fmean(per_cell):.1f} ({min(per_cell):.1f} to {max(per_cell):.1f}) |"
        )

    e2_rows = [
        "| System | Non-federated acc % | 4 clients acc % | Runs | Parameters | Mean wall-clock per run s | Status vs classical |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    e2_rows.append(
        f"| Paper QFL VQTN (Table IV) | not reported | {100 * PAPER_E2['accuracy']:.2f} (no std) | n/a | not stated | n/a | reference only |"
    )
    for label, sweep, status in (
        ("Hybrid VQTN", "e2_vqtn_hybrid", "COMMENSURABLE (78 vs 88 params, +13 %)"),
        ("Classical MLP 16-4-4", "control_classical_e2", "baseline"),
        (
            "Photonic MerLin (adaptation)",
            "e2_merlin",
            "PARTIALLY_COMMENSURABLE (148 vs 88)",
        ),
    ):
        runs = data[sweep]
        central, fed = (
            cell(runs["fourclass_centralized"]),
            cell(runs["fourclass_clients4"]),
        )
        e2_rows.append(
            f"| {label} | {central[0]:.2f} +- {central[1]:.2f} | {fed[0]:.2f} +- {fed[1]:.2f} | {sum(len(v) for v in runs.values())} / {EXPECTED_RUNS[sweep]} | "
            f"{runs['fourclass_clients4'][0]['trainable_parameters']} | {seconds(runs)[0]:.0f} | {status} |"
        )

    text = (
        "# Phase 5 fair comparison (generated by `utils/fair_comparison.py` from `metrics.json` files)\n\n"
        "All local values: final-round test accuracy %, mean over seeds 1 to 5, Apple M5 laptop CPU (3 runs in parallel, 3 threads each); wall-clock is simulation cost. The paper row is paper-reported, not run locally. MerLin rows are an adaptation, not the paper's model. A difference is counted as above or below only if it exceeds the larger seed standard deviation.\n\n"
        "## Systems (E1)\n\n"
        + "\n".join(system_rows)
        + "\n\n## Pairwise comparisons (E1)\n\n"
        + "\n".join(pair_rows)
        + "\n\n## Communication efficiency (claim C2)\n\n"
        + "\n".join(rounds_rows)
        + "\n\n## E2 (4-class)\n\n"
        + "\n".join(e2_rows)
        + "\n"
    )
    summary = {
        "pairs": {label: {"status": status, **s} for label, status, s, _ in pairs},
        "expected_runs": EXPECTED_RUNS,
    }
    return text, {
        "summary": summary,
        "per_system": {
            k: {c: list(v) for c, v in cells.items()} for k, cells in per_system.items()
        },
    }


def plot(result: dict, outputs: list[Path]) -> None:
    params = {"Hybrid VQTN (reproduction, deviation: eight-<Z> head)": 60, "Faithful VQTN (root-qubit read-out)": 42,
              "Classical MLP 16-3-2 (fair baseline)": 59, "Photonic MerLin (adaptation)": 130,
              "Photonic MerLin, fixed LexGrouping (adaptation)": 112}  # fmt: skip
    colours = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, setting in zip(axes, ("centralized", "clients16")):
        for (label, count), colour in zip(params.items(), colours):
            mean, std = result["per_system"][label][f"task0v9_{setting}"]
            axis.errorbar(
                [count], [mean], [std], fmt="o", color=colour, capsize=4, label=label
            )
        paper = PAPER_ACCURACY["0v9"][PAPER_KEY[setting]]
        axis.axhline(
            paper[0],
            color="black",
            linestyle="--",
            linewidth=1,
            label="paper-reported (parameter count not stated)",
        )
        axis.set_title(
            f"MNIST 0 vs 9, {'non-federated' if setting == 'centralized' else '16 clients'}"
        )
        axis.set_xlabel("trainable parameters")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy (%), 5 seeds")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5)
    figure.suptitle(
        "Phase 5: accuracy vs parameter count (local runs, noiseless simulation; MerLin = adaptation)",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    for path in outputs:
        path.mkdir(parents=True, exist_ok=True)
        figure.savefig(path / "fair_comparison_params.png", dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweeps_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args(argv)
    text, result = build(args.sweeps_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "fair_comparison.md").write_text(text, encoding="utf-8")
    (args.results_dir / "fair_comparison.json").write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8"
    )
    plot(result, [args.results_dir])
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
