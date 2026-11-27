"""Phase 5 comparison table from structured run artifacts (never from logs).

Reads the sweep status files of the gate-model reproduction, the photonic
adaptations and the classical baselines, checks run coverage, and applies the frozen
rule of LOG.md ("Phase 5 Comparison Plan"):

* metric: final-epoch test accuracy; aggregation: mean over the 5 feature
  distributions; variability: sample std;
* primary pairing: classical training seed 42 against the quantum or photonic run of
  the same distribution; secondary: classical mean over its training seeds;
* "no clear difference" when |mean difference| < 1.5 points or does not exceed the
  larger of the two standard deviations.

Example
-------
    python utils/compare_baselines.py --sweeps sweeps.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
PERMUTATION_SEEDS = [42, 43, 44, 45, 46]
LAYOUTS = {
    "E1": ([0, 1, 2, 3, 4, 5, 6, 7], "server_wide"),
    "E2": ([0], "server_narrow"),
    "E3": ([0] * 8, "server_wide"),
    "E4": ([0, 1, 2, 3, 0, 1, 2, 3], "server_wide"),
}
TIE_POINTS = 1.5
PAPER_REPORTED = {"E1": "over 94%", "E4": "about 90%"}


def read_runs(sweep_dir: Path) -> list[dict]:
    """Every run of a sweep with its structured metrics (completed runs only)."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    runs = []
    for run in status["runs"]:
        entry = {
            "status": run["status"],
            "seed": run["seed"],
            "candidate": run["candidate"],
        }
        if run["status"] == "COMPLETED":
            metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
            config = json.loads((sweep_dir / run["config_path"]).read_text())
            value = metrics["final_test_accuracy"]
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite accuracy in {sweep_dir / run['metrics_path']}"
                )
            entry.update(
                accuracy=value,
                seconds=metrics["wall_clock_seconds"],
                parameters=metrics["trainable_parameters"],
                slices=config["model"]["client_slices"],
                server=config["model"]["server_encoding"],
                architecture=config["model"].get("architecture"),
                size=config["model"].get("size"),
                permutation_seed=config["dataset"]["permutation_seed"],
            )
        runs.append(entry)
    return runs


def layout_of(run: dict) -> str:
    for name, (slices, server) in LAYOUTS.items():
        if run["slices"] == slices and run["server"] == server:
            return name
    raise ValueError(f"Unknown layout {run['slices']} {run['server']}")


def system_values(runs: list[dict], seed: int | None) -> dict[int, float]:
    """Accuracy per permutation seed; with ``seed=None`` the mean over training seeds."""
    by_permutation: dict[int, list[float]] = {}
    for run in runs:
        if seed is None or run["seed"] == seed:
            by_permutation.setdefault(run["permutation_seed"], []).append(
                run["accuracy"]
            )
    return {p: statistics.fmean(v) for p, v in by_permutation.items()}


def summarise(values: dict[int, float]) -> tuple[float, float]:
    data = [values[p] for p in PERMUTATION_SEEDS]
    return statistics.fmean(data), statistics.stdev(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sweeps",
        type=Path,
        required=True,
        help="JSON {system: {layout or 'all': dir}}",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args()
    spec = json.loads(args.sweeps.read_text())

    systems: dict[tuple[str, str], list[dict]] = {}
    coverage = []
    for system, entries in spec["systems"].items():
        for directory in entries.values():
            runs = read_runs(PROJECT_DIR / directory)
            done = [r for r in runs if r["status"] == "COMPLETED"]
            coverage.append(
                {
                    "system": system,
                    "sweep": directory,
                    "expected": len(runs),
                    "completed": len(done),
                }
            )
            for run in done:
                layout = (
                    "central" if run["architecture"] == "central" else layout_of(run)
                )
                label = (
                    f"{system}:{run['size']}"
                    if run["architecture"] == "central"
                    else system
                )
                systems.setdefault((label, layout), []).append(run)
    for (label, layout), runs in systems.items():
        keys = [(r["permutation_seed"], r["seed"]) for r in runs]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate runs for {label} {layout}")
        if {r["permutation_seed"] for r in runs} != set(PERMUTATION_SEEDS):
            raise ValueError(
                f"{label} {layout}: permutation seeds do not match the plan"
            )

    rows = []
    for (label, layout), runs in sorted(systems.items()):
        seeds = sorted({r["seed"] for r in runs})
        mean, std = summarise(system_values(runs, 42))
        all_mean, all_std = summarise(system_values(runs, None))
        rows.append(
            {
                "system": label,
                "layout": layout,
                "role": spec["roles"][label.split(":")[0]],
                "provenance": "local run",
                "scope": spec["scopes"][label.split(":")[0]],
                "runs": len(runs),
                "expected_runs": len(PERMUTATION_SEEDS) * len(seeds),
                "training_seeds": seeds,
                "mean_seed42": mean,
                "std_seed42": std,
                "mean_all_seeds": all_mean,
                "std_all_seeds": all_std,
                "parameters_total": runs[0]["parameters"]["total"],
                "classical_parameters": runs[0]["parameters"].get(
                    "server_classical_head",
                    runs[0]["parameters"]["total"]
                    if label.startswith("classical")
                    else 0,
                ),
                "median_wall_clock_seconds": statistics.median(
                    r["seconds"] for r in runs
                ),
                # Individual results per feature distribution (seed 42 and mean over seeds).
                "values_seed42": {
                    str(p): v for p, v in sorted(system_values(runs, 42).items())
                },
                "values_all_seeds": {
                    str(p): v for p, v in sorted(system_values(runs, None).items())
                },
            }
        )

    pairs = []
    for quantum, classical in spec["pairs"]:
        for layout in LAYOUTS:
            if (quantum, layout) not in systems or (classical, layout) not in systems:
                continue
            q = system_values(systems[(quantum, layout)], 42)
            c42 = system_values(systems[(classical, layout)], 42)
            call = system_values(systems[(classical, layout)], None)
            diffs = [q[p] - c42[p] for p in PERMUTATION_SEEDS]
            mean_diff, std_diff = statistics.fmean(diffs), statistics.stdev(diffs)
            q_mean, q_std = summarise(q)
            c_mean, c_std = summarise(c42)
            no_clear = abs(100 * mean_diff) < TIE_POINTS or abs(mean_diff) <= max(
                q_std, c_std
            )
            pairs.append(
                {
                    "pair": f"{quantum} - {classical}",
                    "layout": layout,
                    "mean_difference": mean_diff,
                    "std_difference": std_diff,
                    "paired_t_pvalue": float(stats.ttest_1samp(diffs, 0.0).pvalue),
                    "positive_in": sum(d > 0 for d in diffs),
                    "differences": diffs,
                    "difference_vs_classical_all_seeds": q_mean - summarise(call)[0],
                    "verdict": "no clear difference"
                    if no_clear
                    else (
                        "quantum/photonic higher"
                        if mean_diff > 0
                        else "classical higher"
                    ),
                }
            )

    result = {
        "coverage": coverage,
        "systems": rows,
        "pairs": pairs,
        "paper_reported": PAPER_REPORTED,
        "rule": {"tie_points": TIE_POINTS, "primary_pairing": "classical seed 42"},
    }
    args.out_dir.mkdir(exist_ok=True)
    (args.out_dir / "phase5_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    lines = [
        "| System | Layout | Role | Provenance | Scope | Runs / expected | Test acc, seed 42 (mean +- std over 5 distributions) | All training seeds | Parameters (classical) | Median training wall-clock per run (5 runs in parallel, Apple M5 CPU) |",
        "|---|---|---|---|---|---:|---:|---:|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['system']} | {r['layout']} | {r['role']} | {r['provenance']} | {r['scope']} | {r['runs']} / {r['expected_runs']} | "
            f"{100 * r['mean_seed42']:.2f} +- {100 * r['std_seed42']:.2f} | {100 * r['mean_all_seeds']:.2f} +- {100 * r['std_all_seeds']:.2f} | "
            f"{r['parameters_total']} ({r['classical_parameters']}) | {r['median_wall_clock_seconds']:.1f} s |"
        )
    lines += [
        "",
        "Paper-reported (not local, not ranked): E1 over 94%, E4 about 90%.",
        "",
        "| Pair (quantum/photonic - classical) | Layout | Mean difference (points) | std | paired t p | positive in n of 5 | vs classical 3-seed mean | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for p in pairs:
        lines.append(
            f"| {p['pair']} | {p['layout']} | {100 * p['mean_difference']:+.2f} | {100 * p['std_difference']:.2f} | "
            f"{p['paired_t_pvalue']:.4f} | {p['positive_in']} | {100 * p['difference_vs_classical_all_seeds']:+.2f} | {p['verdict']} |"
        )
    lines += ["", "| Sweep | System | Completed / expected |", "|---|---|---:|"]
    lines += [
        f"| {c['sweep']} | {c['system']} | {c['completed']} / {c['expected']} |"
        for c in coverage
    ]
    (args.out_dir / "phase5_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
