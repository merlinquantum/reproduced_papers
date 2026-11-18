"""Export every sweep's complete candidate table into ``results/sweeps/`` (durable copies).

``python utils/export_sweep_tables.py <sweeps_dir> <results_dir>`` reads each sweep's
plan, ``sweep_status.json`` and the runs' ``metrics.json`` and writes
``<results_dir>/sweeps/<sweep_id>.md`` with the declared selection rule, coverage and
the full per-candidate table (mean and standard deviation over seeds of the declared
metric at the last round). Raw runs stay in ``outdir/``, which Git ignores.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PLANS_DIR = Path(__file__).resolve().parent / "sweep_plans"


def export(sweep_dir: Path, destination: Path) -> str:
    plan = json.loads((PLANS_DIR / f"{sweep_dir.name}.json").read_text())
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    rule = plan["selection"]
    metric = rule["metric"]
    grouped: dict[str, list[tuple[int, dict]]] = {}
    for run in status["runs"]:
        if run["status"] == "COMPLETED":
            metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
            grouped.setdefault(run["candidate_name"], []).append((run["seed"], metrics))
    expected_per_candidate = len(plan["seeds"])
    rows = [
        f"| Candidate | Parameters | Successful / expected runs | Mean {metric} (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    aggregates = {}
    for name, runs in grouped.items():
        runs.sort(key=lambda item: item[0])
        values = [100 * metrics["history"][-1][metric] for _, metrics in runs]
        tests = [100 * metrics["history"][-1]["test_accuracy"] for _, metrics in runs]
        mean = statistics.fmean(values)
        aggregates[name] = mean
        spread = statistics.stdev(values) if len(values) > 1 else float("nan")
        rows.append(
            f"| `{name}` | {runs[0][1]['trainable_parameters']} | {len(values)} / {expected_per_candidate} | "
            f"{mean:.2f} | {spread:.2f} | {', '.join(f'{v:.2f}' for v in values)} | {statistics.fmean(tests):.2f} |"
        )
    completed = sum(len(runs) for runs in grouped.values())
    coverage = (
        "COMPLETE SWEEP"
        if completed == status["expected_runs"] and status["status"] == "COMPLETED"
        else "PARTIAL SWEEP"
    )
    is_selection = rule["metric"] != "test_accuracy"
    ordered = sorted(
        aggregates, key=aggregates.get, reverse=rule["direction"] == "maximize"
    )
    if is_selection and ordered and "diagnosis only" in rule.get("tie_handling", ""):
        outcome = f"Diagnosis sweep: no candidate is selected; highest observed mean: `{ordered[0]}`."
    elif is_selection and ordered:
        best = aggregates[ordered[0]]
        tied = [
            name
            for name in ordered
            if abs(aggregates[name] - best) <= 100 * rule["tie_tolerance"]
        ]
        outcome = f"Highest mean under the declared rule: {', '.join(f'`{n}`' for n in tied)} ({'tie' if len(tied) > 1 else 'no tie'} within {rule['tie_tolerance']})."
    else:
        outcome = "Reporting sweep: no candidate is selected (configuration fixed in advance)."
    text = (
        f"# Sweep `{plan['sweep_id']}`\n\n{plan['purpose']}\n\n"
        f"- Declared rule: metric `{metric}`, direction {rule['direction']}, split {rule['split']}, checkpoint {rule['checkpoint']}, aggregation {rule['aggregation']} over seeds {plan['seeds']}, tie tolerance {rule['tie_tolerance']}; tie handling: {rule.get('tie_handling', 'not declared')}.\n"
        f"- Coverage: **{coverage}**, {completed} of {status['expected_runs']} planned runs completed ({status['started_at']} to {status['completed_at']}).\n"
        f"- {outcome}\n\n" + "\n".join(rows) + "\n"
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{plan['sweep_id']}.md").write_text(text, encoding="utf-8")
    return plan["sweep_id"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweeps_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args(argv)
    for sweep_dir in sorted(
        path
        for path in args.sweeps_dir.iterdir()
        if (path / "sweep_status.json").is_file()
    ):
        print("exported", export(sweep_dir, args.results_dir / "sweeps"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
