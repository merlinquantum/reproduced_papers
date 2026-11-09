"""Read-only check: recompute every sweep from raw ``metrics.json`` and compare.

Usage (from the repository root)::

    python3 papers/contextuality_inductive_bias/utils/verify_from_raw_metrics.py \\
        --sweeps-root papers/contextuality_inductive_bias/outdir/sweeps \\
        --results papers/contextuality_inductive_bias/results

The checker writes nothing. It is deliberately independent of the tools it
checks: it does not import ``summarize_sweep``, ``select_hyperparameters`` or
``fair_comparison_table``. For every sweep it:

1. enumerates the planned runs in ``sweep_status.json`` and reads each
   COMPLETED run's ``metrics.json`` through its ``metrics_path``;
2. groups runs by grid point and recomputes the count, mean and sample std
   (ddof = 1) of every metric declared in the sweep plan, excluding runs with
   a non-finite metric (reported as INVALID);
3. compares them with the sweep's ``summary.json`` (counts exactly, values to
   a relative 1e-12).

For the three tuning sweeps together, it re-applies the selection rule of the
Phase 5 plan and its amendments:
- a candidate is eligible only if all tuning seeds 0-2 COMPLETED with a
  finite final validation NLL;
- the minimum mean final validation NLL wins;
- candidates within 1e-4 are tied, and ties go to the smallest lr, then the
  smallest L2.

It compares the result with ``results/fair_tuning_selection.json``, including
the tied sets. Finally, it checks every row of ``results/fair_comparison.json``
against the means recomputed from ``fair_final``.

Exit status 0 means every check passed. Any mismatch is printed, and the
exit status is 1.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

TUNING_SWEEPS = ("fair_tuning", "fair_tuning_ext", "fair_tuning_ext2")
TUNING_SEEDS = (0, 1, 2)
TIE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-12


def _lookup(metrics: dict, dotted: str):
    value = metrics
    for part in dotted.split("."):
        value = value[part]
    return float(value)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= RELATIVE_TOLERANCE * max(abs(a), abs(b), 1e-300)


def recompute_sweep(sweep_dir: Path) -> tuple[dict, list[dict]]:
    """Return {grid key: {metric: values, 'expected': n, 'invalid': [...]}} and the raw run records."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    metric_names = status["spec"]["plan"]["metrics"]
    groups: dict[str, dict] = defaultdict(
        lambda: {"expected": 0, "values": defaultdict(list), "invalid": []}
    )
    records = []
    run_ids = [run["run_id"] for run in status["runs"]]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError(f"{sweep_dir.name}: duplicate run IDs")
    for run in status["runs"]:
        key = json.dumps(run["grid_point"], sort_keys=True)
        group = groups[key]
        group["expected"] += 1
        record = {
            "grid_point": run["grid_point"],
            "seed": run["seed"],
            "status": run["status"],
        }
        if run["status"] == "COMPLETED":
            metrics = json.loads(
                (sweep_dir / run["metrics_path"]).read_text(encoding="utf-8")
            )
            values = {name: _lookup(metrics, name) for name in metric_names}
            record["metrics"] = metrics
            if all(math.isfinite(v) for v in values.values()):
                for name, value in values.items():
                    group["values"][name].append(value)
            else:
                group["invalid"].append(run["run_id"])
        records.append(record)
    return dict(groups), records


def check_summary(sweep_dir: Path, groups: dict) -> list[str]:
    summary = json.loads((sweep_dir / "summary.json").read_text(encoding="utf-8"))
    problems = []
    candidates = {
        json.dumps(c["grid_point"], sort_keys=True): c for c in summary["candidates"]
    }
    if set(candidates) != set(groups):
        problems.append(
            f"{sweep_dir.name}: candidate sets differ between sweep_status and summary"
        )
    for key, group in groups.items():
        candidate = candidates.get(key)
        if candidate is None:
            continue
        n_valid = len(next(iter(group["values"].values()), []))
        if (
            candidate["successful_runs"] != n_valid
            or candidate["expected_runs"] != group["expected"]
        ):
            problems.append(
                f"{sweep_dir.name} {key}: counts {n_valid}/{group['expected']} vs summary {candidate['successful_runs']}/{candidate['expected_runs']}"
            )
        for name, values in group["values"].items():
            aggregate = candidate["aggregates"][name]
            array = np.array(values)
            std = float(array.std(ddof=1)) if array.size > 1 else 0.0
            if (
                aggregate["n"] != array.size
                or not _close(aggregate["mean"], float(array.mean()))
                or not _close(aggregate["std"], std)
            ):
                problems.append(
                    f"{sweep_dir.name} {key} {name}: recomputed mean {array.mean()!r} std {std!r} vs summary {aggregate['mean']!r} {aggregate['std']!r}"
                )
    total_valid = sum(
        len(next(iter(g["values"].values()), [])) for g in groups.values()
    )
    if summary["successful_runs"] != total_valid:
        problems.append(
            f"{sweep_dir.name}: successful_runs {summary['successful_runs']} vs recomputed {total_valid}"
        )
    return problems


def reselect(tuning_records: list[dict]) -> dict:
    """Independent re-application of the Phase 5 selection rule."""
    candidates: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for record in tuning_records:
        model = json.dumps(
            {
                k: v
                for k, v in record["grid_point"].items()
                if k not in ("--lr", "--l2")
            },
            sort_keys=True,
        )
        hyper = (
            float(record["grid_point"]["--lr"]),
            float(record["grid_point"]["--l2"]),
        )
        candidates[model][hyper].append(record)
    selections = {}
    for model, by_hyper in candidates.items():
        eligible = {}
        for hyper, records in by_hyper.items():
            seeds = sorted(r["seed"] for r in records)
            nll = [
                r["metrics"]["final"]["validation_nll"]
                for r in records
                if r["status"] == "COMPLETED"
            ]
            if (
                seeds == list(TUNING_SEEDS)
                and len(nll) == len(TUNING_SEEDS)
                and all(math.isfinite(v) for v in nll)
            ):
                eligible[hyper] = float(np.mean(nll))
        best = min(eligible.values())
        tied = sorted(h for h, v in eligible.items() if v - best <= TIE_TOLERANCE)
        selections[model] = {
            "selected": tied[0],
            "tied": tied,
            "n_candidates": len(by_hyper),
            "n_eligible": len(eligible),
        }
    return selections


def check_selection(selections: dict, selection_path: Path) -> list[str]:
    stored = {
        json.dumps(item["model"], sort_keys=True): item
        for item in json.loads(selection_path.read_text(encoding="utf-8"))
    }
    problems = []
    if set(stored) != set(selections):
        problems.append(
            "tuning: model sets differ between raw runs and fair_tuning_selection.json"
        )
    for model, mine in selections.items():
        item = stored.get(model)
        if item is None:
            continue
        theirs = (item["selected"]["lr"], item["selected"]["l2"])
        tied = sorted((t["lr"], t["l2"]) for t in item["tied"])
        if theirs != mine["selected"] or tied != mine["tied"]:
            problems.append(
                f"tuning {model}: recomputed {mine['selected']} tied {mine['tied']} vs stored {theirs} tied {tied}"
            )
    return problems


def check_comparison(final_groups: dict, comparison_path: Path) -> list[str]:
    rows = json.loads(comparison_path.read_text(encoding="utf-8"))["rows"]
    recomputed = [
        (
            len(g["values"]["final.test_kl"]),
            g["expected"],
            float(np.mean(g["values"]["final.test_kl"])),
        )
        for g in final_groups.values()
    ]
    problems = []
    for row in rows:
        matches = [
            r
            for r in recomputed
            if r[0] == row["successful"]
            and r[1] == row["expected"]
            and _close(r[2], row["final_test_kl_mean"])
        ]
        if len(matches) != 1:
            problems.append(
                f"fair_comparison row {row['system']!r}: no unique recomputed match for mean {row['final_test_kl_mean']!r}"
            )
    if len(rows) != len(final_groups):
        problems.append(
            f"fair_comparison has {len(rows)} rows, fair_final has {len(final_groups)} candidates"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweeps-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    problems: list[str] = []
    tuning_records: list[dict] = []
    final_groups = None
    sweep_dirs = sorted(
        path.parent for path in args.sweeps_root.glob("*/sweep_status.json")
    )
    print(
        f"{'sweep':<28} {'candidates':>10} {'completed':>9} {'invalid':>7} {'expected':>8}  summary.json"
    )
    for sweep_dir in sweep_dirs:
        groups, records = recompute_sweep(sweep_dir)
        sweep_problems = check_summary(sweep_dir, groups)
        problems += sweep_problems
        completed = sum(r["status"] == "COMPLETED" for r in records)
        invalid = sum(len(g["invalid"]) for g in groups.values())
        print(
            f"{sweep_dir.name:<28} {len(groups):>10} {completed:>9} {invalid:>7} {len(records):>8}  {'match' if not sweep_problems else 'MISMATCH'}"
        )
        if sweep_dir.name in TUNING_SWEEPS:
            tuning_records += records
        if sweep_dir.name == "fair_final":
            final_groups = groups
    selections = reselect(tuning_records)
    selection_problems = check_selection(
        selections, args.results / "fair_tuning_selection.json"
    )
    problems += selection_problems
    print(
        f"\nselection rule re-applied to {len(tuning_records)} tuning runs over {len(selections)} models:"
    )
    for model, mine in sorted(selections.items()):
        print(
            f"  {model:<100} eligible {mine['n_eligible']}/{mine['n_candidates']}  selected lr={mine['selected'][0]:g} L2={mine['selected'][1]:g}  tied {len(mine['tied'])}"
        )
    print(
        "  fair_tuning_selection.json:",
        "match" if not selection_problems else "MISMATCH",
    )
    if final_groups is None:
        problems.append("fair_final sweep not found")
    else:
        comparison_problems = check_comparison(
            final_groups, args.results / "fair_comparison.json"
        )
        problems += comparison_problems
        print(
            "fair_comparison.json rows vs fair_final raw metrics:",
            "match" if not comparison_problems else "MISMATCH",
        )
    if problems:
        print("\nMISMATCHES:")
        for problem in problems:
            print("  " + problem)
        return 1
    print(
        f"\nALL CHECKS PASSED: {len(sweep_dirs)} sweeps recomputed from raw metrics.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
