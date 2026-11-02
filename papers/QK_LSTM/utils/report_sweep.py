#!/usr/bin/env python3
"""Build the result table of a sweep from its workers' ``metrics.json`` files.

Reads only structured artifacts (``sweep_status.json`` and each completed run's
``metrics.json``); never parses logs. Verifies that the observed
candidate-by-seed set matches the plan, that every value is finite, and that
all runs share one code commit, then writes ``report.json`` and ``report.md``
to the sweep directory and to ``papers/QK_LSTM/results/<sweep_name>/``.

Usage::

    .venv/bin/python papers/QK_LSTM/utils/report_sweep.py <SWEEP_DIR> [--reference B1_lstm_1873]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
METRICS = ("rmse", "mae", "mape", "r2")
PAPER = {
    "QK-LSTM (paper)": {
        "rmse": 9.20,
        "mae": 7.15,
        "mape": 9.14,
        "r2": 0.84,
        "params": 209,
    },
    "LSTM (paper)": {
        "rmse": 15.94,
        "mae": 11.07,
        "mape": 13.32,
        "r2": 0.78,
        "params": 1873,
    },
}


def _load(sweep_dir: Path) -> tuple[dict, dict[str, dict[int, dict]]]:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    plan = json.loads((sweep_dir / "plan.json").read_text())
    results: dict[str, dict[int, dict]] = {c["label"]: {} for c in plan["candidates"]}
    commits = set()
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        if run["seed"] in results[run["candidate"]]:
            raise ValueError(
                f"Duplicate result for {run['candidate']} seed {run['seed']}"
            )
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        run_status = json.loads((sweep_dir / run["run_status"]).read_text())
        commits.add(run_status["code"]["commit"])
        results[run["candidate"]][run["seed"]] = metrics
    if len(commits) > 1:
        raise ValueError(f"Runs span several code commits: {commits}")
    expected = {(c["label"], s) for c in plan["candidates"] for s in plan["seeds"]}
    observed = {(c, s) for c, seeds in results.items() for s in seeds}
    status["_coverage"] = "COMPLETE SWEEP" if observed == expected else "PARTIAL SWEEP"
    status["_missing"] = sorted(expected - observed)
    status["_plan"] = plan
    return status, results


def _value(
    metrics: dict, split: str, metric: str, scoring: str = "observed_targets"
) -> float:
    value = float(metrics["splits"][split][scoring][metric])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {metric}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--reference",
        default="B1_lstm_1873",
        help="candidate used for paired seed-wise differences",
    )
    args = parser.parse_args(argv)
    sweep_dir = args.sweep_dir.resolve()
    status, results = _load(sweep_dir)
    plan = status["_plan"]

    rows = []
    for label, by_seed in results.items():
        seeds = sorted(by_seed)
        if not seeds:
            # Reported under "missing"; no aggregate for a candidate without runs.
            continue
        row = {
            "candidate": label,
            "n_seeds": len(seeds),
            "expected_seeds": len(plan["seeds"]),
            "seeds": seeds,
        }
        row["trainable_parameters"] = sorted(
            {m["trainable_parameters"] for m in by_seed.values()}
        )
        for metric in METRICS:
            values = [_value(by_seed[s], args.split, metric) for s in seeds]
            row[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "values": values,
            }
            train_values = [_value(by_seed[s], "train", metric) for s in seeds]
            row[f"train_{metric}_mean"] = statistics.fmean(train_values)
        rows.append(row)

    # Paired (same-seed) RMSE differences vs the reference; deterministic models
    # (identical across seeds) are paired with each seed of the reference.
    reference = results.get(args.reference, {})
    paired = {}
    for label, by_seed in results.items():
        if label == args.reference or not reference:
            continue
        diffs = [
            _value(by_seed[s], args.split, "rmse")
            - _value(reference[s], args.split, "rmse")
            for s in sorted(set(by_seed) & set(reference))
        ]
        if diffs:
            paired[label] = {
                "mean_rmse_minus_reference": statistics.fmean(diffs),
                "std": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
                "n_pairs": len(diffs),
                "n_better_than_reference": sum(d < 0 for d in diffs),
            }

    report = {
        "sweep_id": status["sweep_id"],
        "split": args.split,
        "scoring": "observed_targets (AQI units)",
        "coverage": status["_coverage"],
        "missing": status["_missing"],
        "rows": rows,
        "paired_vs_reference": {"reference": args.reference, "differences": paired},
        "paper": PAPER,
    }

    lines = [
        f"# {status['sweep_id']}: {args.split} metrics (observed AQI targets), mean ± std over seeds",
        "",
        f"Coverage: **{status['_coverage']}**"
        + (f"; missing {status['_missing']}" if status["_missing"] else ""),
        "",
        "| Candidate | Params | Seeds | RMSE | MAE | MAPE (%) | R² | Train RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: r["rmse"]["mean"]):
        cells = [
            f"{row[m]['mean']:.2f} ± {row[m]['std']:.2f}"
            if m != "r2"
            else f"{row[m]['mean']:.3f} ± {row[m]['std']:.3f}"
            for m in METRICS
        ]
        params = "/".join(str(p) for p in row["trainable_parameters"])
        lines.append(
            f"| {row['candidate']} | {params} | {row['n_seeds']}/{row['expected_seeds']} | "
            + " | ".join(cells)
            + f" | {row['train_rmse_mean']:.2f} |"
        )
    for name, values in PAPER.items():
        lines.append(
            f"| {name} | {values['params']} | 1 | {values['rmse']:.2f} | {values['mae']:.2f} | {values['mape']:.2f} | {values['r2']:.2f} | — |"
        )
    if paired:
        lines += [
            "",
            f"Paired test-RMSE difference vs `{args.reference}` (same seed; negative = better):",
            "",
            "| Candidate | Mean Δ RMSE | Std | Seeds better |",
            "|---|---:|---:|---:|",
        ]
        for label, entry in sorted(
            paired.items(), key=lambda kv: kv[1]["mean_rmse_minus_reference"]
        ):
            lines.append(
                f"| {label} | {entry['mean_rmse_minus_reference']:+.2f} | {entry['std']:.2f} | {entry['n_better_than_reference']}/{entry['n_pairs']} |"
            )
    markdown = "\n".join(lines) + "\n"

    results_dir = PROJECT_DIR / "results" / plan["sweep_name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    for directory in (sweep_dir, results_dir):
        (directory / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (directory / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
