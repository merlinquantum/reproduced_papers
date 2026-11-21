"""Build the Phase 05 comparison tables from structured run artifacts.

Usage (from the repository root)::

    python3 papers/photonic_trainability/utils/summarize_comparison.py \\
        --c9-run <estimator_scaling run dir> \\
        --training <label>=<training run dir> [--training ...] \\
        --out-dir papers/photonic_trainability/results

Writes ``comparison_c9.csv`` (one row per system, coefficient family and
|A|; mean and std over unitaries x draws), ``comparison_c9_slopes.csv`` and
``comparison_training.csv`` (one row per training run and method), and prints
them. Every expected cell is enumerated; missing, duplicate or non-finite
values make the script exit non-zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def c9_tables(run_dir: Path) -> tuple[list[dict], list[dict], list[str]]:
    """Per-(system, family, |A|) aggregates, per-family slopes, and problems found."""
    config = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    with (run_dir / "estimator_cases.csv").open(encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    problems = (
        [] if status["status"] == "COMPLETED" else [f"run status {status['status']}"]
    )
    expected_keys = {
        (u, family, int(size), draw)
        for u in range(int(config["unitaries"]))
        for family in config["families"]
        for size in config["term_counts"]
        for draw in range(int(config["draws_per_size"]))
    }
    observed = [
        (int(c["unitary"]), c["family"], int(c["terms"]), int(c["draw"])) for c in cases
    ]
    if len(set(observed)) != len(observed):
        problems.append("duplicate case rows")
    if set(observed) != expected_keys:
        problems.append(
            f"coverage mismatch: expected {len(expected_keys)}, observed {len(set(observed))}"
        )
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case["family"], int(case["terms"]))].append(case)
    rows = []
    systems = (
        (
            "quantum sampling (simulated exact single-shot variance)",
            "reproduced method",
            "n_quantum",
            "shots",
        ),
        (
            "B1 Lim-Oh term-wise, Neyman allocation",
            "additional fair baseline",
            "n_lim_oh",
            "classical samples",
        ),
        (
            "B1 Lim-Oh classical wall-clock",
            "additional fair baseline",
            "lim_oh_classical_seconds",
            "seconds (Apple M2, 1 core numpy)",
        ),
        (
            "B2 exact classical evaluation wall-clock",
            "additional fair baseline",
            "b2_exact_seconds",
            "seconds (Apple M2, Python permanents)",
        ),
    )
    for (family, size), group in sorted(grouped.items()):
        for name, role, column, unit in systems:
            values = np.array([float(c[column]) for c in group])
            if not np.all(np.isfinite(values)):
                problems.append(f"non-finite {column} for {family} |A|={size}")
            rows.append(
                {
                    "system": name,
                    "role": role,
                    "provenance": "locally executed",
                    "scope": "reduced-scope (M=16, n=8, simulator)",
                    "family": family,
                    "terms": size,
                    "successful_runs": len(group),
                    "expected_runs": int(config["unitaries"])
                    * int(config["draws_per_size"]),
                    "metric": f"{column} ({unit}) at eps={metrics['epsilon']}",
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "matching_basis": "same U, Fock input, PNP, eps and N = Var/eps^2 rule",
                }
            )
    slopes = []
    for family, decision in metrics["decisions"].items():
        per_unitary_q = [
            u["slopes"][family]["slope_quantum"] for u in metrics["per_unitary"]
        ]
        per_unitary_lo = [
            u["slopes"][family]["slope_lim_oh"] for u in metrics["per_unitary"]
        ]
        slopes.append(
            {
                "family": family,
                "slope_quantum_mean": float(np.mean(per_unitary_q)),
                "slope_quantum_std": float(np.std(per_unitary_q, ddof=1)),
                "slope_lim_oh_mean": float(np.mean(per_unitary_lo)),
                "slope_lim_oh_std": float(np.std(per_unitary_lo, ddof=1)),
                "slope_difference_mean": decision["mean_difference"],
                "slope_difference_std": decision["std_difference"],
                "unitaries": len(per_unitary_q),
                "outcome_vs_lim_oh_predeclared_rule": decision["outcome_vs_lim_oh"],
                "lim_oh_unbiased": metrics["lim_oh_unbiased"],
                "b2_max_abs_error": metrics["max_b2_abs_error"],
            }
        )
    return rows, slopes, problems


def training_table(entries: list[tuple[str, Path]]) -> tuple[list[dict], list[str]]:
    """One row per training run and method, from ``metrics.json`` and ``run_status.json``."""
    rows, problems = [], []
    for label, run_dir in entries:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
        config = json.loads(
            (run_dir / "config_snapshot.json").read_text(encoding="utf-8")
        )
        if status["status"] != "COMPLETED":
            problems.append(f"{label}: status {status['status']}")
        for method, aggregate in metrics["aggregates"].items():
            values = [
                r["improvement"] for r in metrics["results"] if r["method"] == method
            ]
            if not all(math.isfinite(v) for v in values):
                problems.append(f"{label}/{method}: non-finite improvement")
            rows.append(
                {
                    "run": label,
                    "role": "reproduced method"
                    if method == "gpsr_shots"
                    else "control (Phase 1 plan)",
                    "provenance": "locally executed",
                    "method": method,
                    "modes": config["modes"],
                    "hardware": (config.get("hardware") or {}).get("name", "ideal PNR"),
                    "successful_runs": len(values),
                    "expected_runs": len(config["seeds"]),
                    "improvement_mean": aggregate["improvement_mean"],
                    "improvement_std": aggregate["improvement_std"],
                    "fraction_decreasing_steps_mean": aggregate[
                        "fraction_decreasing_steps_mean"
                    ],
                    "shots_per_iteration": metrics["results"][0]["shots_per_iteration"],
                    "learning_rate": metrics["learning_rate"],
                    "wall_clock_seconds": _elapsed(status),
                }
            )
    return rows, problems


def _elapsed(status: dict) -> float:
    from datetime import datetime

    started = datetime.strptime(status["started_at"], "%Y-%m-%dT%H:%M:%SZ")
    completed = datetime.strptime(status["completed_at"], "%Y-%m-%dT%H:%M:%SZ")
    return (completed - started).total_seconds()


def main(argv: list[str] | None = None) -> int:
    """Generate and print the tables; non-zero exit on any coverage problem."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--c9-run", type=Path, required=True)
    parser.add_argument("--training", action="append", default=[], help="label=run_dir")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    c9_rows, slope_rows, problems = c9_tables(args.c9_run)
    entries = [
        (item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in args.training
    ]
    training_rows, training_problems = training_table(entries)
    problems += training_problems
    _write(args.out_dir / "comparison_c9.csv", c9_rows)
    _write(args.out_dir / "comparison_c9_slopes.csv", slope_rows)
    _write(args.out_dir / "comparison_training.csv", training_rows)
    for row in c9_rows:
        print(
            f"C9 | {row['system']} | {row['family']} | |A|={row['terms']} | "
            f"{row['successful_runs']}/{row['expected_runs']} | mean={row['mean']!r} std={row['std']!r}"
        )
    for row in slope_rows:
        print("SLOPES |", json.dumps(row))
    for row in training_rows:
        print(
            f"TRAIN | {row['run']} | {row['method']} | {row['successful_runs']}/{row['expected_runs']} | "
            f"improvement={row['improvement_mean']!r} ± {row['improvement_std']!r} | "
            f"wall={row['wall_clock_seconds']}s"
        )
    print("PROBLEMS |", problems or "none")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
