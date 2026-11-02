#!/usr/bin/env python3
"""Claim-level statistics across sweeps (paired by seed, 95 % t-intervals).

Each table entry is ``<sweep_dir>:<candidate>``. Values are read from the
runs' ``metrics.json`` (test split, observed AQI targets). For each comparison
``A vs B`` the per-seed differences ``A - B`` are summarised by their mean, sample
std, a two-sided 95 % t-interval and the number of seeds where A is better.
Deterministic models (persistence, ridge) have identical values on every seed
and are paired with each seed of the other model.

Usage::

    .venv/bin/python papers/QK_LSTM/utils/claim_statistics.py --spec papers/QK_LSTM/results/claims/<table>.spec.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
METRICS = ("rmse", "mae", "mape", "r2")


def load_entry(entry: str) -> dict[int, dict]:
    sweep_dir, candidate = entry.rsplit(":", 1)
    sweep_dir = (REPO_ROOT / sweep_dir).resolve()
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    values: dict[int, dict] = {}
    for run in status["runs"]:
        if run["candidate"] != candidate:
            continue
        if run["status"] != "COMPLETED":
            raise ValueError(f"{entry}: seed {run['seed']} is {run['status']}")
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        values[run["seed"]] = {
            "metrics": metrics["splits"]["test"]["observed_targets"],
            "params": metrics["trainable_parameters"],
        }
    if not values:
        raise ValueError(f"No runs for {entry}")
    return values


def paired(a: dict[int, dict], b: dict[int, dict], metric: str) -> dict:
    seeds = sorted(set(a) & set(b))
    diffs = [a[s]["metrics"][metric] - b[s]["metrics"][metric] for s in seeds]
    mean = statistics.fmean(diffs)
    std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    half_width = (
        stats.t.ppf(0.975, len(diffs) - 1) * std / math.sqrt(len(diffs))
        if len(diffs) > 1 and std > 0
        else 0.0
    )
    better = sum(d < 0 for d in diffs) if metric != "r2" else sum(d > 0 for d in diffs)
    return {
        "n": len(diffs),
        "mean": mean,
        "std": std,
        "ci95": [mean - half_width, mean + half_width],
        "a_better_seeds": better,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    entries = {name: load_entry(entry) for name, entry in spec["models"].items()}

    lines = [
        f"# {spec['title']}",
        "",
        spec["scope"],
        "",
        "| Model | Params | Seeds | RMSE | MAE | MAPE (%) | R² |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    table = {}
    for name, values in sorted(
        entries.items(),
        key=lambda kv: statistics.fmean(v["metrics"]["rmse"] for v in kv[1].values()),
    ):
        row = {
            "params": sorted({v["params"] for v in values.values()}),
            "seeds": sorted(values),
        }
        cells = []
        for metric in METRICS:
            series = [values[s]["metrics"][metric] for s in sorted(values)]
            row[metric] = {
                "mean": statistics.fmean(series),
                "std": statistics.stdev(series) if len(series) > 1 else 0.0,
                "values": series,
            }
            fmt = "{:.3f} ± {:.3f}" if metric == "r2" else "{:.2f} ± {:.2f}"
            cells.append(fmt.format(row[metric]["mean"], row[metric]["std"]))
        table[name] = row
        lines.append(
            f"| {name} | {'/'.join(map(str, row['params']))} | {len(values)} | "
            + " | ".join(cells)
            + " |"
        )
    for name, values in spec.get("paper", {}).items():
        lines.append(
            f"| {name} | {values['params']} | 1 | {values['rmse']:.2f} | {values['mae']:.2f} | {values['mape']:.2f} | {values['r2']:.2f} |"
        )

    comparisons = {}
    lines += [
        "",
        "Paired same-seed differences (A − B), test RMSE and R²; 95 % t-interval:",
        "",
        "| Claim | A vs B | ΔRMSE mean [95 % CI] | A better (RMSE) | ΔR² mean [95 % CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for comparison in spec["comparisons"]:
        a, b = entries[comparison["a"]], entries[comparison["b"]]
        result = {metric: paired(a, b, metric) for metric in ("rmse", "r2")}
        comparisons[f"{comparison['a']} vs {comparison['b']}"] = {
            "claim": comparison["claim"],
            **result,
        }
        r, q = result["rmse"], result["r2"]
        lines.append(
            f"| {comparison['claim']} | {comparison['a']} vs {comparison['b']} | {r['mean']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | "
            f"{r['a_better_seeds']}/{r['n']} | {q['mean']:+.3f} [{q['ci95'][0]:+.3f}, {q['ci95'][1]:+.3f}] |"
        )
    markdown = "\n".join(lines) + "\n"
    out_dir = args.spec.parent
    stem = args.spec.name.removesuffix(".spec.json")
    if stem == args.spec.name:
        raise ValueError("Spec files must be named <table>.spec.json")
    (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps({"table": table, "comparisons": comparisons, "spec": spec}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
