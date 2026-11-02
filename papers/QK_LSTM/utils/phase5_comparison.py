#!/usr/bin/env python3
"""Phase 5 fair-comparison table and claim decisions (frozen protocol in LOG.md).

Reads ``sweep_status.json``, every included run's ``run_status.json`` and
``metrics.json``. It enumerates expected vs observed runs, rejects missing,
duplicate, non-finite or non-COMPLETED runs, computes mean ± std of test RMSE
(observed AQI targets), wall-clock from the status timestamps, and the
predeclared paired decision rule:

    A better than B  <=> 95 % t-interval of mean(RMSE_A - RMSE_B) < 0
    A worse than B   <=> interval > 0
    otherwise        "not separated"

Writes ``comparison.{json,md}`` to ``--out``.

Usage::

    .venv/bin/python papers/QK_LSTM/utils/phase5_comparison.py --spec papers/QK_LSTM/results/phase5/phase5_spec.json --out papers/QK_LSTM/results/phase5
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import statistics
import time
from pathlib import Path

from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]


def _seconds(stamp: str) -> int:
    return calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))


def load_system(sweep_dir: str, candidate: str, expected_seeds: list[int]) -> dict:
    sweep_dir = (REPO_ROOT / sweep_dir).resolve()
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    runs = [r for r in status["runs"] if r["candidate"] == candidate]
    by_seed: dict[int, dict] = {}
    rejected = []
    for run in runs:
        if run["status"] != "COMPLETED":
            rejected.append((run["seed"], run["status"]))
            continue
        if run["seed"] in by_seed:
            raise ValueError(f"Duplicate seed {run['seed']} for {candidate}")
        run_status = json.loads((sweep_dir / run["run_status"]).read_text())
        if run_status["status"] != "COMPLETED":
            rejected.append((run["seed"], run_status["status"]))
            continue
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        rmse = float(metrics["splits"]["test"]["observed_targets"]["rmse"])
        if not math.isfinite(rmse):
            rejected.append((run["seed"], "non-finite"))
            continue
        by_seed[run["seed"]] = {
            "test": metrics["splits"]["test"]["observed_targets"],
            "params": metrics["trainable_parameters"],
            "wall_seconds": _seconds(run_status["completed_at"])
            - _seconds(run_status["started_at"]),
            "commit": run_status["code"]["commit"],
        }
    missing = sorted(set(expected_seeds) - set(by_seed))
    return {
        "by_seed": by_seed,
        "missing": missing,
        "rejected": rejected,
        "sweep_id": status["sweep_id"],
    }


def paired(a: dict, b: dict) -> dict:
    seeds = sorted(set(a["by_seed"]) & set(b["by_seed"]))
    diffs = [
        a["by_seed"][s]["test"]["rmse"] - b["by_seed"][s]["test"]["rmse"] for s in seeds
    ]
    mean = statistics.fmean(diffs)
    std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    half = (
        stats.t.ppf(0.975, len(diffs) - 1) * std / math.sqrt(len(diffs))
        if std > 0
        else 0.0
    )
    low, high = mean - half, mean + half
    decision = "A better" if high < 0 else ("A worse" if low > 0 else "not separated")
    return {
        "n": len(diffs),
        "mean": mean,
        "std": std,
        "ci95": [low, high],
        "decision": decision,
        "a_better_seeds": sum(d < 0 for d in diffs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    seeds = spec["seeds"]

    tables = {}
    lines = [f"# {spec['title']}", "", spec["note"], ""]
    for table_name, table in spec["tables"].items():
        systems = {
            name: load_system(entry["sweep"], entry["candidate"], seeds) | entry
            for name, entry in table["systems"].items()
        }
        rows = []
        lines += [f"## {table_name}", "", table["scope"], ""]
        lines += [
            "| System | Role | Provenance | Scope | Successful / expected runs | Test RMSE (mean) | Std | Matching basis | Trainable params | Wall-clock / run (min) | Status |",
            "|---|---|---|---|---:|---:|---:|---|---:|---:|---|",
        ]
        for name, system in systems.items():
            values = [v["test"]["rmse"] for v in system["by_seed"].values()]
            walls = [v["wall_seconds"] for v in system["by_seed"].values()]
            params = sorted({v["params"] for v in system["by_seed"].values()})
            complete = not system["missing"] and not system["rejected"]
            row = {
                "system": name,
                "role": system["role"],
                "provenance": f"local run ({system['sweep_id']}:{system['candidate']})",
                "scope": system["scope"],
                "successful": len(values),
                "expected": len(seeds),
                "rmse_mean": statistics.fmean(values),
                "rmse_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "r2_mean": statistics.fmean(
                    v["test"]["r2"] for v in system["by_seed"].values()
                ),
                "matching": system["matching"],
                "params": params,
                "wall_minutes_mean": statistics.fmean(walls) / 60,
                "status": "COMPLETE"
                if complete
                else f"PARTIAL (missing {system['missing']}, rejected {system['rejected']})",
                "commits": sorted({v["commit"] for v in system["by_seed"].values()}),
            }
            rows.append(row)
        for row in sorted(rows, key=lambda r: r["rmse_mean"]):
            lines.append(
                f"| {row['system']} | {row['role']} | {row['provenance']} | {row['scope']} | {row['successful']} / {row['expected']} | "
                f"{row['rmse_mean']:.3f} | {row['rmse_std']:.3f} | {row['matching']} | {'/'.join(map(str, row['params']))} | {row['wall_minutes_mean']:.1f} | {row['status']} |"
            )
        for name, value in spec["paper"].items():
            lines.append(
                f"| {name} | paper-reported | paper Table I/II (single run) | INCOMPARABLE for ranking (protocol unknown) | 1 / 1 | {value['rmse']:.2f} | — | — | {value['params']} | — | reference only |"
            )
        decisions = []
        lines += [
            "",
            "Paired decisions (frozen rule: 95 % t-CI of mean ΔRMSE, A − B):",
            "",
            "| Claim | A | B | Commensurability | ΔRMSE mean | 95 % CI | A better (seeds) | Decision |",
            "|---|---|---|---|---:|---|---:|---|",
        ]
        for comparison in table["comparisons"]:
            result = paired(systems[comparison["a"]], systems[comparison["b"]])
            result.update(comparison)
            decisions.append(result)
            lines.append(
                f"| {comparison['claim']} | {comparison['a']} | {comparison['b']} | {comparison['commensurability']} | {result['mean']:+.3f} | "
                f"[{result['ci95'][0]:+.3f}, {result['ci95'][1]:+.3f}] | {result['a_better_seeds']}/{result['n']} | **{result['decision']}** |"
            )
        lines.append("")
        all_commits = sorted({c for r in rows for c in r["commits"]})
        lines.append(f"Code commits across included runs: {all_commits}")
        lines.append("")
        tables[table_name] = {
            "rows": rows,
            "decisions": decisions,
            "commits": all_commits,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(
        json.dumps({"spec": spec, "tables": tables}, indent=2) + "\n", encoding="utf-8"
    )
    markdown = "\n".join(lines) + "\n"
    (args.out / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
