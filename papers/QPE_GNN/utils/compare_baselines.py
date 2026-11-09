"""Build the Phase 5 fair-comparison table from structured result artifacts.

A JSON spec (``configs/comparison/phase5_comparison.json``) lists, per
claim, the systems to report: their role, provenance, scope and matching
basis, and where their numbers live. Every locally executed value is read
from a ``sweep_summary.json`` or ``srg_metrics.json``; paper values come from
the spec and are labelled as paper-reported. Pairwise outcomes are read from
the summaries' predeclared tie-rule comparisons, never recomputed by hand.

Writes ``comparison_table.json`` and ``comparison_table.md`` to
``--output-dir``.

Usage (from ``papers/QPE_GNN``)::

    python utils/compare_baselines.py --spec configs/comparison/phase5_comparison.json \
        --output-dir results/comparison
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sweep_row(system: dict) -> dict:
    """Read one candidate's aggregate from a sweep summary."""
    summary = _read_json(PROJECT_DIR / system["summary"])
    records = [
        r for r in summary["candidates"] if r["candidate"] == system["candidate"]
    ]
    if len(records) != 1:
        raise ValueError(f"{system['candidate']} not found once in {system['summary']}")
    record = records[0]
    if any(not math.isfinite(value) for value in record["values"]):
        raise ValueError(f"non-finite metric in {system['candidate']}")
    return {
        "metric_name": summary["metric"],
        "metric": record["mean"],
        "variability": record["std"],
        "per_seed": record["values"],
        "successful_runs": record["observed_runs"],
        "expected_runs": record["expected_runs"],
        "num_params": record["num_params"],
        "wall_clock_s": record["mean_wall_clock_s"],
        "wall_clock_note": "mean per run",
        "sweep_verdict": summary["verdict"],
    }


def _srg_row(system: dict) -> dict:
    """Read one method's SRG pair count, summed over both families."""
    metrics = _read_json(PROJECT_DIR / system["metrics"])
    results = [family[system["method"]] for family in metrics["families"].values()]
    distinguished = sum(r["distinguished_pairs"] for r in results)
    total = sum(r["total_pairs"] for r in results)
    control = max(r["maximum_isomorphic_control_distance"] for r in results)
    if "wall_clock_s" in results[0]:
        wall_clock = sum(r["wall_clock_s"] for r in results)
        note = "this method, both families"
    else:
        status = _read_json(PROJECT_DIR / system["run_status"])
        started = datetime.fromisoformat(status["started_at"].replace("Z", "+00:00"))
        ended = datetime.fromisoformat(status["completed_at"].replace("Z", "+00:00"))
        wall_clock = (ended - started).total_seconds()
        note = "whole run (all methods)"
    return {
        "metric_name": "distinguished_pairs",
        "metric": distinguished,
        "metric_total": total,
        "variability": None,
        "max_isomorphic_control": control,
        "successful_runs": 1,
        "expected_runs": 1,
        "num_params": None,
        "wall_clock_s": wall_clock,
        "wall_clock_note": note,
    }


def _pairwise(comparison: dict) -> dict:
    """Read a predeclared tie-rule outcome from a sweep summary."""
    summary = _read_json(PROJECT_DIR / comparison["summary"])
    pair = {comparison["a"], comparison["b"]}
    matches = [c for c in summary["comparisons"] if {c["a"], c["b"]} == pair]
    if len(matches) != 1:
        raise ValueError(
            f"comparison {sorted(pair)} not found once in {comparison['summary']}"
        )
    match = matches[0]
    return {
        "claim": comparison["claim"],
        # The summary's own order, so the margin sign (b - a) stays correct.
        "a": match["a"],
        "b": match["b"],
        "verdict": match["verdict"],
        "p_value": match["p_value"],
        "margin_b_minus_a": match["mean_margin_b_minus_a"],
        "margin_threshold": match["margin_threshold"],
        "commensurability": comparison["commensurability"],
        "note": comparison.get("note", ""),
    }


def build_table(spec: dict) -> dict:
    """Assemble every row and pairwise outcome named in the spec."""
    rows = []
    for system in spec["systems"]:
        row = {
            key: system[key]
            for key in (
                "claim",
                "system",
                "role",
                "provenance",
                "scope",
                "matching",
                "status",
            )
        }
        if system["source"] == "paper":
            row.update(
                metric_name=system["metric_name"],
                metric=system["paper_value"],
                variability=system.get("paper_std"),
                successful_runs=None,
                expected_runs=None,
                num_params=system.get("num_params"),
                wall_clock_s=None,
                wall_clock_note="not reported",
            )
        elif system["source"] == "sweep":
            row.update(_sweep_row(system))
        elif system["source"] == "srg":
            row.update(_srg_row(system))
        else:
            raise ValueError(f"unknown source {system['source']}")
        rows.append(row)
    pairwise = [_pairwise(comparison) for comparison in spec["comparisons"]]
    return {"spec_version": spec["spec_version"], "rows": rows, "pairwise": pairwise}


def _format_metric(row: dict) -> str:
    if row["metric_name"] == "distinguished_pairs":
        return f"{row['metric']}/{row['metric_total']}"
    value = f"{row['metric']:.4f}"
    if row["variability"] is not None:
        value += f" ± {row['variability']:.4f}"
    return value


def to_markdown(table: dict) -> str:
    """Render the table as Markdown."""
    lines = [
        "| Claim | System | Role | Provenance | Scope | Runs | Metric | Matching basis | Params | Wall-clock | Status |",
        "|---|---|---|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in table["rows"]:
        runs = (
            "—"
            if row["expected_runs"] is None
            else f"{row['successful_runs']}/{row['expected_runs']}"
        )
        params = "—" if row["num_params"] is None else f"{row['num_params']:,}"
        wall = (
            "—"
            if row["wall_clock_s"] is None
            else f"{row['wall_clock_s']:.0f} s ({row['wall_clock_note']})"
        )
        lines.append(
            f"| {row['claim']} | {row['system']} | {row['role']} | {row['provenance']} | {row['scope']} "
            f"| {runs} | {row['metric_name']} {_format_metric(row)} | {row['matching']} | {params} | {wall} | {row['status']} |"
        )
    lines += [
        "",
        "| Claim | A | B | Tie-rule verdict | p | Commensurability | Note |",
        "|---|---|---|---|---:|---|---|",
    ]
    for pair in table["pairwise"]:
        lines.append(
            f"| {pair['claim']} | {pair['a']} | {pair['b']} | {pair['verdict']} | {pair['p_value']:.3g} "
            f"| {pair['commensurability']} | {pair['note']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_DIR / path

    table = build_table(_read_json(resolve(args.spec)))
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_table.json").write_text(
        json.dumps(table, indent=2) + "\n", encoding="utf-8"
    )
    markdown = to_markdown(table)
    (output_dir / "comparison_table.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
