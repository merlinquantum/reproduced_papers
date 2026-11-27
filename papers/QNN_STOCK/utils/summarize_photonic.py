"""Tabulate the Phase 4 photonic sweep against the classical baselines.

Reads ``sweep_status.json`` and each completed run's ``metrics.json`` (full
precision). Writes ``aggregate.csv`` (seed statistics per model and cell),
``paired.csv`` (paired-by-seed differences with a 95 % t interval) and
``summary.md`` into ``--out``.

Usage::

    python papers/QNN_STOCK/utils/summarize_photonic.py <sweep_dir> --out results/phase4_p1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR))

from utils.summarize_sweep import T_975  # noqa: E402

PHOTONIC_VARIANTS = ("n1", "n2_dist", "n2_fock", "n2_unbunched")
BASELINES = ("B0_frequency", "B2_logistic")
METRICS = (
    "inference/test/log_loss",
    "inference/test/accuracy",
    "inference/train/log_loss",
    "keep_rate_mean",
    "n_params",
)
# (label, model, reference): the difference is model minus reference.
COMPARISONS = (
    *((f"{v}_vs_B2", v, "B2_logistic") for v in PHOTONIC_VARIANTS),
    *((f"{v}_vs_B0", v, "B0_frequency") for v in PHOTONIC_VARIANTS),
    ("n2_fock_vs_n1", "n2_fock", "n1"),
    ("n2_fock_vs_n2_dist", "n2_fock", "n2_dist"),
    ("n2_unbunched_vs_n2_fock", "n2_unbunched", "n2_fock"),
)
PAIRED_METRICS = ("inference/test/log_loss", "inference/test/accuracy")


def _leaf(entry: dict, path: str):
    value = entry
    for key in path.split("/"):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _cell(candidate: dict) -> tuple[str, str, str | None]:
    """``(size label, preprocessing, photonic variant or None)`` of a candidate."""

    base = candidate["base_config"]
    size = candidate["config"].split("__")[1]
    preprocessing = "ma5" if "_ma5" in base else "raw"
    if base.startswith("p4_photonic_"):
        variant = base.removeprefix("p4_photonic_").removesuffix("_merlin")
        return size, preprocessing, variant.removesuffix("_ma5")
    return size, preprocessing, None


def load_rows(sweep_dir: Path) -> list[dict]:
    """One row per (cell, ticker, model, seed) with the reported metrics."""

    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    rows = []
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads(
            (sweep_dir / run["metrics_path"]).read_text(encoding="utf-8")
        )
        size, preprocessing, variant = _cell(run["candidate"])
        for ticker, asset in metrics["assets"].items():
            for model_name, entry in asset["models"].items():
                if variant is None and model_name not in BASELINES:
                    continue
                rows.append(
                    {
                        "size": size,
                        "preprocessing": preprocessing,
                        "ticker": ticker,
                        "model": variant or model_name,
                        "seed": run["seed"],
                        **{metric: _leaf(entry, metric) for metric in METRICS},
                    }
                )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["size"], row["preprocessing"], row["ticker"], row["model"])].append(
            row
        )
    table = []
    for (size, preprocessing, ticker, model), members in sorted(groups.items()):
        record = {
            "size": size,
            "preprocessing": preprocessing,
            "ticker": ticker,
            "model": model,
            "n_seeds": len(members),
        }
        for metric in METRICS:
            values = [m[metric] for m in members if m[metric] is not None]
            if not values:
                continue
            record[f"{metric}/mean"] = statistics.fmean(values)
            record[f"{metric}/std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        table.append(record)
    return table


def paired(rows: list[dict]) -> list[dict]:
    by_key = {
        (r["size"], r["preprocessing"], r["ticker"], r["model"], r["seed"]): r
        for r in rows
    }
    cells = sorted({(r["size"], r["preprocessing"], r["ticker"]) for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    table = []
    for label, model, reference in COMPARISONS:
        for cell in cells:
            for metric in PAIRED_METRICS:
                differences = [
                    by_key[(*cell, model, seed)][metric]
                    - by_key[(*cell, reference, seed)][metric]
                    for seed in seeds
                    if (*cell, model, seed) in by_key
                    and (*cell, reference, seed) in by_key
                ]
                if len(differences) < 2:
                    continue
                mean = statistics.fmean(differences)
                half_width = (
                    T_975[len(differences) - 1]
                    * statistics.stdev(differences)
                    / math.sqrt(len(differences))
                )
                table.append(
                    {
                        "comparison": label,
                        "size": cell[0],
                        "preprocessing": cell[1],
                        "ticker": cell[2],
                        "metric": metric,
                        "n_pairs": len(differences),
                        "mean_difference": mean,
                        "ci_low": mean - half_width,
                        "ci_high": mean + half_width,
                        "excludes_zero": not (
                            mean - half_width <= 0.0 <= mean + half_width
                        ),
                    }
                )
    return table


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(aggregated: list[dict], pairs: list[dict]) -> str:
    lines = [
        "# Phase 4 photonic sweep: test log-loss (mean over seeds, AAPL / GOOGL)\n"
    ]
    index = {
        (a["size"], a["preprocessing"], a["ticker"], a["model"]): a for a in aggregated
    }
    models = (*BASELINES, *PHOTONIC_VARIANTS)
    lines.append("| size | data | " + " | ".join(models) + " |")
    lines.append("|---|---|" + "---|" * len(models))
    cells = sorted({(a["size"], a["preprocessing"]) for a in aggregated})
    for size, preprocessing in cells:
        values = []
        for model in models:
            pair = [
                index.get((size, preprocessing, ticker, model))
                for ticker in ("AAPL", "GOOGL")
            ]
            values.append(
                " / ".join(
                    "n/a" if p is None else f"{p['inference/test/log_loss/mean']:.4f}"
                    for p in pair
                )
            )
        lines.append(f"| {size} | {preprocessing} | " + " | ".join(values) + " |")
    lines.append("\n## Paired differences in test log-loss (model minus reference)\n")
    lines.append(
        "| comparison | cells | excludes 0, model lower | excludes 0, model higher | range of mean difference |"
    )
    lines.append("|---|---:|---:|---:|---|")
    for label, _, _ in COMPARISONS:
        rows = [
            p
            for p in pairs
            if p["comparison"] == label and p["metric"] == "inference/test/log_loss"
        ]
        if not rows:
            continue
        lower = sum(p["excludes_zero"] and p["mean_difference"] < 0 for p in rows)
        higher = sum(p["excludes_zero"] and p["mean_difference"] > 0 for p in rows)
        low = min(p["mean_difference"] for p in rows)
        high = max(p["mean_difference"] for p in rows)
        lines.append(
            f"| {label} | {len(rows)} | {lower} | {higher} | {low:+.4f} to {high:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = load_rows(args.sweep_dir)
    aggregated = aggregate(rows)
    pairs = paired(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "aggregate.csv", aggregated)
    _write_csv(args.out / "paired.csv", pairs)
    summary = _summary(aggregated, pairs)
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
