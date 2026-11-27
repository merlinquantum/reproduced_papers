"""Aggregate completed sweeps into result tables and comparison statistics.

Usage (from the repository root)::

    python3 papers/contextuality_inductive_bias/utils/build_results_table.py \\
        --sweep-dir <sweep_dir> [--sweep-dir ...] \\
        --out-dir papers/contextuality_inductive_bias/results --name fig8

Writes ``<name>_candidates.csv`` (every model x protocol x L2 candidate, full
precision: n, mean, sample std, min, max of the final and minimum test KL and
the final train NLL), ``<name>_comparisons.csv`` (per protocol, each biased
quantum candidate against every other unregularised candidate: difference of
mean final test KL with a Welch 95% interval, Welch t-test and Mann-Whitney U
p-values, two-sided) and ``<name>_tables.md`` (the same, rounded for the
README). Values are read only from ``metrics.json`` files listed as COMPLETED
in ``sweep_status.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_io import candidate_key, describe, load_sweep  # noqa: E402

FIELDS = ("final_test_kl", "min_test_kl", "final_train_nll")


def _values(record) -> dict:
    return {
        "final_test_kl": record.metrics["final"]["test_kl"],
        "min_test_kl": record.metrics["min_test_kl"]["test_kl"],
        "final_train_nll": record.metrics["final"]["train_nll"],
    }


def group(records) -> dict:
    """Group runs by model configuration; reject duplicate seeds."""
    groups: dict[tuple, dict] = {}
    for record in records:
        key = candidate_key(record.candidate)
        entry = groups.setdefault(
            key,
            {
                "candidate": record.candidate,
                "seeds": [],
                "values": {f: [] for f in FIELDS},
            },
        )
        if record.seed in entry["seeds"]:
            raise ValueError(f"Duplicate seed {record.seed} for {key}")
        entry["seeds"].append(record.seed)
        for field, value in _values(record).items():
            if not np.isfinite(value):
                raise ValueError(f"Non-finite {field} for {key}, seed {record.seed}")
            entry["values"][field].append(value)
    return groups


def welch_interval(
    a: np.ndarray, b: np.ndarray, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean difference a - b with its Welch-Satterthwaite confidence interval."""
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    dof = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    half = stats.t.ppf(0.5 + level / 2, dof) * np.sqrt(va + vb)
    diff = a.mean() - b.mean()
    return diff, diff - half, diff + half


def build(
    sweep_dirs: list[Path],
    out_dir: Path,
    name: str,
    reference_kinds: tuple[str, ...] = ("biased_quantum",),
) -> dict:
    """Write the tables and return the parsed results."""
    records = [r for d in sweep_dirs for r in load_sweep(d)]
    groups = group(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _key, entry in sorted(
        groups.items(), key=lambda item: (item[0].protocol, str(item[0]))
    ):
        cand = entry["candidate"]
        row = {
            "protocol": cand["protocol"],
            "model": describe(cand),
            "l2": cand["l2"],
            "steps": cand["steps"],
            "train": cand["train_source"]
            if cand["train_seed"] is None
            else f"generated seed {cand['train_seed']}",
            "n": len(entry["seeds"]),
            "seeds": " ".join(map(str, sorted(entry["seeds"]))),
        }
        for field in FIELDS:
            values = np.array(entry["values"][field])
            row[f"{field}_mean"] = values.mean()
            row[f"{field}_std"] = values.std(ddof=1) if values.size > 1 else 0.0
            row[f"{field}_min"] = values.min()
            row[f"{field}_max"] = values.max()
        rows.append(row)
    with (out_dir / f"{name}_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    comparisons = []
    for key, biased in groups.items():
        if key.kind not in reference_kinds or key.l2 != 0.0:
            continue
        a = np.array(biased["values"]["final_test_kl"])
        for other_key, other in groups.items():
            if (
                other_key == key
                or other_key.protocol != key.protocol
                or other_key.l2 != 0.0
                or other_key.steps != key.steps
                or (other_key.train_source, other_key.train_seed)
                != (key.train_source, key.train_seed)
            ):
                continue
            b = np.array(other["values"]["final_test_kl"])
            diff, low, high = welch_interval(a, b)
            comparisons.append(
                {
                    "protocol": key.protocol,
                    "reference": describe(biased["candidate"]),
                    "other": describe(other["candidate"]),
                    "n_reference": a.size,
                    "n_other": b.size,
                    "mean_reference": a.mean(),
                    "mean_other": b.mean(),
                    "diff_reference_minus_other": diff,
                    "diff_ci95_low": low,
                    "diff_ci95_high": high,
                    "welch_p": stats.ttest_ind(a, b, equal_var=False).pvalue,
                    "mannwhitney_p": stats.mannwhitneyu(
                        a, b, alternative="two-sided"
                    ).pvalue,
                }
            )
    if comparisons:
        with (out_dir / f"{name}_comparisons.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)

    lines = [f"### {name}: candidates (final test KL is the primary metric)", ""]
    lines.append(
        "| Protocol | Model | L2 | Steps | Train data | n | Final test KL (mean +- std) | Min test KL (mean) | Final train NLL (mean) |"
    )
    lines.append("|---|---|---:|---:|---|---:|---|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['model']} | {row['l2']:g} | {row['steps']} | {row['train']} | {row['n']} | "
            f"{row['final_test_kl_mean']:.3e} +- {row['final_test_kl_std']:.1e} | {row['min_test_kl_mean']:.3e} | {row['final_train_nll_mean']:.4f} |"
        )
    if comparisons:
        lines += [
            "",
            f"### {name}: reference models vs other model classes (final test KL, two-sided)",
            "",
        ]
        lines.append(
            "| Protocol | Reference | Other | n | Diff (reference - other) | 95% CI (Welch) | Welch p | Mann-Whitney p |"
        )
        lines.append("|---|---|---|---|---:|---|---:|---:|")
        for c in comparisons:
            lines.append(
                f"| {c['protocol']} | {c['reference']} | {c['other']} | {c['n_reference']} vs {c['n_other']} | {c['diff_reference_minus_other']:.2e} | "
                f"[{c['diff_ci95_low']:.2e}, {c['diff_ci95_high']:.2e}] | {c['welch_p']:.1e} | {c['mannwhitney_p']:.1e} |"
            )
    (out_dir / f"{name}_tables.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    result = {
        "sweeps": [str(d) for d in sweep_dirs],
        "candidates": rows,
        "comparisons": comparisons,
    }
    (out_dir / f"{name}_results.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--reference-kind",
        action="append",
        default=None,
        help="Model kinds compared against every other candidate (default biased_quantum)",
    )
    args = parser.parse_args(argv)
    result = build(
        args.sweep_dir,
        args.out_dir,
        args.name,
        tuple(args.reference_kind or ("biased_quantum",)),
    )
    print(
        f"{len(result['candidates'])} candidates, {len(result['comparisons'])} comparisons -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
