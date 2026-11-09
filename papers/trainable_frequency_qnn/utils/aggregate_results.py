#!/usr/bin/env python3
"""Aggregate a completed sweep into result tables (JSON + Markdown).

Reads only structured artifacts: the sweep's ``sweep_status.json`` and every
completed run's ``metrics.json``. Writes ``<out>/<sweep_id>_results.json`` and
``<out>/<sweep_id>_results.md``.

Kinds
-----
``toy``  : per-candidate statistics of train / dense MSE (+ learned multipliers).
``fig3`` : E5 box statistics across the 7 datasets (paper Fig. 3), per model.
``ns``   : MAERM table (min / max / mean over t = 0.5..5.5) per candidate, as in
           paper Tables I and II.

Paired comparisons use the Wilcoxon signed-rank test on per-seed values of the
two candidates (seeds are paired: same ansatz initialisation).

Example
-------
From the repository root::

    python papers/trainable_frequency_qnn/utils/aggregate_results.py \\
        --sweep-dir outdir/sweeps/e2_fig2b --kind toy --pair tf:ff --pair tf:ffs
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import wilcoxon

PAPER_DIR = Path(__file__).resolve().parents[1]
logger = logging.getLogger("aggregate")
FIELDS = ("u", "v", "p")
# Primary MAERM divides by median(|y|): it reproduces the paper's TF v error
# (30.4 %) where the signed median of v crosses zero (LOG.md, A10). The signed
# variant, as literally written in the paper, is kept with the "_signed" suffix.
MAERM_VARIANTS = {"": "maerm_abs_median", "_signed": "maerm"}


def load_runs(
    sweep_dirs: list[Path],
) -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    """Return ``{candidate: {seed: metrics}}`` of completed runs and the statuses.

    Several sweep folders can be merged, e.g. a sweep and its retry, or a
    control sweep with the comparison it is paired against.

    Raises
    ------
    ValueError
        If a (candidate, seed) is completed more than once across the sweeps.
    """
    runs: dict[str, dict[int, dict[str, Any]]] = {}
    statuses = []
    for sweep_dir in sweep_dirs:
        status = json.loads(
            (sweep_dir / "sweep_status.json").read_text(encoding="utf-8")
        )
        statuses.append(status)
        for entry in status["runs"]:
            if entry["status"] != "COMPLETED":
                continue
            metrics = json.loads(
                (sweep_dir / entry["metrics_path"]).read_text(encoding="utf-8")
            )
            by_seed = runs.setdefault(entry["candidate"], {})
            if entry["seed"] in by_seed:
                raise ValueError(
                    f"Duplicate run for {entry['candidate']} seed {entry['seed']}."
                )
            by_seed[entry["seed"]] = metrics
    return runs, statuses


def describe(values: list[float]) -> dict[str, Any]:
    """Return n, mean, std, s.e.m., median, min and max (full precision)."""
    for value in values:
        if not math.isfinite(value):
            raise ValueError("Non-finite metric value.")
    n = len(values)
    std = statistics.stdev(values) if n > 1 else None
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "std": std,
        "sem": std / math.sqrt(n) if std is not None else None,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def paired_test(first: dict[int, float], second: dict[int, float]) -> dict[str, Any]:
    """Wilcoxon signed-rank test on seeds present in both candidates."""
    seeds = sorted(set(first) & set(second))
    differences = [first[s] - second[s] for s in seeds]
    result = {
        "seeds": seeds,
        "mean_difference": statistics.fmean(differences),
        "first_lower_count": sum(d < 0 for d in differences),
    }
    if len(seeds) >= 5 and any(d != 0 for d in differences):
        test = wilcoxon(differences)
        result["wilcoxon_statistic"] = float(test.statistic)
        result["wilcoxon_p_value"] = float(test.pvalue)
    return result


def aggregate_toy(runs, pairs) -> dict[str, Any]:
    table = {}
    for candidate, by_seed in runs.items():
        row = {
            "train_mse": describe([m["train_mse"] for m in by_seed.values()]),
            "dense_mse": describe([m["dense_mse"] for m in by_seed.values()]),
            "parameters": next(iter(by_seed.values()))["parameters"],
        }
        multipliers = [
            m["encoding_multipliers"]
            for m in by_seed.values()
            if "encoding_multipliers" in m
        ]
        if multipliers:
            row["encoding_multipliers_by_seed"] = {
                str(seed): m.get("encoding_multipliers")
                for seed, m in sorted(by_seed.items())
            }
            row["dft_peaks_by_seed"] = {
                str(seed): m.get("dft_peaks") for seed, m in sorted(by_seed.items())
            }
        table[candidate] = row
    comparisons = {}
    for first, second in pairs:
        for metric in ("train_mse", "dense_mse"):
            comparisons[f"{first}_vs_{second}:{metric}"] = paired_test(
                {s: m[metric] for s, m in runs[first].items()},
                {s: m[metric] for s, m in runs[second].items()},
            )
    return {"candidates": table, "paired_comparisons": comparisons}


def aggregate_fig3(runs, pairs) -> dict[str, Any]:
    models: dict[str, dict[int, dict[int, float]]] = {}
    for candidate, by_seed in runs.items():
        model, k = candidate.rsplit("_k", 1)
        models.setdefault(model, {})[int(k)] = {
            s: m["train_mse"] for s, m in by_seed.items()
        }
    table = {}
    for model, by_k in models.items():
        # Paper Fig. 3: one box per model over the 7 datasets; we first average the
        # seeds of each dataset, then describe the 7 dataset-level values.
        dataset_means = {
            k: statistics.fmean(v.values()) for k, v in sorted(by_k.items())
        }
        table[model] = {
            "per_dataset_mean_train_mse": dataset_means,
            "per_dataset_seed_std": {
                k: statistics.stdev(v.values()) if len(v) > 1 else None
                for k, v in sorted(by_k.items())
            },
            "box_over_datasets": describe(list(dataset_means.values())),
            "n_datasets": len(dataset_means),
        }
    comparisons = {}
    for first, second in pairs:
        per_dataset = {}
        for k in sorted(set(models[first]) & set(models[second])):
            per_dataset[k] = paired_test(models[first][k], models[second][k])
        comparisons[f"{first}_vs_{second}"] = per_dataset
    return {"models": table, "paired_comparisons_per_dataset": comparisons}


def aggregate_ns(runs, pairs) -> dict[str, Any]:
    table = {}
    for candidate, by_seed in runs.items():
        row = {}
        for field in FIELDS:
            for suffix, metric_key in MAERM_VARIANTS.items():
                for statistic in ("min", "max", "mean"):
                    row[f"{field}_{statistic}{suffix}"] = describe(
                        [
                            m["maerm_table"][f"{field}_{metric_key}"][statistic]
                            for m in by_seed.values()
                        ]
                    )
        row["final_loss"] = describe([m["final_loss"] for m in by_seed.values()])
        row["final_pde_loss"] = describe(
            [m["final_pde_loss"] for m in by_seed.values()]
        )
        row["final_data_loss"] = describe(
            [m["final_data_loss"] for m in by_seed.values()]
        )
        row["seconds_per_iteration"] = describe(
            [m["seconds_per_iteration"] for m in by_seed.values()]
        )
        first = next(iter(by_seed.values()))
        row["parameters"] = first["parameters"]
        row["cost_factor"] = first["cost_factor"]
        table[candidate] = row
    comparisons = {}
    for first, second in pairs:
        for field in FIELDS:
            for suffix, metric_key in MAERM_VARIANTS.items():
                comparisons[f"{first}_vs_{second}:{field}_mean_maerm{suffix}"] = (
                    paired_test(
                        {
                            s: m["maerm_table"][f"{field}_{metric_key}"]["mean"]
                            for s, m in runs[first].items()
                        },
                        {
                            s: m["maerm_table"][f"{field}_{metric_key}"]["mean"]
                            for s, m in runs[second].items()
                        },
                    )
                )
        comparisons[f"{first}_vs_{second}:final_loss"] = paired_test(
            {s: m["final_loss"] for s, m in runs[first].items()},
            {s: m["final_loss"] for s, m in runs[second].items()},
        )
    return {"candidates": table, "paired_comparisons": comparisons}


def _fmt(stats: dict[str, Any], scale: float = 1.0, digits: int = 3) -> str:
    std = "" if stats["std"] is None else f" ± {stats['std'] * scale:.{digits}g}"
    return f"{stats['mean'] * scale:.{digits}g}{std}"


def markdown(kind: str, sweep_id: str, coverage: str, result: dict[str, Any]) -> str:
    lines = [f"# {sweep_id} ({coverage})", ""]
    if kind == "toy":
        lines += [
            "| Candidate | Params | Train MSE (mean ± std) | Dense MSE (mean ± std) | Median train MSE | n |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, row in result["candidates"].items():
            lines.append(
                f"| {name} | {row['parameters']['total']} | {_fmt(row['train_mse'])} | {_fmt(row['dense_mse'])} | {row['train_mse']['median']:.3g} | {row['train_mse']['n']} |"
            )
    elif kind == "fig3":
        lines += [
            "| Model | Mean | Median | Best | Worst | Datasets |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, row in result["models"].items():
            box = row["box_over_datasets"]
            lines.append(
                f"| {name} | {box['mean']:.3g} | {box['median']:.3g} | {box['min']:.3g} | {box['max']:.3g} | {row['n_datasets']} |"
            )
    else:
        header = (
            "| Candidate | "
            + " | ".join(f"{f} {s}" for f in FIELDS for s in ("min", "max", "mean"))
            + " | n |"
        )
        for suffix, title in (
            ("", "MAERM (%), denominator median(|y|) (primary, A10)"),
            ("_signed", "MAERM (%), denominator |median(y)| (signed variant)"),
        ):
            lines += [f"**{title}**", "", header, "|---|" + "---:|" * 10]
            for name, row in result["candidates"].items():
                cells = " | ".join(
                    _fmt(row[f"{f}_{s}{suffix}"], digits=3)
                    for f in FIELDS
                    for s in ("min", "max", "mean")
                )
                lines.append(f"| {name} | {cells} | {row['u_mean']['n']} |")
            lines.append("")
    comparisons = result.get("paired_comparisons") or {}
    if comparisons:
        lines += [
            "",
            "| Paired comparison | Mean difference | First lower (seeds) | Wilcoxon p |",
            "|---|---:|---:|---:|",
        ]
        for name, test in comparisons.items():
            p_value = test.get("wilcoxon_p_value")
            lines.append(
                f"| {name} | {test['mean_difference']:.3g} | {test['first_lower_count']}/{len(test['seeds'])} | {'n/a' if p_value is None else f'{p_value:.3g}'} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sweep-dir",
        required=True,
        type=Path,
        help="Sweep folder (relative to the paper folder).",
    )
    parser.add_argument("--kind", required=True, choices=("toy", "fig3", "ns"))
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Paired comparison 'first:second' (repeatable).",
    )
    parser.add_argument(
        "--extra-sweep-dir",
        action="append",
        default=[],
        type=Path,
        help="Additional sweep folder to merge (repeatable), e.g. a retry sweep.",
    )
    parser.add_argument(
        "--name",
        help="Output name (default: the first sweep's ID).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="Output folder (relative to the paper folder).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    sweep_dirs = [
        path if path.is_absolute() else PAPER_DIR / path
        for path in [args.sweep_dir, *args.extra_sweep_dir]
    ]
    out_dir = args.out if args.out.is_absolute() else PAPER_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, statuses = load_runs(sweep_dirs)
    pairs = [tuple(p.split(":")) for p in args.pair]
    aggregate = {"toy": aggregate_toy, "fig3": aggregate_fig3, "ns": aggregate_ns}[
        args.kind
    ]
    result = aggregate(runs, pairs)
    # Coverage is judged on the union of planned (candidate, seed) pairs, so a
    # retry sweep that completes runs missing from its parent restores coverage.
    planned = {(e["candidate"], e["seed"]) for st in statuses for e in st["runs"]}
    observed = {(c, seed) for c, by_seed in runs.items() for seed in by_seed}
    if len(statuses) == 1:
        complete_label, partial_label = "COMPLETE SWEEP", "PARTIAL SWEEP"
    else:
        complete_label = f"COMPLETE RUN COVERAGE (union of {len(statuses)} sweeps)"
        partial_label = f"PARTIAL RUN COVERAGE (union of {len(statuses)} sweeps)"
    coverage = (
        complete_label
        if observed == planned
        else f"{partial_label} ({len(observed)}/{len(planned)})"
    )
    name = args.name or statuses[0]["sweep_id"]
    payload = {
        "name": name,
        "sweeps": [
            {
                "sweep_id": st["sweep_id"],
                "status": st["status"],
                "expected_runs": st["expected_runs"],
                "completed_runs": sum(e["status"] == "COMPLETED" for e in st["runs"]),
                "code": st["code"],
            }
            for st in statuses
        ],
        "coverage": coverage,
        "completed_runs": len(observed),
        "expected_runs": len(planned),
        "missing_runs": sorted(planned - observed),
        **result,
    }
    json_path = out_dir / f"{name}_results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / f"{name}_results.md"
    md_path.write_text(markdown(args.kind, name, coverage, result), encoding="utf-8")
    logger.info("Wrote %s and %s (%s)", json_path, md_path, coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
