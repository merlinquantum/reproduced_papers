"""Phase 5 fair comparison of ABS kernels with the classical baselines.

Usage::

    python utils/compare_baselines.py --manifest results/fair_comparison_manifest.json \
        --out results/fair_comparison

The manifest lists, for every comparison condition, the ABS runs (Phase 3
reproduction), the MerLin runs (Phase 4 adaptation) and the classical baseline
runs, plus the paper-reported values. The protocol is frozen in ``LOG.md``
("Phase 5 Comparison Plan").

- Each model is aggregated as the mean over instances (U_0 or random seeds) for
  every (dataset variant, partition), then the mean over partitions per variant.
- Two models are compared only on the dataset variants they share (same data
  seed, K-means seed, noise and cluster-to-outcome assignment) and on the same
  100 partitions.
- Paired difference: per-variant difference of means with a 95 % t-interval over
  variants, or over partitions when there is a single variant (1D).
- "no clear separation" when ``|difference| <= TIE_TOLERANCE`` or the interval
  contains 0.

Writes ``<out>.json`` (every pair, full precision) and ``<out>.md`` (tables).
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
TIE_TOLERANCE = 0.02
N_PARTITIONS = 100
VARIANT_FIELDS = (
    "name", "noise", "data_seed", "kmeans_seed", "assignment", "assignment_seed",
)  # fmt: skip
C4_RANGE = (0.73, 1.0)


def _variant_key(dataset_config: dict) -> tuple:
    return tuple((field, dataset_config.get(field)) for field in VARIANT_FIELDS)


def _matches(dataset_config: dict, variant_filter: dict) -> bool:
    return all(
        dataset_config.get(key) == value for key, value in variant_filter.items()
    )


def load_run(run_dir: Path) -> dict:
    """Read a validated run; refuse incomplete or non-finite evidence."""
    status = json.loads((run_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{run_dir}: status {status['status']}")
    if not (run_dir / status["config_path"]).exists():
        raise ValueError(f"{run_dir}: missing config snapshot")
    metrics = json.loads((run_dir / status["metrics_path"]).read_text())
    started = datetime.fromisoformat(status["started_at"].replace("Z", "+00:00"))
    completed = datetime.fromisoformat(status["completed_at"].replace("Z", "+00:00"))
    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "wall_clock_seconds": (completed - started).total_seconds(),
        "validity": metrics["validity"],
    }


def role_of(label: str, spec: dict) -> str:
    kind = spec["kind"]
    if kind.startswith("abs_"):
        return "ABS, MerLin adaptation" if "merlin" in label else "ABS reproduction"
    if kind == "index_gaussian" and spec["source"] == "outcome_index":
        return "classical, information-matched"
    if kind == "random_rank_d":
        return f"classical, dimension-matched (d = {spec['dimension']})"
    return "classical, unmatched reference (more input information)"


def model_table(run: dict, arm: str, variant_filter: dict) -> dict[str, dict]:
    """Per model: {variant_key: per-partition accuracy averaged over instances}."""
    tables = {}
    for label, result in run["metrics"]["results"].items():
        if arm not in result["arms"]:
            continue
        per_variant: dict[tuple, list[np.ndarray]] = {}
        for instance in result["arms"][arm]["instances"]:
            if not _matches(instance["dataset_config"], variant_filter):
                continue
            accuracies = np.asarray(instance["test_accuracy_per_partition"], float)
            if accuracies.shape != (N_PARTITIONS,) or not np.all(
                np.isfinite(accuracies)
            ):
                raise ValueError(f"{run['run_dir']} {label}: bad partition accuracies")
            key = _variant_key(instance["dataset_config"])
            per_variant.setdefault(key, []).append(accuracies)
        if not per_variant:
            continue
        counts = {len(instances) for instances in per_variant.values()}
        if len(counts) != 1:
            raise ValueError(f"{run['run_dir']} {label}: uneven instances {counts}")
        tables[label] = {
            "spec": result["spec"],
            "role": role_of(label, result["spec"]),
            "instances_per_variant": counts.pop(),
            "partitions": {
                key: np.mean(instances, axis=0)
                for key, instances in per_variant.items()
            },
            "run": run,
        }
    return tables


def summarize_model(entry: dict) -> dict:
    variant_means = np.array([acc.mean() for acc in entry["partitions"].values()])
    if variant_means.size > 1:
        spread, spread_kind = float(variant_means.std(ddof=1)), "std over variants"
    else:
        only = next(iter(entry["partitions"].values()))
        spread = float(only.std(ddof=1) / math.sqrt(only.size))
        spread_kind = "SE over partitions"
    return {
        "mean_test_accuracy": float(variant_means.mean()),
        "variability": spread,
        "variability_kind": spread_kind,
        "n_variants": int(variant_means.size),
        "instances_per_variant": entry["instances_per_variant"],
        "role": entry["role"],
        "run": str(entry["run"]["run_dir"].relative_to(PROJECT_DIR)),
        "validity": entry["run"]["validity"],
        "wall_clock_seconds_of_run": entry["run"]["wall_clock_seconds"],
    }


def paired_difference(first: dict, second: dict) -> dict:
    """Mean of first minus second on their shared variants and partitions."""
    shared = sorted(set(first["partitions"]) & set(second["partitions"]))
    if not shared:
        raise ValueError("no shared dataset variants")
    if len(shared) > 1:
        samples = np.array(
            [
                first["partitions"][k].mean() - second["partitions"][k].mean()
                for k in shared
            ]
        )
        unit = "variants"
    else:
        samples = first["partitions"][shared[0]] - second["partitions"][shared[0]]
        unit = "partitions"
    mean = float(samples.mean())
    std = float(samples.std(ddof=1))
    half_width = float(
        stats.t.ppf(0.975, samples.size - 1) * std / math.sqrt(samples.size)
    )
    low, high = mean - half_width, mean + half_width
    if abs(mean) <= TIE_TOLERANCE or low <= 0.0 <= high:
        verdict = "no clear separation"
    else:
        verdict = "first higher" if mean > 0 else "second higher"
    return {
        "difference": mean,
        "std": std,
        "ci95": [low, high],
        "n": int(samples.size),
        "paired_over": unit,
        "n_positive": int(np.sum(samples > 0)),
        "verdict": verdict,
    }


def _dimension(label: str) -> int | None:
    if label.startswith("B2_qubit"):
        return 2
    if label.startswith("B3_qutrit"):
        return 3
    return None


def compare_condition(condition: dict) -> dict:
    runs = {
        group: [load_run(PROJECT_DIR / path) for path in condition[group]]
        for group in ("abs_runs", "merlin_runs", "baseline_runs")
    }
    variant_filter = condition.get("variant_filter", {})
    report = {"name": condition["name"], "description": condition["description"],
              "paper": condition["paper"], "variant_filter": variant_filter,
              "arms": {}}  # fmt: skip
    for arm in condition["arms"]:
        models: dict[str, dict] = {}
        for group in ("abs_runs", "merlin_runs", "baseline_runs"):
            for run in runs[group]:
                for label, entry in model_table(run, arm, variant_filter).items():
                    if label in models:
                        raise ValueError(
                            f"duplicate model {label} in {condition['name']}"
                        )
                    models[label] = entry
        quantum = [label for label, e in models.items() if e["role"].startswith("ABS")]
        classical = [label for label in models if label not in quantum]
        expected = condition["expected_variants"]
        for label, entry in models.items():
            if len(entry["partitions"]) != expected:
                raise ValueError(
                    f"{condition['name']} {label}: {len(entry['partitions'])} variants, "
                    f"expected {expected}"
                )
        pairs = [
            {"first": q, "second": c, **paired_difference(models[q], models[c])}
            for q in quantum
            for c in classical
        ]
        qutrit_minus_qubit = []
        for label in quantum:
            if _dimension(label) == 3:
                twin = label.replace("B3_qutrit", "B2_qubit").replace("fig4b", "fig4a")
                if twin in models:
                    qutrit_minus_qubit.append(
                        {"first": label, "second": twin,
                         **paired_difference(models[label], models[twin])}
                    )  # fmt: skip
        if {"random_rank3_kernel", "random_rank2_kernel"} <= set(models):
            qutrit_minus_qubit.append(
                {"first": "random_rank3_kernel", "second": "random_rank2_kernel",
                 **paired_difference(models["random_rank3_kernel"],
                                     models["random_rank2_kernel"])}
            )  # fmt: skip
        report["arms"][arm] = {
            "models": {
                label: summarize_model(entry) for label, entry in models.items()
            },
            "quantum_vs_classical": pairs,
            "dimension_effect": qutrit_minus_qubit,
        }
    return report


def compare_alternatives(entry: dict) -> dict:
    rows = {}
    for path in entry["runs"]:
        run = load_run(PROJECT_DIR / path)
        for label, result in run["metrics"]["results"].items():
            accuracies = np.array(
                [i["mean_test_accuracy"] for i in result["arms"]["paper"]["instances"]]
            )
            if not np.all(np.isfinite(accuracies)):
                raise ValueError(f"{path} {label}: non-finite accuracy")
            rows[label] = {
                "role": role_of(label, result["spec"]),
                "run": path,
                "n_kernels": int(accuracies.size),
                "mean_test_accuracy": float(accuracies.mean()),
                "std_over_kernels": float(accuracies.std(ddof=1)),
                "fraction_in_paper_range": float(
                    np.mean((accuracies >= C4_RANGE[0]) & (accuracies <= C4_RANGE[1]))
                ),
                "min": float(accuracies.min()),
                "max": float(accuracies.max()),
                "wall_clock_seconds_of_run": run["wall_clock_seconds"],
            }
    return {"paper_range": list(C4_RANGE), "paper": entry["paper"], "models": rows}


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(report: dict) -> str:
    lines = ["# Phase 5 fair comparison", "",
             "Generated by `utils/compare_baselines.py`. Paper values are reported by "
             "the paper; every other value was executed locally (CPU simulation). "
             f"Tie rule: |difference| <= {TIE_TOLERANCE} or 95 % CI contains 0.", ""]  # fmt: skip
    for condition in report["conditions"]:
        lines += [f"## {condition['name']}", "", condition["description"], "",
                  f"Paper-reported: {json.dumps(condition['paper'])}", ""]  # fmt: skip
        for arm, block in condition["arms"].items():
            lines += [f"### Arm `{arm}`", "",
                      "| System | Role | Scope | Variants | Mean test acc. | Variability "
                      "| Run | Run wall-clock (s) |",
                      "|---|---|---|---:|---:|---:|---|---:|"]  # fmt: skip
            ordered = sorted(block["models"].items(),
                             key=lambda item: -item[1]["mean_test_accuracy"])  # fmt: skip
            for label, model in ordered:
                lines.append(
                    f"| {label} | {model['role']} | {model['validity'].split(' (')[0]} | "
                    f"{model['n_variants']} | {_fmt(model['mean_test_accuracy'])} | "
                    f"{_fmt(model['variability'])} ({model['variability_kind']}) | "
                    f"{model['run']} | {model['wall_clock_seconds_of_run']:.0f} |"
                )
            lines += ["", "| ABS model | Classical model | Difference | 95 % CI | Paired over "
                      "| Verdict |", "|---|---|---:|---|---|---|"]  # fmt: skip
            for pair in block["quantum_vs_classical"]:
                # Unmatched references stay in the JSON; the table keeps the
                # information- and dimension-matched controls.
                if not pair["second"].startswith(("gaussian_outcome", "random_rank")):
                    continue
                lines.append(
                    f"| {pair['first']} | {pair['second']} | {pair['difference']:+.3f} | "
                    f"[{pair['ci95'][0]:+.3f}, {pair['ci95'][1]:+.3f}] | "
                    f"{pair['n']} {pair['paired_over']} | {pair['verdict']} |"
                )
            if block["dimension_effect"]:
                lines += ["", "Dimension effect (d = 3 minus d = 2):", "",
                          "| Higher-d model | Lower-d model | Difference | 95 % CI | Verdict |",
                          "|---|---|---:|---|---|"]  # fmt: skip
                for pair in block["dimension_effect"]:
                    lines.append(
                        f"| {pair['first']} | {pair['second']} | {pair['difference']:+.3f} | "
                        f"[{pair['ci95'][0]:+.3f}, {pair['ci95'][1]:+.3f}] | {pair['verdict']} |"
                    )
            lines.append("")
    alternatives = report["alternatives"]
    lines += ["## C4: alternative kernels (1D, paper arm)", "",
              f"Paper: {alternatives['paper']}", "",
              "| Kernel family | Role | Kernels | Mean acc. | Std | Min | Max "
              f"| In [{C4_RANGE[0]}, {C4_RANGE[1]}] | Run |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]  # fmt: skip
    for label, row in alternatives["models"].items():
        lines.append(
            f"| {label} | {row['role']} | {row['n_kernels']} | "
            f"{_fmt(row['mean_test_accuracy'])} | {_fmt(row['std_over_kernels'])} | "
            f"{_fmt(row['min'])} | {_fmt(row['max'])} | "
            f"{100 * row['fraction_in_paper_range']:.0f} % | {row['run']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="path without suffix")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = {
        "protocol": "LOG.md, Phase 5 Comparison Plan (2026-11-05T14:45:11Z)",
        "tie_tolerance": TIE_TOLERANCE,
        "conditions": [compare_condition(c) for c in manifest["conditions"]],
        "alternatives": compare_alternatives(manifest["alternatives"]),
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(render_markdown(report))
    for condition in report["conditions"]:
        paper_arm = condition["arms"]["paper"]
        print(condition["name"])
        for label, model in sorted(paper_arm["models"].items()):
            print(f"  {label:40s} {model['mean_test_accuracy']:.4f} "
                  f"± {model['variability']:.4f} ({model['n_variants']} variants)")  # fmt: skip


if __name__ == "__main__":
    main()
