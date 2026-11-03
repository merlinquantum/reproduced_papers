#!/usr/bin/env python3
"""Generate the Phase 5 fair-comparison table from structured run artifacts.

Implements the frozen protocol in ``LOG.md`` (``## Phase 5 Comparison Plan``).
Reads only ``configs/sweeps/<id>.json`` (expected runs), ``sweep_status.json``,
``run_status.json`` and ``metrics.json``. Paper-reported values (Tables I and II)
are kept as separate ``paper`` rows and are never mixed into local statistics.

Every included run must be COMPLETED, unique per (system, seed), finite, and its
``metrics.json`` must match the SHA-256 recorded in its ``run_status.json``.

Writes ``results/p5_comparison.json`` and ``results/p5_comparison.md``.

Example (from the paper folder)::

    python utils/compare_fair.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_results import describe, paired_test  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
SWEEPS_DIR = PAPER_DIR / "outdir" / "sweeps"
SPECS_DIR = PAPER_DIR / "configs" / "sweeps"
RESULTS_DIR = PAPER_DIR / "results"
logger = logging.getLogger("compare_fair")

TIE_TOLERANCE = 0.05
SIGN_FRACTION = 0.8
ALPHA = 0.05
FIELDS = ("u", "v", "p")
K_VALUES = range(1, 8)

TOY_LOCAL = "local Apple M3 CPU, 6 workers"
NS_REMOTE = "remote gpu-reproduced-papers-1 (L4 host, 8 vCPU), 4 workers"


def system(
    system_id, experiment, sweeps, candidate, role, modality, scope, kind,
    hardware, seeds=None, matching="",
):  # fmt: skip
    """Declare one locally executed system (a candidate over one or more sweeps)."""
    return {
        "id": system_id,
        "experiment": experiment,
        "sweeps": sweeps,
        "candidate": candidate,
        "role": role,
        "modality": modality,
        "scope": scope,
        "kind": kind,
        "hardware": hardware,
        "seed_filter": seeds,
        "matching": matching,
    }


PA = "paper-accurate"
ADAPT = "MerLin adaptation (simulation)"
SYSTEMS = [
    # E2 / C1
    system("E2.tf", "E2", ["e2_fig2b"], "tf", "method", "qubit", PA, "toy", TOY_LOCAL),
    system(
        "E2.ff",
        "E2",
        ["e2_fig2b"],
        "ff",
        "paper baseline (B1)",
        "qubit",
        PA,
        "toy",
        TOY_LOCAL,
        matching="same circuit, θ_F = 1",
    ),
    system(
        "E2.ffs",
        "E2",
        ["e2_fig2b"],
        "ffs",
        "fairness control (B2)",
        "qubit",
        PA,
        "toy",
        TOY_LOCAL,
        matching="same circuit + 1 global scale",
    ),
    system(
        "E2.tf_n4",
        "E2",
        ["e2_fig2b"],
        "tf_n4",
        "method (N = 4)",
        "qubit",
        PA,
        "toy",
        TOY_LOCAL,
    ),
    system(
        "E2.ff_n4",
        "E2",
        ["e2_fig2b"],
        "ff_n4",
        "paper baseline (N = 4)",
        "qubit",
        PA,
        "toy",
        TOY_LOCAL,
        matching="same circuit, θ_F = 1",
    ),
    system(
        "E2.mlp",
        "E2",
        ["e2_fig2b"],
        "mlp",
        "classical context (B4a)",
        "classical",
        PA,
        "toy",
        TOY_LOCAL,
        matching="params ±10 % of TF",
    ),
    system(
        "E2.fourier",
        "E2",
        ["e2_fig2b"],
        "fourier",
        "classical context (B4b)",
        "classical",
        PA,
        "toy",
        TOY_LOCAL,
        matching="K = #FF gaps",
    ),
    *[
        system(
            f"E3.ff_L{depth}",
            "E3",
            ["e3_ff_depth"],
            f"ff_L{depth}",
            "capacity control (B3)",
            "qubit",
            PA,
            "toy",
            TOY_LOCAL,
            matching=f"L = {depth}",
        )
        for depth in (4, 8, 16, 32, 64, 128)
    ],
    # lr robustness (seeds 1-3; lr 1e-3 taken from the main sweeps)
    *[
        system(
            f"E2.{model}.lr1e-3",
            "E2-lr",
            ["e2_fig2b"],
            model,
            role,
            "qubit",
            "preliminary (3 seeds)",
            "toy",
            TOY_LOCAL,
            seeds=(1, 2, 3),
        )
        for model, role in (("tf", "method"), ("ff", "paper baseline (B1)"))
    ],
    *[
        system(
            f"E2.{model}.lr{lr}",
            "E2-lr",
            ["lr_robustness"],
            f"e2_{model}_lr{lr}",
            role,
            "qubit",
            "preliminary (3 seeds)",
            "toy",
            TOY_LOCAL,
        )
        for lr in ("3e-3", "1e-2")
        for model, role in (("tf", "method"), ("ff", "paper baseline (B1)"))
    ],
    # E5 / C3
    *[
        system(
            f"E5.{model}",
            "E5",
            ["e5_fig3"],
            f"{model}_k{{k}}",
            role,
            "qubit",
            PA,
            "fig3",
            TOY_LOCAL,
            matching=match,
        )
        for model, role, match in (
            ("tf", "method", ""),
            ("simple", "paper baseline (B1)", "same circuit, simple FM"),
            ("tower", "paper baseline (B1)", "same circuit, tower FM"),
            ("exponential", "paper baseline (B1)", "same circuit, exponential FM"),
            ("ffs", "fairness control (B2)", "simple FM + 1 global scale"),
        )
    ],
    *[
        system(
            f"E5b.{model}_oracle",
            "E5b",
            ["e5b_oracle_scale"],
            f"{model}_oracle_k{{k}}",
            "prior-knowledge control",
            "qubit",
            "diagnosis control",
            "fig3",
            TOY_LOCAL,
            matching="input scaled by data base frequency",
        )
        for model in ("simple", "tower", "exponential")
    ],
    *[
        system(
            f"E5.{model}.lr{lr}",
            "E5-lr",
            ["lr_robustness"],
            f"e5_{model}_k{{k}}_lr{lr}",
            "method" if model == "tf" else "paper baseline (B1)",
            "qubit",
            "preliminary (3 seeds)",
            "fig3",
            TOY_LOCAL,
        )
        for lr in ("3e-3", "1e-2")
        for model in ("tf", "simple", "tower", "exponential")
    ],
    # E6 / C4
    system("E6.tf", "E6", ["e6_table1"], "tf", "method", "qubit", PA, "ns", NS_REMOTE),
    system(
        "E6.ff",
        "E6",
        ["e6_table1"],
        "ff",
        "paper baseline (B1)",
        "qubit",
        PA,
        "ns",
        NS_REMOTE,
        matching="same circuit, θ_F = 1",
    ),
    system(
        "E6.ffs",
        "E6",
        ["e6_controls"],
        "ffs",
        "fairness control (B2)",
        "qubit",
        PA,
        "ns",
        NS_REMOTE,
        matching="same circuit + 3 scales / QNN",
    ),
    system(
        "E6.tf_minmax",
        "E6",
        ["e6_minmax"],
        "tf_minmax",
        "method, min–max inputs",
        "qubit",
        PA,
        "ns",
        NS_REMOTE,
    ),
    system(
        "E6.ff_minmax",
        "E6",
        ["e6_minmax"],
        "ff_minmax",
        "paper baseline, min–max inputs",
        "qubit",
        PA,
        "ns",
        NS_REMOTE,
        matching="same circuit, θ_F = 1",
    ),
    system(
        "E6.mlp",
        "E6",
        ["e6_controls"],
        "mlp_lr1e-2",
        "classical context (B4, selected lr)",
        "classical",
        PA,
        "ns",
        NS_REMOTE,
        matching="≈ same params, not circuit-matched",
    ),
    # E7 / C5
    system(
        "E7.tf",
        "E7",
        ["e7_appc", "e7_appc_retry"],
        "tf",
        "method",
        "qubit",
        PA + " (union of 2 sweeps)",
        "ns",
        NS_REMOTE,
    ),
    system(
        "E7.ff",
        "E7",
        ["e7_appc"],
        "ff",
        "paper baseline (B1)",
        "qubit",
        PA,
        "ns",
        NS_REMOTE,
        matching="same circuit, θ_F = 1",
    ),
    system(
        "E7.ffs",
        "E7",
        ["e7_scaling_controls"],
        "ffs",
        "fairness control (B2)",
        "qubit",
        PA,
        "ns",
        NS_REMOTE,
        matching="same circuit + 3 scales / QNN",
    ),
    system(
        "E7.tf_minmax",
        "E7",
        ["e7_scaling_controls"],
        "tf_minmax",
        "method, min–max inputs",
        "qubit",
        PA,
        "ns",
        NS_REMOTE,
    ),
    system(
        "E7.ff_minmax",
        "E7",
        ["e7_scaling_controls"],
        "ff_minmax",
        "paper baseline, min–max inputs",
        "qubit",
        PA,
        "ns",
        NS_REMOTE,
        matching="same circuit, θ_F = 1",
    ),
    # MerLin adaptation
    system(
        "M2.tf",
        "M2",
        ["m2_fig2b"],
        "tf",
        "photonic method",
        "merlin",
        ADAPT,
        "toy",
        TOY_LOCAL,
    ),
    system(
        "M2.ff",
        "M2",
        ["m2_fig2b"],
        "ff",
        "photonic paper-baseline analogue",
        "merlin",
        ADAPT,
        "toy",
        TOY_LOCAL,
        matching="same circuit, w = 1",
    ),
    system(
        "M2.ffs",
        "M2",
        ["m2_fig2b"],
        "ffs",
        "photonic fairness control",
        "merlin",
        ADAPT,
        "toy",
        TOY_LOCAL,
        matching="same circuit + 1 global scale",
    ),
    *[
        system(
            f"M5.{model}",
            "M5",
            ["m5_fig3"],
            f"{model}_k{{k}}",
            "photonic method" if model == "tf" else "photonic FF analogue",
            "merlin",
            ADAPT,
            "fig3",
            TOY_LOCAL,
        )
        for model in ("tf", "simple", "tower", "exponential", "ffs")
    ],
    system(
        "M6.tf",
        "M6",
        ["m6_table1_tf"],
        "tf",
        "photonic method",
        "merlin",
        ADAPT,
        "ns",
        NS_REMOTE,
    ),
    system(
        "M6.ff",
        "M6",
        ["m6_table1_ff"],
        "ff",
        "photonic paper-baseline analogue",
        "merlin",
        ADAPT,
        "ns",
        NS_REMOTE,
        matching="same circuit, w = 1",
    ),
    system(
        "M6.ffs",
        "M6",
        ["m6_table1_ffs"],
        "ffs",
        "photonic fairness control",
        "merlin",
        ADAPT,
        "ns",
        NS_REMOTE,
        matching="same circuit + 3 scales / network",
    ),
]

# Paper Tables I and II, mean column (mean ± std over the paper's 10 runs), MAERM %.
PAPER_ROWS = {
    "E6.paper.ff": {
        "experiment": "E6",
        "table": "Table I",
        "u": (17.5, 0.3),
        "v": (103.2, 1.4),
        "p": (73.2, 1.9),
    },
    "E6.paper.tf": {
        "experiment": "E6",
        "table": "Table I",
        "u": (13.2, 2.0),
        "v": (30.4, 2.7),
        "p": (57.0, 3.7),
    },
    "E7.paper.ff": {
        "experiment": "E7",
        "table": "Table II",
        "u": (7.7, 0.3),
        "v": (26.0, 0.5),
        "p": (45.2, 3.3),
    },
    "E7.paper.tf": {
        "experiment": "E7",
        "table": "Table II",
        "u": (6.2, 0.4),
        "v": (18.9, 0.7),
        "p": (36.8, 2.2),
    },
}

C = "COMMENSURABLE"
PC = "PARTIALLY_COMMENSURABLE"
NS_METRICS = [f"{field}_mean_maerm" for field in FIELDS]


def comparison(claim, first, second, metrics, mode, status, note=""):
    """Declare one pairwise comparison; ``first`` is the method side (A)."""
    return {
        "claim": claim,
        "a": first,
        "b": second,
        "metrics": metrics,
        "mode": mode,
        "commensurability": status,
        "note": note,
    }


COMPARISONS = [
    comparison("C1", "E2.tf", "E2.ff", ["train_mse"], "paired", C),
    comparison("C1", "E2.tf", "E2.ffs", ["train_mse"], "paired", C),
    comparison("C1", "E2.tf_n4", "E2.ff_n4", ["train_mse"], "paired", C),
    *[
        comparison(
            "C1",
            "E2.tf",
            f"E3.ff_L{depth}",
            ["train_mse"],
            "unpaired",
            C,
            "capacity control; FF has up to 32× the parameters",
        )
        for depth in (4, 8, 16, 32, 64, 128)
    ],
    comparison(
        "C1",
        "E2.tf",
        "E2.mlp",
        ["train_mse"],
        "paired",
        PC,
        "classical context; not circuit-matched",
    ),
    comparison(
        "C1",
        "E2.tf",
        "E2.fourier",
        ["train_mse"],
        "paired",
        PC,
        "classical context; trainable-frequency Fourier regressor",
    ),
    *[
        comparison(
            "C7",
            f"E2.tf.lr{lr}",
            f"E2.ff.lr{lr}",
            ["train_mse"],
            "paired",
            C,
            "lr robustness, 3 seeds",
        )
        for lr in ("1e-3", "3e-3", "1e-2")
    ],
    *[
        comparison(
            "C3",
            "E5.tf",
            f"E5.{model}",
            ["box_mean", "box_worst", "box_best"],
            "box",
            C,
        )
        for model in ("simple", "tower", "exponential", "ffs")
    ],
    *[
        comparison(
            "C3",
            "E5.tf",
            f"E5b.{model}_oracle",
            ["box_mean", "box_worst"],
            "box",
            PC,
            "FF given the data base frequency (prior knowledge TF does not get)",
        )
        for model in ("simple", "tower", "exponential")
    ],
    *[
        comparison(
            "C7",
            f"E5.tf.lr{lr}",
            f"E5.{model}.lr{lr}",
            ["box_mean", "box_worst"],
            "box",
            C,
            "lr robustness, 3 seeds",
        )
        for lr in ("3e-3", "1e-2")
        for model in ("simple", "tower", "exponential")
    ],
    comparison("C4", "E6.tf", "E6.ff", NS_METRICS, "paired", C),
    comparison(
        "C4",
        "E6.tf",
        "E6.ffs",
        NS_METRICS,
        "paired",
        C,
        "separate launch, same machine and code state",
    ),
    comparison(
        "C4",
        "E6.tf_minmax",
        "E6.ff_minmax",
        NS_METRICS,
        "paired",
        C,
        "both with min–max inputs (A7)",
    ),
    comparison(
        "C4",
        "E6.tf",
        "E6.mlp",
        NS_METRICS,
        "paired",
        PC,
        "classical context; not circuit-matched",
    ),
    comparison("C5", "E7.tf", "E7.ff", NS_METRICS, "paired", C),
    comparison(
        "C5",
        "E7.tf",
        "E7.ffs",
        NS_METRICS,
        "paired",
        C,
        "code states S1/S2 shown equivalent",
    ),
    comparison(
        "C5",
        "E7.tf",
        "E7.ff_minmax",
        NS_METRICS,
        "paired",
        PC,
        "preprocessing differs by design: tests whether normalised FF closes the gap",
    ),
    comparison(
        "C5",
        "E7.tf_minmax",
        "E7.ff_minmax",
        NS_METRICS,
        "paired",
        C,
        "both with min–max inputs",
    ),
    comparison(
        "MerLin", "M2.tf", "M2.ff", ["train_mse"], "paired", C, "adaptation scope"
    ),
    comparison(
        "MerLin", "M2.tf", "M2.ffs", ["train_mse"], "paired", C, "adaptation scope"
    ),
    *[
        comparison(
            "MerLin",
            "M5.tf",
            f"M5.{model}",
            ["box_mean", "box_worst", "box_best"],
            "box",
            C,
            "adaptation scope",
        )
        for model in ("simple", "tower", "exponential", "ffs")
    ],
    comparison("MerLin", "M6.tf", "M6.ff", NS_METRICS, "paired", C, "adaptation scope"),
    comparison(
        "MerLin", "M6.tf", "M6.ffs", NS_METRICS, "paired", C, "adaptation scope"
    ),
]

# Side-by-side only: different model class, parameter count and initialisation.
NOT_RANKED = [
    (
        "M2.tf",
        "E2.tf",
        "photonic vs qubit TF: different model class; reported side by side, not ranked",
    ),
    (
        "M6.tf",
        "E6.tf",
        "photonic vs qubit TF: different model class; reported side by side, not ranked",
    ),
    (
        "E6.tf",
        "E6.paper.tf",
        "local vs paper: unstated normalisation and batch composition (A7, A8)",
    ),
    ("E7.ff", "E7.paper.ff", "local vs paper: FF input-scale plateau (G1)"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_seeds(sweep: str, candidate: str) -> set[int]:
    spec = load_json(SPECS_DIR / f"{sweep}.json")
    for entry in spec["candidates"]:
        if entry["name"] == candidate:
            return set(entry.get("seeds") or spec["seeds"])
    return set()


def load_candidate(sweeps: list[str], candidate: str, seed_filter) -> dict[str, Any]:
    """Return per-seed metrics plus run accounting for one candidate.

    Raises
    ------
    ValueError
        On a duplicate completed (candidate, seed), a stale metrics file, or a
        candidate absent from every listed sweep spec.
    """
    expected: set[int] = set()
    by_seed: dict[int, dict[str, Any]] = {}
    for sweep in sweeps:
        expected |= expected_seeds(sweep, candidate)
        status = load_json(SWEEPS_DIR / sweep / "sweep_status.json")
        for entry in status["runs"]:
            if entry["candidate"] != candidate or entry["status"] != "COMPLETED":
                continue
            seed = entry["seed"]
            if seed_filter is not None and seed not in seed_filter:
                continue
            if seed in by_seed:
                raise ValueError(
                    f"Duplicate completed run: {sweep} {candidate} seed {seed}"
                )
            metrics_path = SWEEPS_DIR / sweep / entry["metrics_path"]
            run_status = load_json(metrics_path.parent / "run_status.json")
            if run_status["status"] != "COMPLETED":
                raise ValueError(f"run_status not COMPLETED: {metrics_path.parent}")
            digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
            if digest != run_status["metrics_sha256"]:
                raise ValueError(f"Stale metrics (SHA-256 mismatch): {metrics_path}")
            by_seed[seed] = load_json(metrics_path)
    if not expected:
        raise ValueError(f"Candidate {candidate} not in specs {sweeps}")
    if seed_filter is not None:
        expected &= set(seed_filter)
    return {"by_seed": by_seed, "expected": expected}


def parameter_summary(parameters: dict[str, Any]) -> str:
    if "total" in parameters:
        return f"{parameters['total']} (freq {parameters['frequency']})"
    per_network = [value["total"] for value in parameters.values()]
    frequency = [value["frequency"] for value in parameters.values()]
    return f"{per_network[0]} / network × {len(per_network)} (freq {frequency[0]})"


def per_seed_values(spec: dict[str, Any], runs: dict[str, Any]) -> dict[str, dict]:
    """Return ``{metric: values}``; toy/ns values are ``{seed: x}``, fig3 ``{k: {seed: x}}``."""
    if spec["kind"] == "toy":
        return {"train_mse": {s: m["train_mse"] for s, m in runs["by_seed"].items()}}
    if spec["kind"] == "ns":
        return {
            f"{field}_mean_maerm": {
                s: m["maerm_table"][f"{field}_maerm_abs_median"]["mean"]
                for s, m in runs["by_seed"].items()
            }
            for field in FIELDS
        }
    return {"per_dataset": runs["per_dataset"]}


def evaluate_system(spec: dict[str, Any]) -> dict[str, Any]:
    if spec["kind"] == "fig3":
        per_dataset: dict[int, dict[int, float]] = {}
        all_metrics, expected_total, observed_total = [], 0, 0
        for k in K_VALUES:
            runs = load_candidate(
                spec["sweeps"], spec["candidate"].format(k=k), spec["seed_filter"]
            )
            expected_total += len(runs["expected"])
            observed_total += len(set(runs["by_seed"]) & runs["expected"])
            per_dataset[k] = {s: m["train_mse"] for s, m in runs["by_seed"].items()}
            all_metrics.extend(runs["by_seed"].values())
        dataset_means = [statistics.fmean(v.values()) for v in per_dataset.values()]
        box = describe(dataset_means)
        values = {"per_dataset": per_dataset}
        stats = {
            "box_mean": box["mean"],
            "box_worst": box["max"],
            "box_best": box["min"],
            "box_median": box["median"],
            "per_dataset_mean": dict(zip(K_VALUES, dataset_means)),
        }
        metrics_list = all_metrics
    else:
        runs = load_candidate(spec["sweeps"], spec["candidate"], spec["seed_filter"])
        expected_total = len(runs["expected"])
        observed_total = len(set(runs["by_seed"]) & runs["expected"])
        values = per_seed_values(spec, runs)
        stats = {metric: describe(list(v.values())) for metric, v in values.items()}
        metrics_list = list(runs["by_seed"].values())
    first = metrics_list[0]
    return {
        **{
            key: spec[key]
            for key in (
                "id",
                "experiment",
                "role",
                "modality",
                "scope",
                "hardware",
                "matching",
                "sweeps",
            )
        },
        "provenance": "local",
        "successful_runs": observed_total,
        "expected_runs": expected_total,
        "coverage": "COMPLETE" if observed_total == expected_total else "PARTIAL",
        "stats": stats,
        "values": values,
        "parameters": parameter_summary(first["parameters"]),
        "cost_factor": first.get("cost_factor"),
        "seconds_per_iteration": describe(
            [m["seconds_per_iteration"] for m in metrics_list]
        ),
        "training_seconds": describe([m["training_seconds"] for m in metrics_list]),
    }


def relative(first: float, second: float) -> float:
    return (first - second) / second


def decide_paired(first: dict[int, float], second: dict[int, float]) -> dict[str, Any]:
    """Apply the frozen paired decision rule (LOG.md, Phase 5 Comparison Plan)."""
    test = paired_test(first, second)
    seeds = test["seeds"]
    n = len(seeds)
    mean_first = statistics.fmean(first[s] for s in seeds)
    mean_second = statistics.fmean(second[s] for s in seeds)
    r = relative(mean_first, mean_second)
    first_lower = test["first_lower_count"]
    second_lower = sum(first[s] > second[s] for s in seeds)
    p_value = test.get("wilcoxon_p_value")
    differences = [first[s] - second[s] for s in seeds]
    difference_std_rel = statistics.stdev(differences) / mean_second if n > 1 else None
    decision = "NO_SEPARATION"
    for side, lower_count, sign in (("A", first_lower, -1), ("B", second_lower, 1)):
        if sign * r <= TIE_TOLERANCE:
            continue
        if (
            n >= 6
            and lower_count >= SIGN_FRACTION * n
            and p_value is not None
            and p_value < ALPHA
        ):
            decision = f"{side}_LOWER"
        elif n == 5 and lower_count == 5:
            decision = f"{side}_LOWER_CONSISTENT"
    return {
        "n_paired": n,
        "mean_a": mean_first,
        "mean_b": mean_second,
        "relative_difference": r,
        "paired_difference_std_rel": difference_std_rel,
        "a_lower_seeds": first_lower,
        "wilcoxon_p": p_value,
        "decision": decision,
    }


def decide_unpaired(
    first: dict[int, float], second: dict[int, float]
) -> dict[str, Any]:
    mean_first, mean_second = (
        statistics.fmean(first.values()),
        statistics.fmean(second.values()),
    )
    r = relative(mean_first, mean_second)
    decision = "NO_SEPARATION"
    if r < -TIE_TOLERANCE and max(first.values()) < min(second.values()):
        decision = "A_LOWER"
    elif r > TIE_TOLERANCE and max(second.values()) < min(first.values()):
        decision = "B_LOWER"
    return {
        "n_a": len(first),
        "n_b": len(second),
        "mean_a": mean_first,
        "mean_b": mean_second,
        "relative_difference": r,
        "decision": decision,
    }


def decide_box(
    first: dict[str, Any], second: dict[str, Any], metric: str
) -> dict[str, Any]:
    value_first, value_second = first["stats"][metric], second["stats"][metric]
    if value_second == 0:
        raise ValueError("Zero baseline value in box comparison.")
    r = relative(value_first, value_second)
    decision = "TIE"
    if r < -TIE_TOLERANCE:
        decision = "A_LOWER"
    elif r > TIE_TOLERANCE:
        decision = "B_LOWER"
    signs = {}
    for k in K_VALUES:
        a_k, b_k = first["values"]["per_dataset"][k], second["values"]["per_dataset"][k]
        common = sorted(set(a_k) & set(b_k))
        signs[k] = f"{sum(a_k[s] < b_k[s] for s in common)}/{len(common)}"
    return {
        "mean_a": value_first,
        "mean_b": value_second,
        "relative_difference": r,
        "a_lower_seeds_per_dataset": signs,
        "decision": decision,
    }


def run_comparisons(systems: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in COMPARISONS:
        first, second = systems[spec["a"]], systems[spec["b"]]
        for metric in spec["metrics"]:
            if spec["mode"] == "box":
                result = decide_box(first, second, metric)
            elif spec["mode"] == "paired":
                result = decide_paired(
                    first["values"][metric], second["values"][metric]
                )
            else:
                result = decide_unpaired(
                    first["values"][metric], second["values"][metric]
                )
            rows.append(
                {
                    **{
                        k: spec[k]
                        for k in ("claim", "a", "b", "mode", "commensurability", "note")
                    },
                    "metric": metric,
                    **result,
                }
            )
    return rows


def cost_rows(systems: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """TF / FF seconds-per-iteration ratio within one launch context (claim C6)."""
    rows = []
    for first, second, context in (
        ("E6.tf", "E6.ff", "same launch, 4 workers"),
        ("E7.tf", "E7.ff", "TF partly from the retry launch (different contention)"),
        ("M6.tf", "M6.ff", "separate sequential launches, 4 workers each"),
        ("E2.tf", "E2.ff", "same sweep, 6 local workers"),
    ):
        tf, ff = systems[first], systems[second]
        rows.append(
            {
                "a": first,
                "b": second,
                "a_s_per_iter": tf["seconds_per_iteration"]["mean"],
                "b_s_per_iter": ff["seconds_per_iteration"]["mean"],
                "ratio": tf["seconds_per_iteration"]["mean"]
                / ff["seconds_per_iteration"]["mean"],
                "cost_factor_c_f": tf["cost_factor"],
                "context": context,
            }
        )
    return rows


def _g(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}g}"


def metric_cell(system_row: dict[str, Any]) -> tuple[str, str, str]:
    stats = system_row["stats"]
    if "box_mean" in stats:
        return (
            "box mean / worst train MSE",
            f"{_g(stats['box_mean'])} / {_g(stats['box_worst'])}",
            "7 datasets",
        )
    if "train_mse" in stats:
        s = stats["train_mse"]
        return ("train MSE", _g(s["mean"]), f"± {_g(s['std'], 2)}")
    means = " / ".join(_g(stats[m]["mean"]) for m in NS_METRICS)
    stds = " / ".join(_g(stats[m]["std"], 2) for m in NS_METRICS)
    return ("mean MAERM % u / v / p", means, f"± {stds}")


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Phase 5 fair comparison (generated by `utils/compare_fair.py`)",
        "",
        "Protocol: `LOG.md` → `## Phase 5 Comparison Plan — 2026-10-28T10:15:12Z`. Tie tolerance 5 % relative; "
        "paired rule `A_LOWER` needs ≥ 80 % paired seeds and Wilcoxon p < 0.05 (n ≥ 6); n = 5 with 5/5 → "
        "`_CONSISTENT` (minimum p = 0.0625); A = method side. Lower is better for every metric. "
        "MerLin rows are an adaptation (simulation), not the reproduction.",
        "",
        "## Systems (locally executed)",
        "",
        "| System | Role | Provenance | Scope | Successful / expected runs | Metric | Value | Variability | Matching basis | Parameters | s / iter (mean) | Wall-clock train s (mean) | Hardware | Status |",
        "|---|---|---|---|---:|---|---:|---:|---|---|---:|---:|---|---|",
    ]
    for row in result["systems"].values():
        name, value, variability = metric_cell(row)
        lines.append(
            f"| {row['id']} | {row['role']} | {row['provenance']} ({row['modality']}) | {row['scope']} | "
            f"{row['successful_runs']} / {row['expected_runs']} | {name} | {value} | {variability} | "
            f"{row['matching'] or '—'} | {row['parameters']} | {_g(row['seconds_per_iteration']['mean'])} | "
            f"{_g(row['training_seconds']['mean'])} | {row['hardware']} | {row['coverage']} |"
        )
    lines += [
        "",
        "## Paper-reported values (provenance: paper; not locally executed)",
        "",
        "| Row | Source | u | v | p |",
        "|---|---|---:|---:|---:|",
    ]
    for key, row in PAPER_ROWS.items():
        cells = " | ".join(f"{row[f][0]} ± {row[f][1]}" for f in FIELDS)
        lines.append(
            f"| {key} | paper {row['table']} (mean MAERM %, 10 runs) | {cells} |"
        )
    lines += [
        "",
        "## Pairwise comparisons",
        "",
        "| Claim | A | B | Metric | Mode | A value | B value | r = (A−B)/B | A lower (seeds) | Wilcoxon p | Decision | Commensurability | Note |",
        "|---|---|---|---|---|---:|---:|---:|---|---:|---|---|---|",
    ]
    for row in result["comparisons"]:
        if row["mode"] == "box":
            signs = ", ".join(
                f"k{k}:{v}" for k, v in row["a_lower_seeds_per_dataset"].items()
            )
        elif row["mode"] == "paired":
            signs = f"{row['a_lower_seeds']}/{row['n_paired']}"
        else:
            signs = f"unpaired (n {row['n_a']} vs {row['n_b']})"
        lines.append(
            f"| {row['claim']} | {row['a']} | {row['b']} | {row['metric']} | {row['mode']} | {_g(row['mean_a'])} | "
            f"{_g(row['mean_b'])} | {row['relative_difference']:+.1%} | {signs} | {_g(row.get('wilcoxon_p'), 2)} | "
            f"{row['decision']} | {row['commensurability']} | {row['note'] or '—'} |"
        )
    lines += [
        "",
        "## Training cost (claim C6)",
        "",
        "| A | B | A s/iter | B s/iter | A / B | C_f (θ_F + θ_A) / θ_A | Context |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in result["cost"]:
        lines.append(
            f"| {row['a']} | {row['b']} | {_g(row['a_s_per_iter'])} | {_g(row['b_s_per_iter'])} | "
            f"{row['ratio']:.3f} | {_g(row['cost_factor_c_f'], 4)} | {row['context']} |"
        )
    lines += [
        "",
        "## Side-by-side only (not ranked)",
        "",
        "| A | B | Reason |",
        "|---|---|---|",
    ]
    for first, second, reason in NOT_RANKED:
        lines.append(f"| {first} | {second} | {reason} |")
    lines += [
        "",
        f"Run accounting: {result['run_accounting']['successful']} / {result['run_accounting']['expected']} "
        f"expected runs included across {len(result['systems'])} systems; every system {result['run_accounting']['coverage']}.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    systems = {}
    for spec in SYSTEMS:
        systems[spec["id"]] = evaluate_system(spec)
        logger.info(
            "system %s: %d / %d runs",
            spec["id"],
            systems[spec["id"]]["successful_runs"],
            systems[spec["id"]]["expected_runs"],
        )
    comparisons = run_comparisons(systems)
    successful = sum(s["successful_runs"] for s in systems.values())
    expected = sum(s["expected_runs"] for s in systems.values())
    result = {
        "protocol": "LOG.md#Phase 5 Comparison Plan — 2026-10-28T10:15:12Z",
        "tie_tolerance_rel": TIE_TOLERANCE,
        "systems": {
            key: {k: v for k, v in row.items() if k != "values"}
            for key, row in systems.items()
        },
        "paper_rows": PAPER_ROWS,
        "comparisons": comparisons,
        "cost": cost_rows(systems),
        "not_ranked": [list(row) for row in NOT_RANKED],
        "run_accounting": {
            "successful": successful,
            "expected": expected,
            "coverage": "COMPLETE"
            if all(s["coverage"] == "COMPLETE" for s in systems.values())
            else "PARTIAL",
        },
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "p5_comparison.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8"
    )
    (RESULTS_DIR / "p5_comparison.md").write_text(
        markdown({**result, "systems": systems}), encoding="utf-8"
    )
    logger.info(
        "wrote results/p5_comparison.{json,md}; runs %d / %d (%s)",
        successful,
        expected,
        result["run_accounting"]["coverage"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
