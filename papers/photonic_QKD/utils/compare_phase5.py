"""Phase 5 comparison table: PQKD, fair classical baselines and the MerLin adaptation.

Reads only structured artifacts (``sweep_status.json``, ``spec.json`` and each
run's ``metrics.json``) and applies the protocol frozen in LOG.md
(``Phase 5 Comparison Plan``): final-epoch test accuracy, mean ± s.d. over
seeds 0-2, paired per-seed differences, and "differs only if |mean paired
Δ| > s.d. of the paired Δ". Coverage must match the plan exactly; a missing,
duplicate or non-finite run raises an error.

Usage (from the repository root)::

    python papers/photonic_QKD/utils/compare_phase5.py \
        --teacher-sweep <S1 dir> \
        --s4-sweeps <S4 dir> <S4b dir> \
        --s6-sweeps <S6 dir> <resume1> <resume2> <resume3>

Writes ``results/phase5_comparison.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_s4 import _format, _paper_relative, _stats
from analyze_s6 import merge_sweeps

PAPER_DIR = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
TEST = "final_evaluation.test_accuracy"
VAL = "final_evaluation.val_accuracy"
VAL_CE = "final_evaluation.val_cross_entropy"
# Paper Table 1, MNIST: validation accuracy, n = 5, 100 epochs.
PAPER_ROWS = (
    ("teacher (paper)", "paper reference", 224394, (99.07, 0.13)),
    ("PQKD (paper)", "paper method", 149439, (99.09, 0.09)),
)
# (system, role, sweep, arm, matching basis, comparison status, note)
SYSTEMS = (
    (
        "PQKD, 8 photons",
        "reproduced paper method",
        "s4",
        "pqkd",
        "reference",
        "COMMENSURABLE",
        "",
    ),
    (
        "B1 dense KD",
        "fair classical baseline (claim A)",
        "s4",
        "B1_kd_dense",
        "trainable params ±5% (−0.5%)",
        "COMMENSURABLE",
        "",
    ),
    (
        "B2 learned mixing",
        "upper reference (claim A)",
        "s4",
        "B2_dictionary_trainable",
        "same R, +33k params",
        "PARTIALLY_COMMENSURABLE",
        "not iso-parameter; upper reference only",
    ),
    (
        "B3f fixed random z",
        "fair classical baseline (claim B)",
        "s4",
        "B3f_random_fixed",
        "identical architecture, no θ",
        "COMMENSURABLE",
        "",
    ),
    (
        "B3r random z per epoch",
        "fair classical baseline (claims B, C)",
        "s4",
        "B3r_random_resampled",
        "identical architecture, no θ",
        "COMMENSURABLE",
        "",
    ),
    (
        "B4 photonic, θ frozen",
        "paper control (Supp. Note 5), reproduced",
        "s4",
        "B4_fixed_photonic",
        "identical architecture, θ not trained",
        "COMMENSURABLE",
        "",
    ),
    (
        "B5 classical generator, SPSA",
        "fair classical baseline (claim B)",
        "s4",
        "B5_surrogate",
        "dim φ = dim θ = 45, equal SPSA grid",
        "COMMENSURABLE",
        "",
    ),
    (
        "B6 classical histograms, S = 200",
        "fair classical baseline (claims B, C)",
        "s4",
        "B6_classical_histogram",
        "same Φ, S, EMA",
        "COMMENSURABLE",
        "",
    ),
    (
        "B0 z = 0",
        "paper control (Fig. 4)",
        "s4",
        "B0_zero",
        "identical architecture",
        "INCOMPARABLE",
        "degenerate: M = 0 zeroes every compressed kernel; metric agreement only",
    ),
    (
        "P0 PQKD, 4 photons",
        "MerLin adaptation reference",
        "s6",
        "P0_pqkd_theta0",
        "identical to PQKD, 4 photons",
        "COMMENSURABLE",
        "code revision differs from S4 by additive changes only",
    ),
    (
        "F1 mixing θ, S = 200, EMA",
        "MerLin adaptation (claim C)",
        "s6",
        "F1_mixing_s200_ema",
        "identical architecture",
        "COMMENSURABLE",
        "",
    ),
    (
        "F2 mixing θ, S = 200, no EMA",
        "MerLin adaptation (claim C)",
        "s6",
        "F2_mixing_s200_noema",
        "identical architecture",
        "COMMENSURABLE",
        "",
    ),
    (
        "F3 mixing θ, S = 25, EMA",
        "MerLin adaptation (claim C)",
        "s6",
        "F3_mixing_s25_ema",
        "identical architecture",
        "COMMENSURABLE",
        "",
    ),
    (
        "F4 mixing θ, S = 25, no EMA",
        "MerLin adaptation (claim C)",
        "s6",
        "F4_mixing_s25_noema",
        "identical architecture",
        "COMMENSURABLE",
        "",
    ),
    (
        "N1 exact, autograd θ, θ₀ random",
        "MerLin adaptation (claims A, B)",
        "s6",
        "N1_native_grad_random",
        "identical architecture, 45 trained angles",
        "COMMENSURABLE",
        "simulation-only gradients",
    ),
    (
        "N2 exact, autograd θ, θ₀ = 0",
        "MerLin adaptation (claim B)",
        "s6",
        "N2_native_grad_zero",
        "identical architecture, 45 trained angles",
        "COMMENSURABLE",
        "simulation-only gradients",
    ),
    (
        "N3 classical generator, autograd",
        "fair classical counterpart to N1/N2 (Phase 4 sweep)",
        "s6",
        "N3_surrogate_grad",
        "dim φ = 45, same optimiser",
        "COMMENSURABLE",
        "",
    ),
)
# (claim, arm sweep, arm, reference sweep, reference) as frozen in the plan.
PAIRS = (
    ("A", "s4", "B1_kd_dense", "s4", "pqkd"),
    ("A", "s6", "N1_native_grad_random", "s4", "B1_kd_dense"),
    ("A", "s6", "P0_pqkd_theta0", "s4", "B1_kd_dense"),
    ("B", "s4", "B3f_random_fixed", "s4", "pqkd"),
    ("B", "s4", "B4_fixed_photonic", "s4", "pqkd"),
    ("B", "s4", "B5_surrogate", "s4", "pqkd"),
    ("B", "s4", "B6_classical_histogram", "s4", "pqkd"),
    ("B", "s4", "B3r_random_resampled", "s4", "pqkd"),
    ("B", "s6", "N1_native_grad_random", "s6", "N3_surrogate_grad"),
    ("B", "s6", "N1_native_grad_random", "s4", "B3f_random_fixed"),
    ("B", "s6", "N2_native_grad_zero", "s6", "N3_surrogate_grad"),
    ("C", "s6", "F1_mixing_s200_ema", "s6", "P0_pqkd_theta0"),
    ("C", "s6", "F2_mixing_s200_noema", "s6", "P0_pqkd_theta0"),
    ("C", "s6", "F3_mixing_s25_ema", "s6", "P0_pqkd_theta0"),
    ("C", "s6", "F4_mixing_s25_noema", "s6", "P0_pqkd_theta0"),
    ("C", "s6", "F3_mixing_s25_ema", "s6", "F4_mixing_s25_noema"),
    ("C", "s6", "F1_mixing_s200_ema", "s6", "F2_mixing_s200_noema"),
    ("C", "s4", "B3r_random_resampled", "s4", "B3f_random_fixed"),
    ("check", "s6", "P0_pqkd_theta0", "s4", "pqkd"),
)


def _value(metrics: dict[str, Any], path: str) -> float:
    value: Any = metrics
    for key in path.split("."):
        value = value[key]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {path}: {value}")
    return value


def _hours(metrics: dict[str, Any]) -> float:
    # Sum of per-epoch seconds; excludes pauses and checkpoint resumes.
    return sum(epoch["seconds"] for epoch in metrics["history"]) / 3600.0


def _check_coverage(name: str, runs: dict[str, dict[int, Any]], spec: dict) -> None:
    expected = {(c["name"], int(s)) for c in spec["candidates"] for s in spec["seeds"]}
    observed = {(arm, seed) for arm, seeds in runs.items() for seed in seeds}
    if observed != expected:
        raise ValueError(
            f"{name}: coverage mismatch; missing {sorted(expected - observed)}, "
            f"unexpected {sorted(observed - expected)}"
        )


def _sweep_resources(sweep_dir: Path) -> str:
    spec = json.loads((sweep_dir / "spec.json").read_text())
    return f"{spec['max_parallel']} parallel × {spec['threads_per_run']} threads"


def build(teacher_sweep: Path, s4: list[Path], s6: list[Path]) -> dict[str, Any]:
    """Return the comparison table, paired comparisons and provenance."""
    sweeps = {}
    for name, dirs in (("s4", s4), ("s6", s6)):
        spec, runs, sources = merge_sweeps(dirs)
        _check_coverage(name, runs, spec)
        sweeps[name] = {
            "runs": runs,
            "sources": sources,
            "commits": sorted(
                {
                    json.loads((d / "sweep_status.json").read_text())["code"]["commit"][
                        :7
                    ]
                    for d in dirs
                }
            ),
            "resources": _sweep_resources(dirs[0]),
        }

    teacher_status = json.loads((teacher_sweep / "sweep_status.json").read_text())
    teacher_runs = {
        int(run["seed"]): json.loads((teacher_sweep / run["metrics_path"]).read_text())
        for run in teacher_status["runs"]
        if run["status"] == "COMPLETED"
    }
    if sorted(teacher_runs) != list(SEEDS):
        raise ValueError(f"teacher seeds {sorted(teacher_runs)} != {SEEDS}")

    rows = [
        {
            "system": name,
            "role": role,
            "provenance": "paper-reported",
            "scope": "100 epochs, 5 seeds",
            "runs": "5 / 5 (paper)",
            "metric": "val acc %",
            "mean": mean,
            "std": std,
            "test_accuracy": None,
            "val_accuracy": {"mean": mean, "std": std, "n": 5},
            "matching_basis": "parameter count inferred (paper reports C_x and ~2.25e5)",
            "trainable_parameters": parameters,
            "training_hours": None,
            "resources": "CUDA-Q ORCA simulator (paper)",
            "status": "PARTIALLY_COMMENSURABLE",
            "note": "different budget (100 epochs, n = 5); validation metric tuned by SPSA",
        }
        for name, role, parameters, (mean, std) in PAPER_ROWS
    ]
    rows.append(
        {
            "system": "teacher (S1)",
            "role": "distillation teacher, not iso-parameter",
            "provenance": "locally executed",
            "scope": "partial RED-tier run (30 epochs, 3 seeds)",
            "runs": f"{len(teacher_runs)} / 3",
            "metric": "test acc % (best-val checkpoint)",
            "test_accuracy": _stats(
                [teacher_runs[s]["best_checkpoint"]["test_accuracy"] for s in SEEDS]
            ),
            "val_accuracy": _stats(
                [teacher_runs[s]["final_epoch"]["val_accuracy"] for s in SEEDS]
            ),
            "matching_basis": "224,394 params (1.5× PQKD)",
            "trainable_parameters": teacher_runs[0]["trainable_parameters"],
            "training_hours": _stats([_hours(teacher_runs[s]) for s in SEEDS]),
            "resources": _sweep_resources(teacher_sweep) + ", CPU laptop",
            "status": "PARTIALLY_COMMENSURABLE",
            "note": "checkpoint rule differs (best-val vs final epoch); not a fair comparator for claim A",
        }
    )
    for system, role, sweep, arm, basis, status, note in SYSTEMS:
        runs = sweeps[sweep]["runs"][arm]
        # N3 runs in the Phase 4 sweep but has no photonic component.
        photonic_adaptation = sweep == "s6" and arm != "N3_surrogate_grad"
        rows.append(
            {
                "system": system,
                "role": role,
                "provenance": "locally executed"
                + (" (MerLin adaptation)" if photonic_adaptation else ""),
                "scope": "partial RED-tier run (30 epochs, 3 seeds)"
                + (", 4 photons, simulated" if photonic_adaptation else ""),
                "runs": f"{len(runs)} / 3",
                "metric": "test acc % (final epoch)",
                "test_accuracy": _stats([_value(runs[s], TEST) for s in SEEDS]),
                "val_accuracy": _stats([_value(runs[s], VAL) for s in SEEDS]),
                "matching_basis": basis,
                "trainable_parameters": int(
                    _value(runs[0], "parameters.student_total_trainable")
                ),
                "training_hours": _stats([_hours(runs[s]) for s in SEEDS]),
                "resources": sweeps[sweep]["resources"] + ", CPU laptop",
                "status": status,
                "note": note,
            }
        )

    paired = []
    for claim, arm_sweep, arm, reference_sweep, reference in PAIRS:
        arm_runs = sweeps[arm_sweep]["runs"][arm]
        reference_runs = sweeps[reference_sweep]["runs"][reference]
        differences = [
            _value(arm_runs[s], TEST) - _value(reference_runs[s], TEST) for s in SEEDS
        ]
        entry = _stats(differences)
        entry["separated"] = abs(entry["mean"]) > entry["std"]
        entry["signs"] = {
            "positive": sum(d > 0 for d in differences),
            "negative": sum(d < 0 for d in differences),
            "zero": sum(d == 0 for d in differences),
        }
        paired.append(
            {
                "claim": claim,
                "arm": arm,
                "arm_sweep": arm_sweep,
                "reference": reference,
                "reference_sweep": reference_sweep,
                "test_accuracy_difference": entry,
                "val_cross_entropy_difference": _stats(
                    [
                        _value(arm_runs[s], VAL_CE) - _value(reference_runs[s], VAL_CE)
                        for s in SEEDS
                    ]
                ),
            }
        )

    return {
        "protocol": "LOG.md 'Phase 5 Comparison Plan — 2026-10-28T21:08Z'",
        "rule": "final-epoch test accuracy, mean ± s.d. over seeds 0-2; differs only if |mean paired Δ| > s.d. of paired Δ; no winner from means alone",
        "coverage": {
            "teacher": f"{len(teacher_runs)}/3",
            "s4": f"{sum(len(v) for v in sweeps['s4']['runs'].values())}/27",
            "s6": f"{sum(len(v) for v in sweeps['s6']['runs'].values())}/24",
        },
        "sweeps": {
            "teacher": _paper_relative(teacher_sweep),
            "s4": [_paper_relative(d) for d in s4],
            "s6": [_paper_relative(d) for d in s6],
        },
        "code_revisions": {name: sweep["commits"] for name, sweep in sweeps.items()},
        "rows": rows,
        "paired": paired,
    }


def _short(arm: str) -> str:
    return arm.split("_")[0] if arm != "pqkd" else "PQKD"


def render(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        "# Phase 5 comparison: PQKD vs fair baselines and the MerLin adaptation (MNIST)",
        "",
        f"Protocol: {report['protocol']}. Rule: {report['rule']}.",
        "",
        f"Coverage: teacher {coverage['teacher']}, S4+S4b {coverage['s4']}, S6+resumes {coverage['s6']} (complete). "
        f"Code revisions: S4 {report['code_revisions']['s4']}, S6 {report['code_revisions']['s6']}.",
        "",
        "## Systems",
        "",
    ]
    header = [
        "System",
        "Role",
        "Provenance",
        "Scope",
        "Runs",
        "Metric",
        "Mean",
        "s.d.",
        "Matching basis",
        "Trainable params",
        "Train h/run",
        "Status",
    ]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in report["rows"]:
        primary = row["test_accuracy"] or row["val_accuracy"]
        hours = row["training_hours"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["system"],
                    row["role"],
                    row["provenance"],
                    row["scope"],
                    row["runs"],
                    row["metric"],
                    f"{primary['mean']:.2f}",
                    f"{primary['std']:.2f}",
                    row["matching_basis"],
                    f"{row['trainable_parameters']:,}",
                    "—" if hours is None else f"{hours['mean']:.2f}",
                    row["status"] + (f" ({row['note']})" if row["note"] else ""),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Wall-clock: sum of per-epoch seconds (pauses excluded), CPU laptop, "
        + report["rows"][3]["resources"].split(",")[0]
        + " for every local sweep.",
        "",
        "## Paired per-seed differences (final-epoch test accuracy, arm − reference)",
        "",
        "| Claim | Arm | Reference | Δ test acc (pp) | Separated? | Signs (+/−/0) | Δ val CE |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in report["paired"]:
        test = entry["test_accuracy_difference"]
        reference = _short(entry["reference"]) + (
            " (S4)" if entry["reference_sweep"] == "s4" else " (S6)"
        )
        arm = _short(entry["arm"]) + (
            " (S4)" if entry["arm_sweep"] == "s4" else " (S6)"
        )
        lines.append(
            f"| {entry['claim']} | {arm} | {reference} | {_format(test)} | "
            f"{'yes' if test['separated'] else 'no'} | "
            "{positive}/{negative}/{zero} | ".format(**test["signs"])
            + f"{_format(entry['val_cross_entropy_difference'], 4)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--teacher-sweep", type=Path, required=True)
    parser.add_argument("--s4-sweeps", type=Path, nargs="+", required=True)
    parser.add_argument("--s6-sweeps", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    report = build(
        args.teacher_sweep.resolve(),
        [path.resolve() for path in args.s4_sweeps],
        [path.resolve() for path in args.s6_sweeps],
    )
    text = render(report)
    results_dir = PAPER_DIR / "results"
    (results_dir / "phase5_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "phase5_comparison.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
