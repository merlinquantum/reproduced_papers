"""Analyse the S6 Phase 4 sweep (4 photons): photonic regimes vs PQKD and classical arms.

Reads only structured artifacts (``sweep_status.json`` and each run's
``metrics.json``). The comparison rule is the one predeclared in LOG.md
(Phase 4 Adaptation Plan and its 4-photon amendment): an arm differs from a
reference only when the mean paired per-seed difference exceeds its standard
deviation across seeds. No winner is declared.

Paired references:

- ``P0_pqkd_theta0``: paper-faithful PQKD at 4 photons (same sweep);
- ``N3_surrogate_grad`` for N1 and N2: the C9 photonic-vs-classical pair;
- S4 ``B3f_random_fixed`` and ``B1_kd_dense``: photon-independent classical
  baselines with the same teachers and seeds;
- S4 ``pqkd``: the 8-photon Phase 3 PQKD, only as context.

Usage (from the repository root)::

    python papers/photonic_QKD/utils/analyze_s6.py <S6 sweep dir> [<resume sweep dir> ...] \
        --s4-sweeps <S4 sweep dir> <S4b sweep dir>

Writes ``results/s6_mnist_4ph_<scope>.json`` and ``.md`` (``scope`` is
``complete`` or ``partial``) and a copy in the last sweep directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_s4 import _dotted, _format, _paper_relative, _stats, load_runs

PAPER_DIR = Path(__file__).resolve().parents[1]
ARM_METRICS = {
    "test_accuracy": "final_evaluation.test_accuracy",
    "val_accuracy": "final_evaluation.val_accuracy",
    "val_cross_entropy": "final_evaluation.val_cross_entropy",
    "train_accuracy": "final_epoch.train_accuracy",
    "delta_vs_zero_feature": "final_evaluation.delta_test_accuracy_vs_zero_feature",
    "test_accuracy_fresh_feature": "final_evaluation.test_accuracy_fresh_feature",
    "theta_norm": "final_epoch.parameter_norm",
    "proxy_delta_first_minus_final": "proxy_objective_delta_first_minus_final",
    "trainable_parameters": "parameters.student_total_trainable",
    "conditioning_trainable": "parameters.conditioning_trainable",
}
PAIRED_METRICS = ("test_accuracy", "val_accuracy", "val_cross_entropy")
# (arm, reference, reference source) in the order the plan lists the questions.
PAIRINGS = (
    *(
        (arm, "P0_pqkd_theta0", "s6")
        for arm in (
            "F1_mixing_s200_ema",
            "F2_mixing_s200_noema",
            "F3_mixing_s25_ema",
            "F4_mixing_s25_noema",
            "N1_native_grad_random",
            "N2_native_grad_zero",
            "N3_surrogate_grad",
        )
    ),
    ("N1_native_grad_random", "N3_surrogate_grad", "s6"),
    ("N2_native_grad_zero", "N3_surrogate_grad", "s6"),
    ("N1_native_grad_random", "N2_native_grad_zero", "s6"),
    ("F1_mixing_s200_ema", "F2_mixing_s200_noema", "s6"),
    ("F3_mixing_s25_ema", "F4_mixing_s25_noema", "s6"),
    ("F1_mixing_s200_ema", "F3_mixing_s25_ema", "s6"),
    ("P0_pqkd_theta0", "B3f_random_fixed", "s4"),
    ("P0_pqkd_theta0", "B1_kd_dense", "s4"),
    ("N1_native_grad_random", "B3f_random_fixed", "s4"),
    ("N1_native_grad_random", "B1_kd_dense", "s4"),
    ("N3_surrogate_grad", "B3f_random_fixed", "s4"),
    ("N3_surrogate_grad", "B1_kd_dense", "s4"),
    ("P0_pqkd_theta0", "pqkd", "s4"),
)


def merge_sweeps(
    sweep_dirs: list[Path],
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]], dict[str, str]]:
    """Merge completed runs of a planned sweep and its completion sweeps.

    The first directory holds the planned spec. A (candidate, seed) pair
    completed in more than one sweep is an error, never silently de-duplicated.
    """
    spec = json.loads((sweep_dirs[0] / "spec.json").read_text())
    runs: dict[str, dict[int, dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for sweep_dir in sweep_dirs:
        _, sweep_runs = load_runs(sweep_dir)
        for arm, seeds in sweep_runs.items():
            for seed, metrics in seeds.items():
                label = f"{arm}__seed{seed}"
                if label in sources:
                    raise ValueError(
                        f"{label} completed in both {sources[label]} and {sweep_dir.name}"
                    )
                sources[label] = sweep_dir.name
                runs.setdefault(arm, {})[seed] = metrics
    return spec, runs, sources


def _training_hours(metrics: dict[str, Any]) -> float:
    # History survives checkpoint resumes, so this is the full 30-epoch compute
    # time of one run (3 runs in parallel x 4 threads), excluding pauses.
    return sum(epoch["seconds"] for epoch in metrics["history"]) / 3600.0


def _arm_statistics(seeds: dict[int, dict[str, Any]]) -> dict[str, Any]:
    entry: dict[str, Any] = {"seeds": sorted(seeds)}
    for name, path in ARM_METRICS.items():
        entry[name] = _stats(
            [
                float(value)
                for seed in sorted(seeds)
                if isinstance(value := _dotted(seeds[seed], path), (int, float))
                and math.isfinite(value)
            ]
        )
    entry["training_hours"] = _stats(
        [_training_hours(seeds[seed]) for seed in sorted(seeds)]
    )
    return entry


def _paired(
    runs: dict[int, dict[str, Any]], reference: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    seeds = sorted(set(runs) & set(reference))
    result: dict[str, Any] = {"seeds": seeds}
    for name in PAIRED_METRICS:
        path = ARM_METRICS[name]
        differences = [
            float(_dotted(runs[s], path)) - float(_dotted(reference[s], path))
            for s in seeds
        ]
        entry = _stats(differences)
        if entry is not None:
            entry["exceeds_seed_std"] = (
                entry["std"] is not None and abs(entry["mean"]) > entry["std"]
            )
            entry["sign_counts"] = {
                "positive": sum(d > 0 for d in differences),
                "negative": sum(d < 0 for d in differences),
                "zero": sum(d == 0 for d in differences),
            }
        result[name] = entry
    return result


def analyse(sweep_dirs: list[Path], s4_sweep_dirs: list[Path]) -> dict[str, Any]:
    """Return per-arm statistics and the predeclared paired comparisons."""
    spec, runs, sources = merge_sweeps(sweep_dirs)
    _, s4_runs, _ = merge_sweeps(s4_sweep_dirs)
    arm_order = [candidate["name"] for candidate in spec["candidates"]]
    expected = {(arm, int(seed)) for arm in arm_order for seed in spec["seeds"]}
    observed = {(arm, seed) for arm, seeds in runs.items() for seed in seeds}
    complete = observed == expected

    references = {"s6": runs, "s4": s4_runs}
    paired = [
        {
            "arm": arm,
            "reference": reference,
            "reference_source": source,
            **_paired(runs.get(arm, {}), references[source].get(reference, {})),
        }
        for arm, reference, source in PAIRINGS
    ]
    s4_context = {
        arm: _arm_statistics(s4_runs[arm])
        for arm in ("pqkd", "B3f_random_fixed", "B1_kd_dense")
    }
    return {
        "sweep_id": spec["sweep_id"],
        "sweep_dirs": [_paper_relative(path) for path in sweep_dirs],
        "s4_sweep_dirs": [_paper_relative(path) for path in s4_sweep_dirs],
        "run_sources": sources,
        "scope": "complete" if complete else "partial",
        "label": "Phase 4 reduced-scope validation (MNIST, 4 photons, 30 epochs, 3 seeds)"
        + ("" if complete else "; INCOMPLETE COVERAGE"),
        "coverage": {
            "observed_runs": len(observed),
            "expected_runs": len(expected),
            "missing": sorted(f"{a}__seed{s}" for a, s in expected - observed),
        },
        "comparison_rule": "no winner declared; arm differs from a reference only if |mean paired difference| > its s.d. across seeds",
        "arms": {arm: _arm_statistics(runs.get(arm, {})) for arm in arm_order},
        "s4_reference_arms": s4_context,
        "paired": paired,
        "notes": [
            "Student metrics are final-epoch; test accuracy is the primary metric.",
            "test_accuracy_fresh_feature re-evaluates with one new feature draw (shots without EMA for F arms, exact for N arms).",
            "theta_norm is the L2 norm of the conditioning parameters (theta for photonic arms, phi for N3).",
            "S4 arms (pqkd, B3f, B1) come from the 8-photon Phase 3 sweep; B3f and B1 involve no photons.",
            "training_hours sums per-epoch seconds with 3 runs in parallel x 4 CPU threads; pauses are excluded.",
        ],
    }


def render(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        f"# S6 MNIST, 4 photons: photonic regimes ({report['scope']})",
        "",
        f"Label: {report['label']}",
        "",
        f"Coverage: {coverage['observed_runs']}/{coverage['expected_runs']} runs. Missing: {coverage['missing'] or 'none'}",
        "",
        "## Per-arm results (final epoch, mean ± s.d. over seeds)",
        "",
    ]
    header = [
        "arm",
        "test acc %",
        "fresh-feature test acc %",
        "val acc %",
        "val CE",
        "Δ vs z=0 (pp)",
        "‖θ‖",
        "train h/run",
    ]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    arm_rows = list(report["arms"].items())
    arm_rows += [
        (f"S4 {arm}", entry) for arm, entry in report["s4_reference_arms"].items()
    ]
    for arm, entry in arm_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    _format(entry["test_accuracy"]),
                    _format(entry["test_accuracy_fresh_feature"]),
                    _format(entry["val_accuracy"]),
                    _format(entry["val_cross_entropy"], 4),
                    _format(entry["delta_vs_zero_feature"]),
                    _format(entry["theta_norm"]),
                    _format(entry["training_hours"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Paired per-seed differences (arm − reference)",
        "",
        f"Rule: {report['comparison_rule']}",
        "",
    ]
    header = [
        "arm",
        "reference",
        "seeds",
        "Δ test acc (pp)",
        "exceeds s.d.?",
        "signs (+/−/0)",
        "Δ val acc (pp)",
        "Δ val CE",
    ]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for entry in report["paired"]:
        test = entry.get("test_accuracy")
        reference = entry["reference"] + (
            " (S4)" if entry["reference_source"] == "s4" else ""
        )
        signs = (
            "—"
            if test is None
            else "{positive}/{negative}/{zero}".format(**test["sign_counts"])
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    entry["arm"],
                    reference,
                    str(entry["seeds"]),
                    _format(test),
                    "—"
                    if test is None
                    else ("yes" if test["exceeds_seed_std"] else "no"),
                    signs,
                    _format(entry.get("val_accuracy")),
                    _format(entry.get("val_cross_entropy"), 4),
                ]
            )
            + " |"
        )
    lines += ["", *[f"- {note}" for note in report["notes"]], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sweep_dirs",
        type=Path,
        nargs="+",
        help="Planned S6 sweep first, then its completion sweeps",
    )
    parser.add_argument("--s4-sweeps", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    sweep_dirs = [path.resolve() for path in args.sweep_dirs]
    report = analyse(sweep_dirs, [path.resolve() for path in args.s4_sweeps])
    text = render(report)
    results_dir = PAPER_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    stem = f"s6_mnist_4ph_{report['scope']}"
    for directory in (results_dir, sweep_dirs[-1]):
        (directory / f"{stem}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (directory / f"{stem}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
