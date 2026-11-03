"""Analyse the S4 sweep: per-arm statistics, paired deltas vs PQKD, Table 1 comparison.

Reads only structured artifacts (``sweep_status.json`` and each run's
``metrics.json``), never logs. The comparison rule is the one predeclared in
LOG.md (Sweep Plan, S4): no winner is declared; an arm differs from PQKD only
when the mean paired per-seed difference exceeds its standard deviation
across seeds.

Usage (from the repository root)::

    python papers/photonic_QKD/utils/analyze_s4.py <S4 sweep dir> [<completion sweep dir> ...]         --teacher-sweep <S1 sweep dir>

Writes ``results/s4_mnist_<scope>.json`` and ``.md`` (``scope`` is
``complete`` or ``partial``) and a copy in the last sweep directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PAPER_DIR = Path(__file__).resolve().parents[1]
REFERENCE_ARM = "pqkd"
# Paper Table 1, MNIST (mean ± s.d. over 5 runs, 100 epochs).
PAPER_TABLE1_MNIST = {
    "teacher": {
        "train_accuracy": (100.00, 0.00),
        "val_accuracy": (99.07, 0.13),
        "val_cross_entropy": (0.031, 0.002),
    },
    "pqkd": {
        "train_accuracy": (99.71, 0.02),
        "val_accuracy": (99.09, 0.09),
        "val_cross_entropy": (0.030, 0.001),
        "photonic_delta": (5.94, 0.02),
    },
}
ARM_METRICS = {
    "test_accuracy": "final_evaluation.test_accuracy",
    "val_accuracy": "final_evaluation.val_accuracy",
    "val_cross_entropy": "final_evaluation.val_cross_entropy",
    "train_accuracy": "final_epoch.train_accuracy",
    "test_drop_vs_teacher": "final_evaluation.test_accuracy_drop_vs_teacher",
    "delta_vs_zero_feature": "final_evaluation.delta_test_accuracy_vs_zero_feature",
    "test_accuracy_fresh_feature": "final_evaluation.test_accuracy_fresh_feature",
    "proxy_delta_first_minus_final": "proxy_objective_delta_first_minus_final",
    "trainable_parameters": "parameters.student_total_trainable",
    "compression_overall": "parameters.compression_overall",
}
PAIRED_METRICS = ("test_accuracy", "val_accuracy", "val_cross_entropy")


def _dotted(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    std = (
        math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        if len(values) > 1
        else None
    )
    return {"mean": mean, "std": std, "n": len(values), "values": values}


def _paper_relative(path: Path) -> str:
    """Record a sweep directory relative to the paper folder, not as a local absolute path."""
    try:
        return path.relative_to(PAPER_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _format(entry: dict[str, Any] | None, digits: int = 2) -> str:
    if entry is None:
        return "—"
    if entry["std"] is None:
        return f"{entry['mean']:.{digits}f} (n=1)"
    return f"{entry['mean']:.{digits}f} ± {entry['std']:.{digits}f} (n={entry['n']})"


def load_runs(
    sweep_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]]]:
    """Return the sweep status and ``{arm: {seed: metrics}}`` for completed runs."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    runs: dict[str, dict[int, dict[str, Any]]] = {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        runs.setdefault(run["candidate"], {})[int(run["seed"])] = metrics
    return status, runs


def analyse(sweep_dirs: list[Path], teacher_sweep: Path) -> dict[str, Any]:
    """Merge completed runs of the planned S4 sweep and any completion sweeps.

    The first directory holds the planned spec; coverage is judged against
    its full candidates x seeds matrix. A (candidate, seed) pair completed in
    more than one sweep is an error, never silently de-duplicated.
    """
    spec = json.loads((sweep_dirs[0] / "spec.json").read_text())
    arm_order = [candidate["name"] for candidate in spec["candidates"]]
    expected = {(arm, int(seed)) for arm in arm_order for seed in spec["seeds"]}
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
    observed = {(arm, seed) for arm, seeds in runs.items() for seed in seeds}
    complete = observed == expected

    arms = {}
    for arm in arm_order:
        seeds = runs.get(arm, {})
        arms[arm] = {
            "seeds": sorted(seeds),
            **{
                name: _stats(
                    [
                        float(value)
                        for seed in sorted(seeds)
                        if isinstance(value := _dotted(seeds[seed], path), (int, float))
                        and math.isfinite(value)
                    ]
                )
                for name, path in ARM_METRICS.items()
            },
        }

    reference = runs.get(REFERENCE_ARM, {})
    paired = {}
    for arm in arm_order:
        if arm == REFERENCE_ARM:
            continue
        seeds = sorted(set(reference) & set(runs.get(arm, {})))
        paired[arm] = {"seeds": seeds}
        for name in PAIRED_METRICS:
            path = ARM_METRICS[name]
            differences = [
                float(_dotted(runs[arm][s], path)) - float(_dotted(reference[s], path))
                for s in seeds
            ]
            entry = _stats(differences)
            if entry is not None:
                exceeds = entry["std"] is not None and abs(entry["mean"]) > entry["std"]
                entry["exceeds_seed_std"] = exceeds
                entry["sign_counts"] = {
                    "positive": sum(d > 0 for d in differences),
                    "negative": sum(d < 0 for d in differences),
                    "zero": sum(d == 0 for d in differences),
                }
            paired[arm][name] = entry

    teacher_status = json.loads((teacher_sweep / "sweep_status.json").read_text())
    teacher_values: dict[str, list[float]] = {
        "train_accuracy": [],
        "val_accuracy": [],
        "val_cross_entropy": [],
        "test_accuracy": [],
    }
    for run in teacher_status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads((teacher_sweep / run["metrics_path"]).read_text())
        teacher_values["train_accuracy"].append(
            metrics["final_epoch"]["train_accuracy"]
        )
        teacher_values["val_accuracy"].append(metrics["final_epoch"]["val_accuracy"])
        teacher_values["val_cross_entropy"].append(
            metrics["final_epoch"]["val_cross_entropy"]
        )
        teacher_values["test_accuracy"].append(
            metrics["best_checkpoint"]["test_accuracy"]
        )
    teacher = {name: _stats(values) for name, values in teacher_values.items()}

    return {
        "sweep_id": spec["sweep_id"],
        "sweep_dirs": [_paper_relative(path) for path in sweep_dirs],
        "run_sources": sources,
        "scope": "complete" if complete else "partial",
        "label": "partial RED-tier run (30 epochs, 3 seeds)"
        + ("" if complete else "; INCOMPLETE COVERAGE"),
        "coverage": {
            "observed_runs": len(observed),
            "expected_runs": len(expected),
            "missing": sorted(f"{a}__seed{s}" for a, s in expected - observed),
        },
        "comparison_rule": "no winner declared; arm differs from pqkd only if |mean paired difference| > its s.d. across seeds",
        "arms": arms,
        "paired_vs_pqkd": paired,
        "teacher_s1": teacher,
        "paper_table1_mnist": PAPER_TABLE1_MNIST,
        "notes": [
            "Teacher val/train are final-epoch (as in the paper's Table 1); teacher test is the best-val checkpoint.",
            "Student metrics are final-epoch (paper reports final epoch).",
            "The paper tunes theta on the validation split (SPSA) and reports validation; test is the primary metric here.",
        ],
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        f"# S4 MNIST: PQKD vs baselines ({report['scope']})",
        "",
        f"Label: {report['label']}",
        "",
    ]
    coverage = report["coverage"]
    lines.append(
        f"Coverage: {coverage['observed_runs']}/{coverage['expected_runs']} runs. Missing: {coverage['missing'] or 'none'}"
    )
    lines += ["", "## Per-arm results (final epoch, mean ± s.d. over seeds)", ""]
    header = [
        "arm",
        "test acc %",
        "val acc %",
        "val CE",
        "train acc %",
        "Δ vs z=0 (pp)",
        "trainable params",
        "C_x",
    ]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for arm, entry in report["arms"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    _format(entry["test_accuracy"]),
                    _format(entry["val_accuracy"]),
                    _format(entry["val_cross_entropy"], 4),
                    _format(entry["train_accuracy"]),
                    _format(entry["delta_vs_zero_feature"]),
                    _format(entry["trainable_parameters"], 0),
                    _format(entry["compression_overall"], 3),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Paired per-seed differences vs pqkd (arm − pqkd)",
        "",
        f"Rule: {report['comparison_rule']}",
        "",
    ]
    header = [
        "arm",
        "seeds",
        "Δ test acc (pp)",
        "exceeds s.d.?",
        "Δ val acc (pp)",
        "Δ val CE",
    ]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for arm, entry in report["paired_vs_pqkd"].items():
        test = entry.get("test_accuracy")
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    str(entry["seeds"]),
                    _format(test),
                    "—"
                    if test is None
                    else ("yes" if test["exceeds_seed_std"] else "no"),
                    _format(entry.get("val_accuracy")),
                    _format(entry.get("val_cross_entropy"), 4),
                ]
            )
            + " |"
        )
    teacher = report["teacher_s1"]
    pqkd = report["arms"].get(REFERENCE_ARM, {})
    paper = report["paper_table1_mnist"]
    lines += [
        "",
        "## Table 1 (MNIST) comparison",
        "",
        "| quantity | paper | reproduced |",
        "|---|---|---|",
    ]
    rows = [
        (
            "teacher train acc %",
            paper["teacher"]["train_accuracy"],
            teacher["train_accuracy"],
        ),
        (
            "teacher val acc %",
            paper["teacher"]["val_accuracy"],
            teacher["val_accuracy"],
        ),
        (
            "teacher val CE",
            paper["teacher"]["val_cross_entropy"],
            teacher["val_cross_entropy"],
        ),
        (
            "PQKD train acc %",
            paper["pqkd"]["train_accuracy"],
            pqkd.get("train_accuracy"),
        ),
        ("PQKD val acc %", paper["pqkd"]["val_accuracy"], pqkd.get("val_accuracy")),
        (
            "PQKD val CE",
            paper["pqkd"]["val_cross_entropy"],
            pqkd.get("val_cross_entropy"),
        ),
        (
            "photonic Δ (J first − final)",
            paper["pqkd"]["photonic_delta"],
            pqkd.get("proxy_delta_first_minus_final"),
        ),
    ]
    for name, (paper_mean, paper_std), ours in rows:
        digits = 4 if "CE" in name else 2
        lines.append(
            f"| {name} | {paper_mean:.{digits}f} ± {paper_std:.{digits}f} (n=5) | {_format(ours, digits)} |"
        )
    lines += ["", *[f"- {note}" for note in report["notes"]], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sweep_dirs",
        type=Path,
        nargs="+",
        help="Planned S4 sweep first, then completion sweeps",
    )
    parser.add_argument("--teacher-sweep", type=Path, required=True)
    args = parser.parse_args()
    sweep_dirs = [path.resolve() for path in args.sweep_dirs]
    report = analyse(sweep_dirs, args.teacher_sweep.resolve())
    text = render(report)
    results_dir = PAPER_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    stem = f"s4_mnist_{report['scope']}"
    for directory in (results_dir, sweep_dirs[-1]):
        (directory / f"{stem}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (directory / f"{stem}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
