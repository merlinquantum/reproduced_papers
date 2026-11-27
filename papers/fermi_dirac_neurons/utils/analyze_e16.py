"""E16 analysis: scaling of the Pauli-measurement gap, gradient decay rates, cross-class learning.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/analyze_e16.py structure \
        papers/fermi_dirac_neurons/outdir/sweeps/e15_structure_v1 \
        papers/fermi_dirac_neurons/outdir/sweeps/e16_structure_v1
    python3 papers/fermi_dirac_neurons/utils/analyze_e16.py teacher \
        papers/fermi_dirac_neurons/outdir/sweeps/e16_teacher_v1

Rules declared in LOG.md ("Pre-declared design: E16") before any E16 run.

* E16a: gap R(n) = K_eps(contextual) / K_eps(noncontextual) for TFIM vs
  ising_x0 and for the random contextual vs random noncontextual classes
  (median over draws per set, then median over the 3 sets of a class); slope
  b of log10 R over n = 4..10 with a 95 % bootstrap CI (1000 resamples of the
  20 draws at every n and set). Grows exponentially if the CI is above 0,
  constant if it contains 0, shrinks if below 0.
* E16c: per term set, input and parameter position, slope of log10 Var vs n
  over n = 4..10 by weighted least squares (SE of each point from its 95 %
  bootstrap CI); class mean slope over its sets, with a standard error that
  combines the fit errors and, for the random classes, the spread between
  sets; two classes differ if their mean slopes differ by more than 2
  combined standard errors.
* E16b (``teacher``): see :func:`teacher`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

FIT_QUBITS = range(4, 11)
BOOTSTRAP = 1000
CONTEXTUAL = ["tfim", "ctxrand0", "ctxrand1", "ctxrand2"]
NONCONTEXTUAL = ["ising_x0", "ncrand0", "ncrand1", "ncrand2"]
CLASSES = {
    "contextual": CONTEXTUAL,
    "noncontextual": NONCONTEXTUAL,
    "commuting": ["ising"],
}
NAME_PATTERN = re.compile(r"e1[56]_(pauli|grad)_(\w+?)_(\d+)q")


def _completed(sweep_dir: Path):
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_dir} is {status['status']}, not COMPLETED")
    for run in status["runs"]:
        yield run, json.loads((sweep_dir / run["metrics_path"]).read_text())


def load_structure(sweep_dirs: list[Path]) -> tuple[dict, dict]:
    """Return ``paulis[(name, n, theta_range, eps)] = per-draw K`` and ``grads[(name, input, position)][n] = record``."""
    from fermi_dirac_neurons.utils.analyze_e10 import _position

    paulis, grads = {}, defaultdict(dict)
    for sweep_dir in sweep_dirs:
        for run, metrics in _completed(sweep_dir):
            match = NAME_PATTERN.fullmatch(run["config"])
            if match is None:
                continue
            kind, name, n_text = match.groups()
            n_qubits = int(n_text)
            if kind == "pauli":
                for theta_range in sorted(
                    {record["theta_range"] for record in metrics["records"]}
                ):
                    for epsilon in metrics["records"][0]["paulis_needed"]:
                        key = (name, n_qubits, theta_range, epsilon)
                        if key in paulis:
                            raise ValueError(f"Duplicate Pauli cell {key}")
                        paulis[key] = np.array(
                            [
                                record["paulis_needed"][epsilon]
                                for record in metrics["records"]
                                if record["theta_range"] == theta_range
                            ],
                            dtype=float,
                        )
                continue
            n_terms = len(metrics["term_labels"])
            for record in metrics["records"]:
                if record["quantity"] == "output":
                    key = (
                        name,
                        record["input"],
                        _position(record["parameter_index"], n_terms),
                    )
                    if n_qubits in grads[key]:
                        raise ValueError(f"Duplicate gradient cell {key} n={n_qubits}")
                    grads[key][n_qubits] = record
    return paulis, grads


def _class_k(
    paulis: dict,
    names: list[str],
    n_qubits: int,
    theta_range: float,
    epsilon: str,
    rng=None,
) -> float:
    medians = []
    for name in names:
        draws = paulis[(name, n_qubits, theta_range, epsilon)]
        if rng is not None:
            draws = draws[rng.integers(0, len(draws), len(draws))]
        medians.append(np.median(draws))
    return float(np.median(medians))


def gap_scaling(paulis: dict) -> list[dict]:
    """E16a: slope of log10 K_ctx / K_nc over n with a bootstrap CI."""
    rng = np.random.default_rng(16)
    rows = []
    pairings = {
        "tfim vs ising_x0": (["tfim"], ["ising_x0"]),
        "random contextual vs random noncontextual": (
            CONTEXTUAL[1:],
            NONCONTEXTUAL[1:],
        ),
    }
    for label, (contextual, noncontextual) in pairings.items():
        for theta_range in (0.5, 2.0):
            for epsilon in ("0.01", "0.001"):
                qubits = np.array(list(FIT_QUBITS))
                ratios = np.array(
                    [
                        _class_k(paulis, contextual, n, theta_range, epsilon)
                        / _class_k(paulis, noncontextual, n, theta_range, epsilon)
                        for n in qubits
                    ]
                )
                slope = float(np.polyfit(qubits, np.log10(ratios), 1)[0])
                samples = []
                for _ in range(BOOTSTRAP):
                    boot = [
                        _class_k(paulis, contextual, n, theta_range, epsilon, rng)
                        / _class_k(paulis, noncontextual, n, theta_range, epsilon, rng)
                        for n in qubits
                    ]
                    samples.append(np.polyfit(qubits, np.log10(boot), 1)[0])
                low, high = np.percentile(samples, [2.5, 97.5])
                verdict = (
                    "grows exponentially"
                    if low > 0
                    else ("shrinks" if high < 0 else "constant")
                )
                rows.append(
                    {
                        "pairing": label,
                        "theta_range": theta_range,
                        "epsilon": float(epsilon),
                        "ratio_by_n": {
                            str(n): float(r) for n, r in zip(qubits, ratios)
                        },
                        "slope_log10_per_qubit": slope,
                        "slope_ci95": [float(low), float(high)],
                        "factor_per_qubit": 10**slope,
                        "verdict": verdict,
                    }
                )
    return rows


def _weighted_slope(points: dict) -> tuple[float, float]:
    qubits = np.array([n for n in FIT_QUBITS if n in points])
    if len(qubits) != len(FIT_QUBITS):
        raise ValueError(f"Missing n in gradient cell: have {sorted(points)}")
    values = np.array([math.log10(points[n]["variance"]) for n in qubits])
    errors = np.array(
        [
            (
                math.log10(points[n]["variance_ci95"][1])
                - math.log10(points[n]["variance_ci95"][0])
            )
            / 3.92
            for n in qubits
        ]
    )
    weights = 1.0 / np.maximum(errors, 1e-6) ** 2
    design = np.vstack([np.ones_like(qubits, dtype=float), qubits]).T
    covariance = np.linalg.inv(design.T @ (weights[:, None] * design))
    coefficients = covariance @ design.T @ (weights * values)
    return float(coefficients[1]), float(math.sqrt(covariance[1, 1]))


def decay_rates(grads: dict) -> dict:
    """E16c: per-set slopes, class mean slopes and pairwise class differences."""
    per_set, classes = [], []
    inputs = sorted({key[1] for key in grads})
    positions = sorted({key[2] for key in grads})
    for input_kind in inputs:
        for position in positions:
            class_stats = {}
            for class_name, names in CLASSES.items():
                slopes, errors = [], []
                for name in names:
                    slope, error = _weighted_slope(grads[(name, input_kind, position)])
                    slopes.append(slope)
                    errors.append(error)
                    per_set.append(
                        {
                            "term_set": name,
                            "class": class_name,
                            "input": input_kind,
                            "parameter_position": position,
                            "slope_log10_per_qubit": slope,
                            "slope_se": error,
                            "factor_per_qubit": 10**slope,
                        }
                    )
                k = len(slopes)
                between = float(np.var(slopes, ddof=1)) / k if k > 1 else 0.0
                standard_error = math.sqrt(sum(e**2 for e in errors) / k**2 + between)
                class_stats[class_name] = (float(np.mean(slopes)), standard_error)
                classes.append(
                    {
                        "class": class_name,
                        "input": input_kind,
                        "parameter_position": position,
                        "mean_slope": class_stats[class_name][0],
                        "se": standard_error,
                        "factor_per_qubit": 10 ** class_stats[class_name][0],
                    }
                )
            for first, second in (
                ("contextual", "noncontextual"),
                ("contextual", "commuting"),
                ("noncontextual", "commuting"),
            ):
                difference = class_stats[first][0] - class_stats[second][0]
                combined = math.hypot(class_stats[first][1], class_stats[second][1])
                classes.append(
                    {
                        "comparison": f"{first} - {second}",
                        "input": input_kind,
                        "parameter_position": position,
                        "slope_difference": difference,
                        "combined_se": combined,
                        "verdict": "differ"
                        if abs(difference) > 2 * combined
                        else "not distinguishable",
                    }
                )
    return {"per_set": per_set, "classes": classes}


def main() -> None:
    """Dispatch to the requested part."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["structure", "teacher"])
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.part == "structure":
        paulis, grads = load_structure(args.sweep_dirs)
        summary = {
            "sweep_dirs": [str(path) for path in args.sweep_dirs],
            "gap_scaling": gap_scaling(paulis),
            "decay_rates": decay_rates(grads),
        }
        output = PROJECT_DIR / "results" / "e16_structure_summary.json"
    else:
        summary = teacher(args.sweep_dirs[0])
        output = PROJECT_DIR / "results" / "e16_teacher_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output)


TEACHER_PATTERN = re.compile(r"e16_teach_(\w+?)_(\d+)q")
TEACHER_METRICS = ("validation_mse", "train_loss")


def _class_of(name: str) -> str:
    for class_name, names in CLASSES.items():
        if name in names:
            return class_name
    raise ValueError(f"Term set {name} belongs to no class")


def _improvement(sweep_dir: Path) -> list[dict]:
    """Post-hoc diagnostic: log10(final / epoch-0 validation MSE) per student.

    Separates the student whose term set equals the teacher's (it can reach
    zero loss) from the other students, to show how far training moved any
    student from its initialization.
    """
    shifts = defaultdict(list)
    for run, metrics in _completed(sweep_dir):
        teacher_name, _ = TEACHER_PATTERN.fullmatch(run["config"]).groups()
        initial = {
            entry["model"]: entry["mse"]
            for entry in metrics["evaluation_history"]
            if entry["epoch"] == 0 and entry["split"] == "validation"
        }
        for model, record in metrics["models"].items():
            role = "matched" if model == f"student_{teacher_name}" else "other"
            shifts[(role, metrics["n_qubits"])].append(
                math.log10(record["final"]["validation"]["mse"] / initial[model])
            )
    return [
        {
            "student": role,
            "n_qubits": n_qubits,
            "count": len(values),
            "median_log10_final_over_initial": float(np.median(values)),
            "min_log10_final_over_initial": float(np.min(values)),
        }
        for (role, n_qubits), values in sorted(shifts.items())
    ]


def teacher(sweep_dir: Path) -> dict:
    """E16b: advantage of contextual over noncontextual students per teacher class.

    For one (teacher, seed) the advantage is mean log10 loss of the
    noncontextual students minus mean log10 loss of the contextual students,
    leaving out the student identical to the teacher. Positive means the
    contextual students fit better. Reported as the median over teachers x
    seeds of a teacher class at each n, with the count of positive values.
    The commuting (Ising) teacher and student are reported but enter no
    advantage.
    """
    from fermi_dirac_neurons.utils.sweep_statistics import load_values

    values, seeds = load_values(sweep_dir)
    teachers = {}
    for config, _, _ in values:
        match = TEACHER_PATTERN.fullmatch(config)
        if match is None:
            raise ValueError(f"Unexpected config {config} in {sweep_dir}")
        teachers[config] = (match.group(1), int(match.group(2)))
    per_pair, losses = [], []
    for config, (teacher_name, n_qubits) in sorted(teachers.items()):
        for metric in TEACHER_METRICS:
            for seed in seeds:
                by_class = defaultdict(list)
                for student in [name for names in CLASSES.values() for name in names]:
                    log_loss = math.log10(
                        values[(config, f"student_{student}", metric)][seed]
                    )
                    losses.append(
                        {
                            "teacher": teacher_name,
                            "teacher_class": _class_of(teacher_name),
                            "student": student,
                            "student_class": _class_of(student),
                            "n_qubits": n_qubits,
                            "seed": seed,
                            "metric": metric,
                            "log10_loss": log_loss,
                        }
                    )
                    if student != teacher_name:
                        by_class[_class_of(student)].append(log_loss)
                per_pair.append(
                    {
                        "teacher": teacher_name,
                        "teacher_class": _class_of(teacher_name),
                        "n_qubits": n_qubits,
                        "seed": seed,
                        "metric": metric,
                        "contextual_advantage": float(
                            np.mean(by_class["noncontextual"])
                            - np.mean(by_class["contextual"])
                        ),
                        "students_contextual": len(by_class["contextual"]),
                        "students_noncontextual": len(by_class["noncontextual"]),
                    }
                )
    summary = []
    for metric in TEACHER_METRICS:
        for teacher_class in CLASSES:
            for n_qubits in sorted({n for _, n in teachers.values()}):
                advantages = [
                    row["contextual_advantage"]
                    for row in per_pair
                    if row["metric"] == metric
                    and row["teacher_class"] == teacher_class
                    and row["n_qubits"] == n_qubits
                ]
                student_medians = {}
                for student_class in CLASSES:
                    class_losses = [
                        row["log10_loss"]
                        for row in losses
                        if row["metric"] == metric
                        and row["teacher_class"] == teacher_class
                        and row["student_class"] == student_class
                        and row["n_qubits"] == n_qubits
                        and row["student"] != row["teacher"]
                    ]
                    # The only commuting student is the Ising teacher itself.
                    student_medians[student_class] = (
                        float(np.median(class_losses)) if class_losses else None
                    )
                summary.append(
                    {
                        "metric": metric,
                        "teacher_class": teacher_class,
                        "n_qubits": n_qubits,
                        "pairs": len(advantages),
                        "median_contextual_advantage_log10": float(
                            np.median(advantages)
                        ),
                        "positive": int(sum(a > 0 for a in advantages)),
                        "median_student_log10_loss": student_medians,
                    }
                )
    return {
        "sweep_dir": str(sweep_dir),
        "seeds": seeds,
        "summary": summary,
        "improvement_over_initialization": _improvement(sweep_dir),
        "per_pair": per_pair,
        "losses": losses,
    }


if __name__ == "__main__":
    main()
