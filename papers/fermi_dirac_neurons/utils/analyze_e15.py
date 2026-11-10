"""E15 analysis: contextual vs noncontextual term sets at T = 2 with J = 2n - 1.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/analyze_e15.py learning \
        papers/fermi_dirac_neurons/outdir/sweeps/e15_learning_v1
    python3 papers/fermi_dirac_neurons/utils/analyze_e15.py structure \
        papers/fermi_dirac_neurons/outdir/sweeps/e15_structure_v1

``learning`` (part 1): per (task, n, term set) the median final metric over
seeds; paired by seed, the declared contrasts (5 seeds: medians and seed
counts only, no significance test): TFIM (contextual) vs ising_x0
(noncontextual), each of them vs Ising (commuting), and the mean over the
three random contextual sets vs the mean over the three random
noncontextual sets. Loss metrics are compared in log10; accuracy in points.
Every model's Kirby-Love flag is recomputed from the term labels stored in
its metrics. Writes ``results/e15_learning_summary.json``.

``structure`` (parts 2 and 3): see :func:`structure`.
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

from fermi_dirac_neurons.lib import contextuality as cx  # noqa: E402

LEARN_PATTERN = re.compile(r"e15_learn_(fig[78])_(\d)q")
METRICS = {
    "fig7": ["train_loss", "validation_mse"],
    "fig8": ["validation_accuracy_linear", "train_loss"],
}


def _completed_runs(sweep_dir: Path) -> list[tuple[dict, dict]]:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"Sweep is {status['status']}, not COMPLETED")
    return [
        (run, json.loads((sweep_dir / run["metrics_path"]).read_text()))
        for run in status["runs"]
    ]


def _transform(metric: str, value: float) -> float:
    return (
        100 * value if metric.startswith("validation_accuracy") else math.log10(value)
    )


def _group_means(
    values: dict, task: tuple, group: list[str], metric: str, seeds: list[int]
) -> np.ndarray:
    """Per seed, the mean transformed metric over the term sets in ``group``."""
    return np.array(
        [
            np.mean(
                [
                    _transform(metric, values[(task, name, metric)][seed])
                    for name in group
                ]
            )
            for seed in seeds
        ]
    )


def learning(sweep_dir: Path) -> dict:
    """Part 1: learning performance per term set and the declared contrasts."""
    values: dict = defaultdict(dict)
    flags = {}
    for run, metrics in _completed_runs(sweep_dir):
        figure, n_text = LEARN_PATTERN.fullmatch(run["config"]).groups()
        task = (figure, int(n_text))
        for model, record in metrics["models"].items():
            name = model.removeprefix("model_")
            labels = record["term_labels"]
            flags[(task[1], name)] = {
                "kirby_love_noncontextual": cx.is_noncontextual(labels),
                "commuting": bool(cx.commutation_matrix(labels).all()),
            }
            entries = {
                "train_loss": record["last_logged_train_loss"],
                **{
                    f"validation_{key}": value
                    for key, value in record["final"]["validation"].items()
                },
            }
            for metric, value in entries.items():
                values[(task, name, metric)][run["seed"]] = value
    seeds = sorted({seed for bucket in values.values() for seed in bucket})
    rows, contrasts = [], []
    tasks = sorted({key[0] for key in values})
    for task in tasks:
        figure, n_qubits = task
        names = sorted({key[1] for key in values if key[0] == task})
        for metric in METRICS[figure]:
            for name in names:
                series = [values[(task, name, metric)][seed] for seed in seeds]
                rows.append(
                    {
                        "task": f"{figure}_{n_qubits}q",
                        "term_set": name,
                        "metric": metric,
                        "median": float(np.median(series)),
                        **flags[(n_qubits, name)],
                    }
                )

            higher_is_better = metric.startswith("validation_accuracy")
            for label, first, second in (
                (
                    "tfim (contextual) vs ising_x0 (noncontextual)",
                    ["tfim"],
                    ["ising_x0"],
                ),
                ("tfim (contextual) vs ising (commuting)", ["tfim"], ["ising"]),
                (
                    "ising_x0 (noncontextual) vs ising (commuting)",
                    ["ising_x0"],
                    ["ising"],
                ),
                (
                    "random contextual vs random noncontextual (means of 3 sets)",
                    [n for n in names if n.startswith("ctxrand")],
                    [n for n in names if n.startswith("ncrand")],
                ),
            ):
                difference = _group_means(
                    values, task, first, metric, seeds
                ) - _group_means(values, task, second, metric, seeds)
                advantage = difference if higher_is_better else -difference
                contrasts.append(
                    {
                        "task": f"{figure}_{n_qubits}q",
                        "metric": metric,
                        "contrast": label,
                        "unit": "accuracy points"
                        if higher_is_better
                        else "decades of loss",
                        "median_advantage_of_first": float(np.median(advantage)),
                        "seeds_favouring_first": int(np.sum(advantage > 0)),
                        "n_seeds": len(seeds),
                    }
                )
    mislabeled = [
        (n, name)
        for (n, name), flag in flags.items()
        if (name.startswith("ctxrand") or (name == "tfim" and n >= 3))
        and flag["kirby_love_noncontextual"]
        or (name.startswith("ncrand") or name == "ising_x0")
        and not flag["kirby_love_noncontextual"]
    ]
    if mislabeled:
        raise ValueError(
            f"Term sets whose Kirby-Love flag contradicts their class: {mislabeled}"
        )
    return {
        "sweep_dir": str(sweep_dir),
        "seeds": seeds,
        "per_term_set": rows,
        "contrasts": contrasts,
    }


GRAD_PATTERN = re.compile(r"e15_grad_(\w+?)_(\d+)q")
PAULI_PATTERN = re.compile(r"e15_pauli_(\w+?)_(\d+)q")
CLASS_OF = {
    "ising": "ising (commuting)",
    "ising_x0": "ising_x0 (noncontextual)",
    "tfim": "tfim (contextual)",
}


def _class(name: str) -> str:
    if name.startswith("ncrand"):
        return "random noncontextual"
    if name.startswith("ctxrand"):
        return "random contextual"
    return CLASS_OF[name]


def structure(sweep_dir: Path) -> dict:
    """Parts 2 and 3: gradient variance vs n and Pauli strings needed, per term-set class.

    Gradient variance: for each class, input and parameter position, the
    variance at each n (median over the 3 random sets for the random
    classes), classified with the E10 rule (``analyze_e10.classify``).
    Pauli strings needed: median K_eps over the 20 draws of each run, then
    the median over the random sets of a class.
    """
    from fermi_dirac_neurons.utils.analyze_e10 import _position, classify

    gradient: dict = defaultdict(lambda: defaultdict(list))
    pauli: dict = defaultdict(lambda: defaultdict(list))
    for run, metrics in _completed_runs(sweep_dir):
        grad_match = GRAD_PATTERN.fullmatch(run["config"])
        if grad_match:
            name, n_text = grad_match.groups()
            n_terms = len(metrics["term_labels"])
            for record in metrics["records"]:
                if record["quantity"] != "output":
                    continue
                key = (
                    _class(name),
                    record["input"],
                    _position(record["parameter_index"], n_terms),
                )
                gradient[key][int(n_text)].append(record["variance"])
            continue
        name, n_text = PAULI_PATTERN.fullmatch(run["config"]).groups()
        for theta_range in sorted(
            {record["theta_range"] for record in metrics["records"]}
        ):
            for epsilon in metrics["records"][0]["paulis_needed"]:
                needed = [
                    record["paulis_needed"][epsilon]
                    for record in metrics["records"]
                    if record["theta_range"] == theta_range
                ]
                pauli[(_class(name), theta_range, epsilon)][int(n_text)].append(
                    float(np.median(needed))
                )
    gradient_rows = []
    for (cls, input_kind, position), by_n in sorted(gradient.items()):
        points = {
            n: {"variance": float(np.median(values))} for n, values in by_n.items()
        }
        gradient_rows.append(
            {
                "class": cls,
                "input": input_kind,
                "parameter_position": position,
                **classify(points),
                "variance_by_n": {
                    str(n): points[n]["variance"] for n in sorted(points)
                },
            }
        )
    pauli_rows = []
    for (cls, theta_range, epsilon), by_n in sorted(pauli.items()):
        pauli_rows.append(
            {
                "class": cls,
                "theta_range": theta_range,
                "epsilon": float(epsilon),
                "paulis_needed_by_n": {
                    str(n): float(np.median(values))
                    for n, values in sorted(by_n.items())
                },
                "fraction_of_all_strings_by_n": {
                    str(n): float(np.median(values)) / (4**n - 1)
                    for n, values in sorted(by_n.items())
                },
            }
        )
    return {
        "sweep_dir": str(sweep_dir),
        "gradient_variance": gradient_rows,
        "paulis_needed": pauli_rows,
    }


# Categorical slots in fixed order, one per term-set class.
CLASS_COLOURS = {
    "ising (commuting)": "#eb6834",
    "ising_x0 (noncontextual)": "#1baf7a",
    "tfim (contextual)": "#2a78d6",
    "random noncontextual": "#eda100",
    "random contextual": "#e87ba4",
}


def plot_structure(summary: dict, sweep_dir: Path) -> None:
    """Gradient variance vs n (first parameter) and Pauli strings needed vs n."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fermi_dirac_neurons.lib.plotting import (
        TEXT_SECONDARY,
        apply_style,
        finish,
        save,
    )

    apply_style()
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    titles = {
        "zero": "|0…0⟩ input",
        "product": "random product inputs",
        "haar": "Haar-random inputs",
    }
    for axis, input_kind in zip(axes[0], ("zero", "product", "haar")):
        for cls, colour in CLASS_COLOURS.items():
            row = next(
                (
                    r
                    for r in summary["gradient_variance"]
                    if r["class"] == cls
                    and r["input"] == input_kind
                    and r["parameter_position"] == "first"
                ),
                None,
            )
            if row is None:
                continue
            axis.plot(
                [int(n) for n in row["variance_by_n"]],
                list(row["variance_by_n"].values()),
                color=colour,
                marker="o",
                markersize=3,
                label=cls,
            )
        axis.set_yscale("log")
        axis.set_title(f"Gradient variance, {titles[input_kind]}")
        axis.set_xlabel("qubits n")
        axis.set_ylabel("Var over θ of ∂f/∂θ₁")
    for axis, theta_range in zip(axes[1][:2], (0.5, 2.0)):
        for cls, colour in CLASS_COLOURS.items():
            row = next(
                (
                    r
                    for r in summary["paulis_needed"]
                    if r["class"] == cls
                    and r["theta_range"] == theta_range
                    and r["epsilon"] == 0.01
                ),
                None,
            )
            if row is None:
                continue
            axis.plot(
                [int(n) for n in row["paulis_needed_by_n"]],
                list(row["paulis_needed_by_n"].values()),
                color=colour,
                marker="o",
                markersize=3,
                label=cls,
            )
        reference = np.arange(3, 9)
        axis.plot(
            reference,
            4.0**reference - 1,
            color=TEXT_SECONDARY,
            linestyle=":",
            linewidth=1.0,
            label="all 4ⁿ − 1 strings",
        )
        axis.set_yscale("log")
        axis.set_title(
            f"Strings for error ≤ 0.01 on any input, θ ~ U[−{theta_range:g}, {theta_range:g}]"
        )
        axis.set_xlabel("qubits n")
        axis.set_ylabel("Pauli strings needed (median)")
    axis = axes[1][2]
    for cls, colour in CLASS_COLOURS.items():
        row = next(
            (
                r
                for r in summary["paulis_needed"]
                if r["class"] == cls
                and r["theta_range"] == 2.0
                and r["epsilon"] == 0.001
            ),
            None,
        )
        if row is None:
            continue
        axis.plot(
            [int(n) for n in row["fraction_of_all_strings_by_n"]],
            list(row["fraction_of_all_strings_by_n"].values()),
            color=colour,
            marker="o",
            markersize=3,
        )
    axis.set_yscale("log")
    axis.set_title("Fraction of all strings, error ≤ 0.001, θ ~ U[−2, 2]")
    axis.set_xlabel("qubits n")
    axis.set_ylabel("strings needed / (4ⁿ − 1)")
    handles, labels = axes[1][0].get_legend_handles_labels()
    finish(
        figure,
        handles,
        labels,
        "E15: commuting, noncontextual and contextual term sets with J = 2n − 1, T = 2",
        "extension (not in the paper); seed 0; gradients: 200 θ draws ~ U[−0.5, 0.5], first parameter; random classes: median over 3 sets",
        columns=3,
    )
    for path in save(figure, "e15_structure", sweep_dir):
        print(path)


def main() -> None:
    """Dispatch to the requested part."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["learning", "structure"])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    if args.part == "learning":
        output = PROJECT_DIR / "results" / "e15_learning_summary.json"
        output.write_text(
            json.dumps(learning(args.sweep_dir), indent=2) + "\n", encoding="utf-8"
        )
        print(output)
    else:
        summary = structure(args.sweep_dir)
        output = PROJECT_DIR / "results" / "e15_structure_summary.json"
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(output)
        plot_structure(summary, args.sweep_dir)


if __name__ == "__main__":
    main()
