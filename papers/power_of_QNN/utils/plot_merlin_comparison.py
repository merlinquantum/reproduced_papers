"""Phase 04 figures: MerLin photonic adaptation vs the Phase 03 qubit and classical models.

Usage (from papers/power_of_QNN):
    python utils/plot_merlin_comparison.py --runs results/phase04_runs.json --phase3-runs results/phase03_runs.json --scope paper-accurate

Writes to results/ (and the photonic run directories):
- ``merlin_effective_dimension_d40.png``: normalized effective dimension vs n at d=40;
- ``merlin_fisher_trace_scaling.png``: tr F / d vs input size for photonic and qubit models;
- ``merlin_training_iris.png``: Iris loss curves (photonic d=8 vs Phase 03 models).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from figure_io import RESULTS_DIR, load_metrics  # noqa: E402
from power_of_QNN.lib.plotting import _stamp  # noqa: E402


def _curve(axis, n_values, entry, label, style):
    mean = np.asarray(entry["normalized_effective_dimension_mean"])[: len(n_values)]
    std = np.asarray(entry["normalized_effective_dimension_std"])[: len(n_values)]
    axis.plot(n_values, mean, style, marker="o", markersize=3, label=label)
    axis.fill_between(n_values, mean - std, mean + std, alpha=0.15)


def _save(figure, scope: str, name: str, run_dirs: list[Path]) -> None:
    figure.tight_layout()
    _stamp(figure, scope)
    RESULTS_DIR.mkdir(exist_ok=True)
    for directory in [RESULTS_DIR, *run_dirs]:
        figure.savefig(directory / f"{name}.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--phase3-runs", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    runs = {k: Path(v) for k, v in json.loads(args.runs.read_text()).items()}
    phase3 = {k: Path(v) for k, v in json.loads(args.phase3_runs.read_text()).items()}
    merlin = load_metrics(runs["effdim_d40_merlin"])
    qubit = load_metrics(phase3["effdim_d40_original"])
    n_values = np.asarray(qubit["n_values"][:11])

    figure, axis = plt.subplots(figsize=(7, 4.5))
    _curve(axis, n_values, qubit["models"]["qnn"], "qubit QNN (paper)", "-")
    _curve(axis, n_values, qubit["models"]["easy_qnn"], "easy QNN (paper)", "-")
    _curve(
        axis,
        n_values,
        qubit["models"]["classical_4442"],
        "classical [4,4,4,2] (paper)",
        "-",
    )
    _curve(
        axis,
        n_values,
        merlin["models"]["photonic_hard"],
        "photonic 6-mode planned design",
        "--",
    )
    _curve(
        axis,
        n_values,
        merlin["models"]["photonic_easy"],
        "photonic 6-mode planned design, easy encoding",
        "--",
    )
    if "effdim_d40_merlin_8modes" in runs:
        wide = load_metrics(runs["effdim_d40_merlin_8modes"])
        _curve(
            axis,
            n_values,
            wide["models"]["photonic_hard"],
            "photonic 8-mode planned design (diagnostic)",
            ":",
        )
        _curve(
            axis,
            n_values,
            wide["models"]["photonic_easy"],
            "photonic 8-mode planned design, easy encoding (diagnostic)",
            ":",
        )
    axis.set_xscale("log")
    axis.set_xlabel("number of data n")
    axis.set_ylabel(r"normalised effective dimension $d_{1,n}/d$ (d = 40)")
    axis.legend(fontsize=7)
    _save(
        figure,
        args.scope,
        "merlin_effective_dimension_d40",
        [runs["effdim_d40_merlin"]],
    )

    scaling = load_metrics(runs["spectrum_scaling_merlin"])["models"]
    qubit_scaling = load_metrics(phase3["spectrum_scaling_original"])["models"]
    families = {
        "photonic planned design": [merlin["models"]["photonic_hard"]]
        + [scaling[f"photonic_hard_s{s}"] for s in (6, 8, 10)],
        "photonic planned design, easy encoding": [merlin["models"]["photonic_easy"]]
        + [scaling[f"photonic_easy_s{s}"] for s in (6, 8, 10)],
        "qubit QNN (paper)": [qubit["models"]["qnn"]]
        + [qubit_scaling[f"qnn_s{s}"] for s in (6, 8, 10)],
        "easy QNN (paper)": [qubit["models"]["easy_qnn"]]
        + [qubit_scaling[f"easy_qnn_s{s}"] for s in (6, 8, 10)],
    }
    figure, axis = plt.subplots(figsize=(6, 4))
    for label, entries in families.items():
        sizes = np.asarray([entry["input_size"] for entry in entries])
        values = np.asarray(
            [
                np.mean([draw["mean_trace"] for draw in entry["draws"]]) / entry["d"]
                for entry in entries
            ]
        )
        errors = np.asarray(
            [
                np.std([draw["mean_trace"] for draw in entry["draws"]]) / entry["d"]
                for entry in entries
            ]
        )
        factor = np.exp(np.polyfit(sizes, np.log(values), 1)[0])
        axis.errorbar(
            sizes,
            values,
            yerr=errors,
            marker="o",
            capsize=3,
            linestyle="--" if "photonic" in label else "-",
            label=f"{label}: x{factor:.2f} per input",
        )
    axis.set_yscale("log")
    axis.set_xlabel("input size s_in")
    axis.set_ylabel("mean tr F / d (unnormalized)")
    axis.legend(fontsize=8)
    _save(
        figure,
        args.scope,
        "merlin_fisher_trace_scaling",
        [runs["spectrum_scaling_merlin"]],
    )

    training = load_metrics(runs["training_iris_merlin"])["models"]
    qubit_training = load_metrics(phase3["training_iris_original"])["models"]
    figure, axis = plt.subplots(figsize=(6, 4))
    for label, entry, style in [
        ("qubit QNN (paper)", qubit_training["qnn"], "-"),
        ("easy QNN (paper)", qubit_training["easy_qnn"], "-"),
        ("classical [4,1,1,1,2] (paper)", qubit_training["classical_41112"], "-"),
        ("photonic 6-mode planned design", training["photonic_hard"], "--"),
        (
            "photonic 6-mode planned design, easy encoding",
            training["photonic_easy"],
            "--",
        ),
    ]:
        mean, std = (
            np.asarray(entry["loss_curve_mean"]),
            np.asarray(entry["loss_curve_std"]),
        )
        steps = np.arange(len(mean))
        axis.plot(steps, mean, style, label=label)
        axis.fill_between(steps, mean - std, mean + std, alpha=0.1)
    axis.set_xlabel("training iteration")
    axis.set_ylabel("cross-entropy loss (Iris classes 0/1, d = 8)")
    axis.legend(fontsize=8)
    _save(figure, args.scope, "merlin_training_iris", [runs["training_iris_merlin"]])


if __name__ == "__main__":
    main()
