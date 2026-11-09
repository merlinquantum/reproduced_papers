"""Plot the S4 comparison (PQKD vs baselines) and the z-stability diagnostic.

Reads the structured reports written by ``utils/analyze_s4.py`` and
``utils/feature_stability.py``; never reads logs or hard-codes run
directories.

Usage (from the repository root)::

    python papers/photonic_QKD/utils/plot_s4.py \
        --report papers/photonic_QKD/results/s4_mnist_complete.json \
        --stability papers/photonic_QKD/results/s4_feature_stability.json \
        [--also-save-to <sweep dir>]

Writes ``results/s4_mnist_test_accuracy.png`` and
``results/s4_feature_stability.png`` (and copies to ``--also-save-to``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
LABELS = {
    "pqkd": "PQKD\n(photonic, SPSA θ)",
    "B4_fixed_photonic": "B4 photonic\nθ frozen",
    "B3f_random_fixed": "B3f random z\nfixed",
    "B1_kd_dense": "B1 dense KD\niso-param",
    "B2_dictionary_trainable": "B2 learned\nmixing (+33k)",
    "B5_surrogate": "B5 classical\nsurrogate",
    "B6_classical_histogram": "B6 classical\nhistogram",
    "B3r_random_resampled": "B3r random z\nper epoch",
}
PHOTONIC = {"pqkd", "B4_fixed_photonic"}


def _save(figure: plt.Figure, name: str, extra: Path | None) -> list[Path]:
    paths = [PAPER_DIR / "results" / name]
    if extra is not None:
        paths.append(extra / name)
    for path in paths:
        figure.savefig(path, dpi=150, bbox_inches="tight")
    return paths


def plot_accuracy(report: dict, extra: Path | None) -> list[Path]:
    arms = [arm for arm in report["arms"] if arm in LABELS]
    order = sorted(arms, key=lambda arm: -report["arms"][arm]["test_accuracy"]["mean"])
    figure, axis = plt.subplots(figsize=(10, 4.2))
    teacher = report["teacher_s1"]["test_accuracy"]
    axis.axhspan(
        teacher["mean"] - teacher["std"],
        teacher["mean"] + teacher["std"],
        color="0.85",
        label="teacher test acc ± s.d.",
    )
    for position, arm in enumerate(order):
        entry = report["arms"][arm]["test_accuracy"]
        colour = "tab:purple" if arm in PHOTONIC else "tab:gray"
        axis.errorbar(
            position,
            entry["mean"],
            yerr=entry["std"],
            fmt="o",
            color=colour,
            capsize=4,
            markersize=7,
        )
        axis.scatter(
            [position] * len(entry["values"]),
            entry["values"],
            color=colour,
            alpha=0.35,
            s=14,
        )
    axis.set_xticks(range(len(order)), [LABELS[arm] for arm in order], fontsize=8)
    axis.set_ylabel("final-epoch test accuracy (%)")
    zero = report["arms"]["B0_zero"]["test_accuracy"]
    axis.set_title(
        "MNIST, conv1+conv2 compressed (C_x ≈ 1.5): PQKD vs fair baselines\n"
        f"{report['label']}; mean ± s.d. and per-seed points. B0 (z = 0) off-scale at {zero['mean']:.1f}%",
        fontsize=9,
    )
    axis.scatter([], [], color="tab:purple", label="photonic feature")
    axis.scatter([], [], color="tab:gray", label="classical / no feature")
    axis.legend(fontsize=8, loc="lower left")
    paths = _save(figure, "s4_mnist_test_accuracy.png", extra)
    plt.close(figure)
    return paths


def plot_stability(stability: dict, extra: Path | None) -> list[Path]:
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    floor = 1e-4
    for row in stability["rows"]:
        if row["arm"] == "B0_zero":
            continue
        steps = [max(run["used_step_mean"], floor) for run in row["runs"]]
        accuracies = [run["test_accuracy"] for run in row["runs"]]
        colour = "tab:purple" if row["arm"] in PHOTONIC else "tab:gray"
        axis.scatter(steps, accuracies, color=colour, s=22)
        axis.annotate(
            LABELS.get(row["arm"], row["arm"]).replace("\n", " "),
            (steps[0], accuracies[0]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.set_xscale("log")
    axis.set_xlabel(
        f"mean relative epoch-to-epoch change of training z (floored at {floor:g})"
    )
    axis.set_ylabel("final-epoch test accuracy (%)")
    axis.set_title(
        "Accuracy follows z stability, not z origin (partial RED-tier, 3 seeds)\n"
        f"Spearman ρ(step, acc) = {stability['spearman_step_vs_test_accuracy']:.2f}, n = {stability['n_runs']} runs, B0 excluded",
        fontsize=9,
    )
    paths = _save(figure, "s4_feature_stability.png", extra)
    plt.close(figure)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stability", type=Path, required=True)
    parser.add_argument("--also-save-to", type=Path, default=None)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    stability = json.loads(args.stability.read_text())
    for path in plot_accuracy(report, args.also_save_to) + plot_stability(
        stability, args.also_save_to
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
