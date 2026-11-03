"""Plot the S6 Phase 4 comparison (4 photons): trained-z and fresh-z test accuracy per arm.

Reads the structured report written by ``utils/analyze_s6.py``; never reads
logs or hard-codes run directories.

Usage (from the repository root)::

    python papers/photonic_QKD/utils/plot_s6.py \
        --report papers/photonic_QKD/results/s6_mnist_4ph_complete.json \
        [--also-save-to <sweep dir>]

Writes ``results/s6_mnist_4ph_test_accuracy.png`` (and a copy to
``--also-save-to``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
ARM_ORDER = (
    "P0_pqkd_theta0",
    "F1_mixing_s200_ema",
    "F2_mixing_s200_noema",
    "F3_mixing_s25_ema",
    "F4_mixing_s25_noema",
    "N1_native_grad_random",
    "N2_native_grad_zero",
    "N3_surrogate_grad",
)
LABELS = {
    "P0_pqkd_theta0": "P0 PQKD\nθ₀ = 0",
    "F1_mixing_s200_ema": "F1 S=200\nEMA",
    "F2_mixing_s200_noema": "F2 S=200\nno EMA",
    "F3_mixing_s25_ema": "F3 S=25\nEMA",
    "F4_mixing_s25_noema": "F4 S=25\nno EMA",
    "N1_native_grad_random": "N1 exact,\ngrad, θ₀ random",
    "N2_native_grad_zero": "N2 exact,\ngrad, θ₀ = 0",
    "N3_surrogate_grad": "N3 classical\nsurrogate, grad",
}
# Categorical slots 1-3 of the reference palette in fixed order, plus a neutral
# for the classical arm; identity is also carried by the legend and tick labels.
FAMILIES = {
    "P0_pqkd_theta0": ("paper-faithful PQKD (θ₀ = 0)", "#2a78d6"),
    "F1_mixing_s200_ema": ("photonic, mixing θ, sampled shots + SPSA", "#eb6834"),
    "F2_mixing_s200_noema": ("photonic, mixing θ, sampled shots + SPSA", "#eb6834"),
    "F3_mixing_s25_ema": ("photonic, mixing θ, sampled shots + SPSA", "#eb6834"),
    "F4_mixing_s25_noema": ("photonic, mixing θ, sampled shots + SPSA", "#eb6834"),
    "N1_native_grad_random": (
        "photonic, exact MerLin probabilities + autograd",
        "#1baf7a",
    ),
    "N2_native_grad_zero": (
        "photonic, exact MerLin probabilities + autograd",
        "#1baf7a",
    ),
    "N3_surrogate_grad": ("classical 45-parameter surrogate + autograd", "#6b6a64"),
}
REFERENCE_INK = "#6b6a64"


def _panel(axis: plt.Axes, report: dict, metric: str) -> None:
    for position, arm in enumerate(ARM_ORDER):
        entry = report["arms"][arm][metric]
        _, colour = FAMILIES[arm]
        axis.scatter(
            [position] * len(entry["values"]),
            entry["values"],
            color=colour,
            alpha=0.45,
            s=22,
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
        axis.errorbar(
            position,
            entry["mean"],
            yerr=entry["std"],
            fmt="o",
            color=colour,
            capsize=4,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
            linewidth=1.5,
            zorder=4,
        )
    axis.set_xticks(
        range(len(ARM_ORDER)), [LABELS[arm] for arm in ARM_ORDER], fontsize=7.5
    )
    axis.grid(axis="y", color="0.9", linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def plot_accuracy(report: dict, extra: Path | None) -> list[Path]:
    figure, (trained, fresh) = plt.subplots(1, 2, figsize=(14, 4.8))

    _panel(trained, report, "test_accuracy")
    reference_handles = {}
    for arm, style, name in (
        ("B3f_random_fixed", "--", "S4 B3f fixed random z"),
        ("B1_kd_dense", ":", "S4 B1 dense KD, iso-param"),
    ):
        mean = report["s4_reference_arms"][arm]["test_accuracy"]["mean"]
        reference_handles[f"{name} ({mean:.2f}%)"] = trained.axhline(
            mean, color=REFERENCE_INK, linestyle=style, linewidth=1.2
        )
    trained.set_ylabel("final-epoch test accuracy (%)")
    trained.set_title(
        "Evaluated with the feature z used in training", fontsize=9.5, loc="left"
    )

    _panel(fresh, report, "test_accuracy_fresh_feature")
    fresh.set_ylim(0, 102)
    fresh.axhline(10, color=REFERENCE_INK, linestyle=":", linewidth=1.0)
    fresh.annotate(
        "chance",
        (len(ARM_ORDER) - 0.5, 10),
        fontsize=7.5,
        color="0.3",
        ha="right",
        va="bottom",
    )
    fresh.set_ylabel("test accuracy with one fresh z draw (%)")
    fresh.set_title(
        "Evaluated with one fresh z (new shots, no EMA; exact arms unchanged)",
        fontsize=9.5,
        loc="left",
    )

    handles = {}
    for arm in ARM_ORDER:
        name, colour = FAMILIES[arm]
        handles.setdefault(
            name,
            plt.Line2D([], [], marker="o", linestyle="", color=colour, markersize=7),
        )
    handles.update(reference_handles)
    figure.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    figure.suptitle(
        "MNIST, 4 photons in 16 modes: photonic conditioning regimes vs PQKD and classical arms\n"
        f"{report['label']}; mean ± s.d. (large) and per-seed points (small)",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    paths = [PAPER_DIR / "results" / "s6_mnist_4ph_test_accuracy.png"]
    if extra is not None:
        paths.append(extra / "s6_mnist_4ph_test_accuracy.png")
    for path in paths:
        figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--also-save-to", type=Path, default=None)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    for path in plot_accuracy(report, args.also_save_to):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
