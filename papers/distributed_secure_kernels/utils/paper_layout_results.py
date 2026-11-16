"""Table 1, Fig. 3 and Fig. 4 in the paper's layout, plus Table 1 bis (corrected protocol).

Reads only the committed ``results/sweep_results.json`` (per-seed values of every
sweep), so it runs from a fresh clone. Writes ``results/table1.md``,
``results/table1_bis.md``, ``results/table1_merlin.md``, ``results/fig3.png``,
``results/fig4.png`` and ``results/fig4_merlin.png``.

Paper values: Table 1 and Fig. 4 as printed; Fig. 3 prints bars only, so its
values come from the authors' logs (``Heart_Dataset_Experiments/
heart_dataset_kernel_1060336.txt``, ``Parkinsons_Dataset_Experiments/
park_dataset_kernel_1055216.txt``, commit ``aed2cf9``).

Usage: ``.venv/bin/python papers/distributed_secure_kernels/utils/paper_layout_results.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from distributed_secure_kernels.utils.plot_style import (  # noqa: E402
    INK,
    MUTED,
    OURS,
    PAPER,
    REFERENCE,
    RESULTS_DIR,
    style_axes,
)

COLUMNS = (
    ("classical_linear", "Centralised classical"),
    ("quantum_centralised", "Centralised quantum"),
    ("quantum_distributed", "Distributed quantum"),
)
ROWS = (
    ("wine", "Wine", "kernel_svm", "kSVM"),
    ("parkinsons", "Parkinson's", "kernel_svm", "kSVM"),
    ("parkinsons", "", "kernel_pca", "kPCA"),
    ("framingham", "Framingham", "kernel_svm", "kSVM"),
    ("framingham", "", "kernel_pca", "kPCA"),
)
# Paper Table 1: (value, fold std or None).
PAPER_TABLE1 = {
    ("wine", "kernel_svm"): ((0.9860, 0.0172), (0.8805, None), (0.8874, 0.0259)),
    ("parkinsons", "kernel_svm"): ((0.8196, 0.0644), (0.7875, None), (0.7983, 0.0798)),
    ("parkinsons", "kernel_pca"): ((0.7872, 0.0716), (0.7451, None), (0.7660, 0.0744)),
    ("framingham", "kernel_svm"): ((0.6788, 0.0108), (0.6308, None), (0.6340, 0.0143)),
    ("framingham", "kernel_pca"): ((0.6788, 0.0095), (0.6249, None), (0.6422, 0.0092)),
}
# Paper Fig. 3, from the authors' logs: group -> (classical, no noise, level 1, level 2).
PAPER_FIG3 = {
    "Heart Study\nkSVM": (0.6788, 0.6340, 0.5807, 0.5814),
    "Parkinson's\nkSVM": (0.8196, 0.7983, 0.7873, 0.6330),
    "Parkinson's\n5-kPCA": (0.7872, 0.6911, 0.7129, 0.6276),
    "Parkinson's\n6-kPCA": (0.7710, 0.7660, 0.7445, 0.6276),
    "Parkinson's\n7-kPCA": (0.7605, 0.7553, 0.7393, 0.6276),
}
FIG3_GROUPS = (
    ("framingham", "kernel_svm"),
    ("parkinsons", "kernel_svm"),
    ("parkinsons", "kernel_pca_5_components"),
    ("parkinsons", "kernel_pca_6_components"),
    ("parkinsons", "kernel_pca_7_components"),
)
# Ordered noise levels: one hue, dark (no noise) to light (1 %).
NOISE_COLORS = ("#1d4f91", "#2a78d6", "#8fb8ea")
SHOTS = (128, 256, 512, 1024)
PAPER_FIG4 = (0.71, 0.71, 0.74, 0.83)
PAPER_FIG4_CLASSICAL = 0.88


def load_rows() -> dict:
    sweeps = json.loads((RESULTS_DIR / "sweep_results.json").read_text())
    return {
        (sweep, row["candidate"], row["method"], row["model"]): row
        for sweep, content in sweeps.items()
        for row in content["rows"]
    }


def cell(row: dict, metric: str) -> str:
    stats = row[metric]
    return f"{stats['seed_mean']:.3f} ± {stats['seed_std']:.3f}"


def paper_cell(value: float, std: float | None) -> str:
    return f"{value:.3f}" if std is None else f"{value:.3f} ± {std:.3f}"


def table1(rows: dict) -> str:
    # Two rows per method, paper then ours, so values compare vertically.
    lines = [
        "| Dataset | Method | | " + " | ".join(label for _, label in COLUMNS) + " |",
        "|---|---|---|" + "---:|" * len(COLUMNS),
    ]
    for dataset, label, model, model_label in ROWS:
        ours_model = "kernel_pca_best_on_test_folds" if model == "kernel_pca" else model
        paper = [
            paper_cell(value, std) for value, std in PAPER_TABLE1[(dataset, model)]
        ]
        ours = [
            cell(
                rows[
                    (
                        "table1_original",
                        f"accuracy_{dataset}_original",
                        method,
                        ours_model,
                    )
                ],
                "accuracy",
            )
            for method, _ in COLUMNS
        ]
        lines.append(f"| {label} | {model_label} | paper | " + " | ".join(paper) + " |")
        lines.append("| | | ours | " + " | ".join(ours) + " |")
    return "\n".join(lines) + "\n"


def table1_bis(rows: dict) -> str:
    header = [
        "Centralised classical",
        "Centralised quantum",
        "Distributed quantum",
        "RBF SVM",
    ]
    lines = [
        "| Dataset | Method | " + " | ".join(header) + " |",
        "|---|---|" + "---:|" * len(header),
    ]
    for dataset, label, model, model_label in ROWS:
        ours_model = "kernel_pca_nested_cv" if model == "kernel_pca" else model
        cells = [
            cell(
                rows[
                    (
                        "table1_corrected",
                        f"accuracy_{dataset}_corrected",
                        method,
                        ours_model,
                    )
                ],
                "balanced_accuracy",
            )
            for method, _ in COLUMNS
        ]
        rbf = rows.get(
            (
                "fair_comparison",
                f"accuracy_{dataset}_fair_classical",
                "tuned_rbf",
                ours_model,
            )
        )
        cells.append(cell(rbf, "balanced_accuracy") if rbf else "—")
        lines.append(f"| {label} | {model_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def fig3(rows: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    style_axes(ax)
    width = 0.2
    series = [("classical_linear", "Classical")] + [
        (f"quantum_distributed_noise_level_{level}", name)
        for level, name in enumerate(
            ("Quantum, no noise", "Quantum, 0.1 %", "Quantum, 1 %")
        )
    ]
    for group_index, ((dataset, model), paper_values) in enumerate(
        zip(FIG3_GROUPS, PAPER_FIG3.values())
    ):
        for bar, ((method, name), paper_value) in enumerate(zip(series, paper_values)):
            row = rows[("noise_original", f"noise_{dataset}_original", method, model)]
            x = group_index + (bar - 1.5) * width
            ax.bar(
                x,
                row["accuracy"]["seed_mean"],
                width * 0.9,
                yerr=row["accuracy"]["seed_std"],
                color="white" if bar == 0 else NOISE_COLORS[bar - 1],
                edgecolor=REFERENCE if bar == 0 else NOISE_COLORS[bar - 1],
                hatch="///" if bar == 0 else None,
                ecolor=INK,
                capsize=2,
                label=name if group_index == 0 else None,
            )
            ax.scatter(
                x,
                paper_value,
                marker="D",
                s=22,
                color=PAPER,
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
                label="paper (authors' logs)"
                if group_index == 0 and bar == 0
                else None,
            )
    ax.set_xticks(range(len(FIG3_GROUPS)), list(PAPER_FIG3))
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("Accuracy (5-fold CV)", color=INK)
    ax.legend(
        frameon=False,
        labelcolor=INK,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=8.5,
    )
    fig.text(
        0.01,
        -0.07,
        "Bars: ours, mean ± std over 5 seeds; diamonds: the authors' logged values. "
        "Paper protocol, authors' depolarizing noise model.",
        fontsize=8,
        color=MUTED,
    )
    _save(fig, "fig3")


def fig4(rows: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    style_axes(ax)
    means, stds = [], []
    for shots in SHOTS:
        row = rows[
            (
                "shots_original",
                "shots_digits_original",
                f"quantum_distributed_{shots}_shots",
                "kernel_svm",
            )
        ]
        means.append(row["accuracy"]["seed_mean"])
        stds.append(row["accuracy"]["seed_std"])
    x = np.arange(len(SHOTS))
    ax.errorbar(x, means, yerr=stds, color=OURS, marker="o", capsize=3, label="ours")
    ax.plot(x, PAPER_FIG4, color=PAPER, marker="s", label="paper")
    classical = rows[
        ("shots_original", "shots_digits_original", "classical_linear", "kernel_svm")
    ]["accuracy"]["seed_mean"]
    ax.axhline(classical, color=OURS, linestyle=":", label="classical SVM, ours")
    ax.axhline(
        PAPER_FIG4_CLASSICAL, color=PAPER, linestyle=":", label="classical SVM, paper"
    )
    ax.set_xticks(x, [f"{s:,}" for s in SHOTS])
    ax.set_xlabel("Number of shots", color=INK)
    ax.set_ylabel("Accuracy (5-fold CV)", color=INK)
    ax.set_ylim(0.65, 0.95)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, ncol=2, loc="lower right")
    fig.text(
        0.01,
        -0.03,
        "Digits, 100 samples, 10 classes; ours: mean ± std over 5 seeds.",
        fontsize=8,
        color=MUTED,
    )
    _save(fig, "fig4")


MERLIN = "#c2185b"  # validated against OURS for colour-blind separation


def table1_merlin(rows: dict) -> str:
    """Table 1 settings: gate-model quantum columns vs the MerLin photonic swap test."""

    header = ["Centralised quantum", "Distributed quantum", "MerLin photonic"]
    lines = [
        "| Dataset | Method | " + " | ".join(header) + " |",
        "|---|---|" + "---:|" * len(header),
    ]
    for dataset, label, model, model_label in ROWS:
        ours_model = "kernel_pca_best_on_test_folds" if model == "kernel_pca" else model
        gate = [
            cell(
                rows[
                    (
                        "table1_original",
                        f"accuracy_{dataset}_original",
                        method,
                        ours_model,
                    )
                ],
                "accuracy",
            )
            for method in ("quantum_centralised", "quantum_distributed")
        ]
        merlin = rows[
            (
                "merlin_original",
                f"accuracy_{dataset}_merlin",
                "merlin_photonic_swap_test",
                ours_model,
            )
        ]
        cells = gate + [cell(merlin, "accuracy")]
        lines.append(f"| {label} | {model_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def fig4_merlin(rows: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    style_axes(ax)
    x = np.arange(len(SHOTS))
    for sweep, candidate, method, color, marker, name in (
        (
            "shots_original",
            "shots_digits_original",
            "quantum_distributed",
            OURS,
            "o",
            "distributed quantum (gate model)",
        ),
        (
            "merlin_original",
            "shots_digits_merlin",
            "merlin_photonic_swap_test",
            MERLIN,
            "s",
            "MerLin photonic",
        ),
    ):
        stats = [
            rows[(sweep, candidate, f"{method}_{shots}_shots", "kernel_svm")][
                "accuracy"
            ]
            for shots in SHOTS
        ]
        ax.errorbar(
            x,
            [s["seed_mean"] for s in stats],
            yerr=[s["seed_std"] for s in stats],
            color=color,
            marker=marker,
            capsize=3,
            label=name,
        )
    ax.set_xticks(x, [f"{s:,}" for s in SHOTS])
    ax.set_xlabel("Number of shots", color=INK)
    ax.set_ylabel("Accuracy (5-fold CV)", color=INK)
    ax.set_ylim(0.65, 0.9)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="lower right")
    fig.text(
        0.01,
        -0.03,
        "Digits, 100 samples, 10 classes; paper protocol; mean ± std over 5 seeds.",
        fontsize=8,
        color=MUTED,
    )
    _save(fig, "fig4_merlin")


def _save(fig, name: str) -> None:
    fig.patch.set_facecolor("#fcfcfb")
    target = RESULTS_DIR / f"{name}.png"
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(target)


def main() -> None:
    rows = load_rows()
    for name, text in (
        ("table1", table1(rows)),
        ("table1_bis", table1_bis(rows)),
        ("table1_merlin", table1_merlin(rows)),
    ):
        target = RESULTS_DIR / f"{name}.md"
        target.write_text(text, encoding="utf-8")
        print(target)
    fig3(rows)
    fig4(rows)
    fig4_merlin(rows)


if __name__ == "__main__":
    main()
