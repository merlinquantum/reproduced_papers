"""Forest plot of quantum - fair-control differences per setting (Phase 5).

Reads ``results/phase5_differences.csv`` (from ``utils/compare_fair.py``) and
draws three panels:

- Task 1: test-AUC difference QNN (and MerLin photonic QNN) - parameter-matched
  ANN, seed-level mean with 95% t-interval;
- Task 2: test-Sharpe difference QLSTM - parameter-matched LSTM per regime;
- Task 3: QLIKE difference angle QSVR (and photonic kernel) - strongest fair
  control (GARCH completed with the known returns) and - SVR-RBF; DM p < 0.05
  shown filled. Single tuning seed, so no interval.

Positive = quantum better in panels 1-2; negative = quantum better in panel 3.
Saves ``results/<--output>.png``.

Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/plot_phase5_forest.py --output phase5_forest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
QUBIT, PHOTONIC, SECOND = "#2a78d6", "#1baf7a", "#eb6834"


def interval_panel(axis, rows: pd.DataFrame, xlabel: str, title: str) -> None:
    rows = rows.reset_index(drop=True)
    for position, row in rows.iterrows():
        photonic = "photonic" in row["model"]
        color = PHOTONIC if photonic else QUBIT
        significant = row["p_value"] < 0.05
        axis.plot(
            [row["ci_low"], row["ci_high"]],
            [position, position],
            color=color,
            linewidth=2,
            zorder=2,
        )
        axis.scatter(
            row["difference"],
            position,
            s=52,
            marker="s" if photonic else "o",
            zorder=3,
            facecolor=color if significant else SURFACE,
            edgecolor=color,
            linewidth=1.8,
        )
    axis.axvline(0, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    axis.set_yticks(
        range(len(rows)),
        [
            f"{r.setting}{' · photonic' if 'photonic' in r.model else ''}"
            for r in rows.itertuples()
        ],
        fontsize=8,
    )
    axis.invert_yaxis()
    axis.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    axis.set_title(title, loc="left", fontsize=10, color=TEXT_PRIMARY)
    axis.grid(axis="x", color=GRID, linewidth=0.8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    differences = pd.read_csv(PROJECT_DIR / "results" / "phase5_differences.csv")

    task1 = differences[
        differences["comparison"] == "leak_free:qnn - leak_free:ann_parameter_matched"
    ]
    task2 = differences[
        differences["comparison"] == "auc:qlstm - auc:lstm_parameter_matched (sharpe)"
    ]
    task3 = differences[
        differences["comparison"].str.contains(
            r"QLIKE\((?:garch|svr_rbf)\)", regex=True
        )
    ]
    photonic3 = differences[differences["comparison"].str.contains("photonic_kernel")]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": GRID,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6.4),
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.25, 1, 1]},
    )
    for axis in axes:
        axis.set_facecolor(SURFACE)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    interval_panel(
        axes[0],
        task1,
        "Δ test AUC: QNN − parameter-matched ANN",
        "Task 1 · direction (5 or 3 seeds)",
    )
    interval_panel(
        axes[1],
        task2,
        "Δ test Sharpe: QLSTM − parameter-matched LSTM",
        "Task 2 · trading (partial, 5 seeds)",
    )

    axis = axes[2]
    rows = []
    for ticker in ["RV GSPC", "RV AAPL", "RV GARAN", "RV KCHOL"]:
        for _, row in task3[task3["setting"] == ticker].iterrows():
            control = (
                "GARCH completed" if "QLIKE(garch)" in row["comparison"] else "SVR-RBF"
            )
            rows.append(
                (
                    f"{ticker[3:]} · QSVR vs {control}",
                    row["difference"],
                    row["p_value"],
                    QUBIT if control == "SVR-RBF" else SECOND,
                    "o",
                )
            )
        for _, row in photonic3[photonic3["setting"] == ticker].iterrows():
            # Stored as QSVR - photonic; flip to photonic - QSVR so negative = photonic better.
            rows.append(
                (
                    f"{ticker[3:]} · photonic vs QSVR",
                    -row["difference"],
                    row["p_value"],
                    PHOTONIC,
                    "s",
                )
            )
    for position, (_, difference, p_value, color, marker) in enumerate(rows):
        axis.scatter(
            difference,
            position,
            s=52,
            marker=marker,
            facecolor=color if p_value < 0.05 else SURFACE,
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
    axis.axvline(0, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    axis.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    axis.invert_yaxis()
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_xlabel(
        "Δ QLIKE (negative = first model better)", color=TEXT_SECONDARY, fontsize=9
    )
    axis.set_title(
        "Task 3 · volatility (single tuning seed)",
        loc="left",
        fontsize=10,
        color=TEXT_PRIMARY,
    )

    figure.text(
        0.01,
        0.01,
        "Local runs only (paper values are single runs, not shown). Filled: p < 0.05 (seed-level t-test; Diebold–Mariano in Task 3); hollow: tie. "
        "Lines: 95% CI over seeds. Circles: paper's qubit model; squares: MerLin photonic adaptation.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(
        PROJECT_DIR / "results" / f"{args.output}.png", dpi=160, facecolor=SURFACE
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
