"""QLIKE of every Task 3 model relative to the angle QSVR, one panel per ticker.

Reads the table written by ``utils/stats_volatility.py`` (``--table``) and
plots ``QLIKE(model) - QLIKE(qsvr_angle)`` (negative = better than the QSVR),
with filled markers where the Diebold-Mariano QLIKE test has p < 0.05 and
hollow markers otherwise. Saves ``results/<--output>.png``.

Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/plot_volatility_qlike.py \\
        --table papers/QML_financial_benchmark/results/e6_with_garch_sigma2.csv \\
        --label "E6, 5+5 lags, angle QSVR 10 qubits (single Optuna seed)" --output e6_qlike_vs_qsvr
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
QUANTUM, CLASSICAL, ECONOMETRIC = "#2a78d6", "#eb6834", "#1baf7a"
MODEL_ORDER = [
    ("garch", "GARCH completed (D20)", ECONOMETRIC, "D"),
    ("garch_sigma2", "GARCH σ² only (paper)", ECONOMETRIC, "d"),
    ("har", "HAR-RV", ECONOMETRIC, "^"),
    ("persistence", "Persistence", TEXT_SECONDARY, "v"),
    ("svr_rbf", "SVR-RBF", CLASSICAL, "o"),
    ("svr_rbf_wide", "SVR-RBF wide γ", CLASSICAL, "s"),
    ("svr_linear", "SVR-linear", CLASSICAL, "<"),
    ("svr_poly", "SVR-poly", CLASSICAL, ">"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--reference", default="qsvr_angle")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--clip",
        type=float,
        default=0.25,
        help="Clip |difference| for display; clipped points are annotated. Default 0.25.",
    )
    args = parser.parse_args(argv)

    table = pd.read_csv(args.table)
    tickers = sorted(table["ticker"].unique())
    figure, axes = plt.subplots(
        1,
        len(tickers),
        figsize=(3.6 * len(tickers), 4.4),
        sharey=True,
        facecolor=SURFACE,
    )
    for axis, ticker in zip(axes, tickers):
        rows = table[table["ticker"] == ticker].set_index("model")
        reference = rows.loc[args.reference, "qlike"]
        axis.set_facecolor(SURFACE)
        axis.axvline(0.0, color=QUANTUM, linewidth=1.5, zorder=1)
        axis.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for position, (model, _, color, marker) in enumerate(MODEL_ORDER):
            if model not in rows.index:
                continue
            difference = rows.loc[model, "qlike"] - reference
            shown = max(-args.clip, min(args.clip, difference))
            significant = rows.loc[model, "dm_qlike_p"] < 0.05
            axis.scatter(
                shown,
                position,
                s=58,
                marker=marker,
                color=color if significant else "none",
                edgecolor=color,
                linewidth=1.8,
                zorder=3,
            )
            if shown != difference:
                axis.annotate(
                    f"{difference:+.2f}",
                    (shown, position),
                    xytext=(-4 if shown > 0 else 4, 7),
                    textcoords="offset points",
                    ha="right" if shown > 0 else "left",
                    fontsize=7.5,
                    color=TEXT_SECONDARY,
                )
        axis.set_yticks(
            range(len(MODEL_ORDER)), [name for _, name, *_ in MODEL_ORDER], fontsize=8.5
        )
        axis.invert_yaxis()
        axis.set_xlim(-args.clip * 1.1, args.clip * 1.1)
        axis.set_title(
            f"{ticker}  (QSVR QLIKE {reference:.3f})",
            fontsize=10,
            color=TEXT_PRIMARY,
            loc="left",
        )
        axis.set_xlabel("QLIKE − QSVR QLIKE", color=TEXT_SECONDARY, fontsize=9)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    figure.suptitle(args.label, x=0.01, ha="left", fontsize=10.5, color=TEXT_PRIMARY)
    figure.text(
        0.01,
        0.01,
        "Left of the blue line: better than the angle QSVR. Filled: Diebold–Mariano QLIKE p < 0.05; hollow: not significant. "
        "Test 2007-11 → 2024-12.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    figure.savefig(results_dir / f"{args.output}.png", dpi=160, facecolor=SURFACE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
