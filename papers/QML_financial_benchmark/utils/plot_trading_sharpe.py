"""Per-seed test Sharpe of the E5 selections per regime, with buy-and-hold and the paper's values.

Reads the per-seed table written by ``utils/stats_trading.py``
(``--per-seed``) and plots one panel per regime for one selection rule:
QLSTM, LSTM and parameter-matched LSTM (dots per seed, bar = mean), the
buy-and-hold Sharpe as a dashed line, and the paper's QLSTM/LSTM Sharpe from
Table XII as hollow diamonds. Saves ``results/<--output>.png``.

Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/plot_trading_sharpe.py \\
        --per-seed papers/QML_financial_benchmark/results/e5_trading_per_seed.csv \\
        --rule auc --output e5_sharpe_auc_rule
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Table XII Sharpe ratios (QLSTM, LSTM) per regime.
PAPER_SHARPE = {
    "F1": (0.3865, -0.0477),
    "F2": (1.2585, 0.4966),
    "F3": (0.7659, 0.8947),
    "F4": (-0.6936, 1.5228),
}
REGIME_LABELS = {
    "F1": "F1 2008–09",
    "F2": "F2 2018–19",
    "F3": "F3 2020–21",
    "F4": "F4 2022–24",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-seed", type=Path, required=True)
    parser.add_argument("--rule", choices=["auc", "tau_sharpe"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    table = pd.read_csv(args.per_seed)
    models = [
        (f"{args.rule}:qlstm", "QLSTM", "#2a78d6", "o"),
        (f"{args.rule}:lstm", "LSTM", "#eb6834", "s"),
        (
            f"{args.rule}:lstm_parameter_matched",
            "LSTM\n(param-matched)",
            "#1baf7a",
            "^",
        ),
    ]
    regimes = sorted(table["regime"].unique())
    figure, axes = plt.subplots(
        1,
        len(regimes),
        figsize=(3.4 * len(regimes), 4.4),
        sharey=True,
        facecolor=SURFACE,
    )
    generator = np.random.default_rng(0)
    for axis, regime in zip(np.atleast_1d(axes), regimes):
        rows = table[table["regime"] == regime]
        axis.set_facecolor(SURFACE)
        axis.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        axis.axhline(0.0, color=GRID, linewidth=1.2, zorder=0)
        buy_and_hold = rows[rows["model"] == "buy_and_hold"]["sharpe"].iloc[0]
        axis.axhline(
            buy_and_hold,
            color=TEXT_SECONDARY,
            linewidth=1,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
        axis.annotate(
            f"buy & hold {buy_and_hold:.2f}",
            (2.45, buy_and_hold),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            fontsize=7.5,
            color=TEXT_SECONDARY,
        )
        for position, (key, _, color, marker) in enumerate(models):
            values = rows[rows["model"] == key]["sharpe"].to_numpy()
            axis.scatter(
                position + generator.uniform(-0.12, 0.12, len(values)),
                values,
                s=46,
                color=color,
                marker=marker,
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
            )
            axis.hlines(
                values.mean(),
                position - 0.28,
                position + 0.28,
                color=TEXT_PRIMARY,
                linewidth=2,
                zorder=4,
            )
        for position, paper_value in enumerate(PAPER_SHARPE[regime]):
            axis.scatter(
                position + 0.36,
                paper_value,
                s=70,
                marker="D",
                facecolor="none",
                edgecolor=models[position][2],
                linewidth=2,
                zorder=5,
            )
        axis.set_xticks(
            range(len(models)), [name for _, name, *_ in models], fontsize=8.5
        )
        axis.set_title(
            REGIME_LABELS.get(regime, regime),
            fontsize=10.5,
            color=TEXT_PRIMARY,
            loc="left",
        )
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Test Sharpe (whole regime)")
    rule_text = "model-select AUC" if args.rule == "auc" else "τ-slice Sharpe"
    figure.suptitle(
        f"E5 trading: selection by {rule_text} on seed 0, retrained on seeds 1–4 (partial, 5 seeds)",
        x=0.01,
        ha="left",
        fontsize=10.5,
        color=TEXT_PRIMARY,
    )
    figure.text(
        0.01,
        0.01,
        "Dots: one seed each. Bar: seed mean. Hollow diamond: paper (Table XII). Dashed line: buy-and-hold.",
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
