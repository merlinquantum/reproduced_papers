#!/usr/bin/env python3
"""Plot test RMSE and R^2 per candidate of a sweep (mean ± std and per-seed dots).

Reads ``report.json`` written by ``utils/report_sweep.py`` in the sweep
directory. Saves ``metric_comparison.png`` to the sweep directory and to
``papers/QK_LSTM/results/<sweep_name>/``.

Usage::

    .venv/bin/python papers/QK_LSTM/utils/plot_metric_comparison.py <SWEEP_DIR> --label "paper-accurate, 5 seeds"
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Validated default categorical palette (dataviz skill), fixed slot order by family.
FAMILY_COLORS = {
    "quantum kernel": "#2a78d6",  # slot 1 blue
    "RBF kernel": "#eb6834",  # slot 2 orange
    "LSTM": "#1baf7a",  # slot 3 aqua
    "photonic kernel (MerLin)": "#eda100",  # slot 4 yellow
    "naive / linear": "#6b6b66",  # neutral ink (non-learned references)
}
TEXT = "#2b2b29"
MUTED = "#6b6b66"
GRID = "#e4e3de"


def family(label: str) -> str:
    lowered = label.lower()
    if "photonic" in lowered:
        return "photonic kernel (MerLin)"
    if any(k in lowered for k in ("quantum", "qubit")) or label.startswith(
        ("D2_", "ANG_")
    ):
        return "quantum kernel"
    if "rbf" in lowered:
        return "RBF kernel"
    if "lstm" in lowered:
        return "LSTM"
    return "naive / linear"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="sweep directory, or a claim-table JSON from claim_statistics.py",
    )
    parser.add_argument(
        "--label", required=True, help="result-scope label shown in the title"
    )
    args = parser.parse_args(argv)
    source = args.sweep_dir.resolve()
    if source.is_file():
        # A claim table written by utils/claim_statistics.py.
        claims = json.loads(source.read_text())
        report = {
            "sweep_id": claims["spec"]["title"].split(" - ")[0],
            "paper": claims["spec"]["paper"],
            "rows": [
                {
                    "candidate": name,
                    "trainable_parameters": row["params"],
                    **{m: row[m] for m in ("rmse", "r2")},
                }
                for name, row in claims["table"].items()
            ],
        }
        output_dirs = [source.parent]
        figure_name = f"{source.name.removesuffix('.json')}.png"
    else:
        report = json.loads((source / "report.json").read_text())
        plan = json.loads((source / "plan.json").read_text())
        results_dir = PROJECT_DIR / "results" / plan["sweep_name"]
        results_dir.mkdir(parents=True, exist_ok=True)
        output_dirs = [source, results_dir]
        figure_name = "metric_comparison.png"
    rows = sorted(report["rows"], key=lambda r: r["rmse"]["mean"], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(rows) + 2.2), sharey=True)
    positions = range(len(rows))
    for axis, metric, title in (
        (axes[0], "rmse", "Test RMSE (AQI) - lower is better"),
        (axes[1], "r2", "Test R² - higher is better"),
    ):
        for y, row in zip(positions, rows):
            color = FAMILY_COLORS[family(row["candidate"])]
            stats = row[metric]
            axis.errorbar(
                stats["mean"],
                y,
                xerr=stats["std"],
                fmt="o",
                ms=8,
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=0,
                zorder=3,
            )
            axis.scatter(
                stats["values"],
                [y] * len(stats["values"]),
                s=14,
                color=color,
                alpha=0.45,
                edgecolors="none",
                zorder=2,
            )
        for name, values in report["paper"].items():
            axis.axvline(values[metric], color=MUTED, lw=1.5, ls="--", zorder=1)
            axis.text(
                values[metric],
                len(rows) - 0.4,
                name,
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color=MUTED,
            )
        # Keep paper reference values inside the axis with a margin.
        plotted = [v for row in rows for v in row[metric]["values"]]
        plotted += [values[metric] for values in report["paper"].values()]
        span = max(plotted) - min(plotted)
        axis.set_xlim(min(plotted) - 0.08 * span, max(plotted) + 0.05 * span)
        axis.set_title(title, fontsize=10, color=TEXT, loc="left")
        axis.grid(axis="x", color=GRID, lw=1)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_yticks(list(positions))
    axes[0].set_yticklabels(
        [
            f"{r['candidate']} ({'/'.join(map(str, r['trainable_parameters']))} p)"
            for r in rows
        ],
        color=TEXT,
        fontsize=9,
    )
    present = {family(row["candidate"]) for row in rows}
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, ms=8, label=f)
        for f, c in FAMILY_COLORS.items()
        if f in present
    ]
    handles.append(
        plt.Line2D([], [], ls="--", color=MUTED, label="paper value (single run)")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )
    title = (
        f"{report['sweep_id']} - {args.label}; dot = mean, bar = ±1 std, faint = seeds"
    )
    fig.suptitle(
        "\n".join(textwrap.wrap(title, width=150)),
        fontsize=10,
        color=TEXT,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))

    for directory in output_dirs:
        fig.savefig(directory / figure_name, dpi=150)
    print(output_dirs[-1] / figure_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
