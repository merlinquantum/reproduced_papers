"""Figure for sweep S2 (E11): context-preserving ansatz vs HEA and classical baselines.

Reads the sweep's ``sweep_status.json`` and ``metrics.json`` files and writes
``fig5_context_preserving.png`` to ``<sweep_dir>/figures`` and ``--results``.

Usage::

    python papers/QNN_STOCK/utils/plot_phase3_s2.py <sweep_dir> --results results/phase3_s2/figures
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR / "utils"))

from plot_phase3_s1 import INK, INK_2, LN2, SERIES, SURFACE, _save, _style, plt  # noqa: E402
from summarize_sweep import load_rows  # noqa: E402

SCOPE = "extension · substitute-data · seeds 42–46 · final epoch"


def plot_context_preserving(rows: list[dict], outputs: list[Path]) -> None:
    """Test log-loss (inference readout) and final fidelity per seed."""

    index = defaultdict(list)
    for row in rows:
        index[(row["candidate"], row["model"], row["asset"], row["metric"])].append(
            row["value"]
        )
    panels = (
        ("Raw daily moves", "", "e3_baselines_classical", "e3_qstl_original"),
        ("5-day moving average", "_ma5", "e3_baselines_ma5", "e3_qstl_ma5"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex="col")
    for column, (title, suffix, base, hea) in enumerate(panels):
        groups = (
            ("B2 logistic\n4 params", base, "B2_logistic"),
            ("B0 table\n8 params", base, "B0_frequency"),
            ("HEA (paper)\n32 params", hea, "QSTL"),
            ("Context-\npreserving\norder 1 (4)", f"e11_qstl_cc1{suffix}", "QSTL"),
            ("order 2 (7)", f"e11_qstl_cc2{suffix}", "QSTL"),
            ("order 3 (8)", f"e11_qstl_cc3{suffix}", "QSTL"),
        )
        for row_index, metric in enumerate(
            ("inference/test/log_loss", "final_fidelity")
        ):
            ax = axes[row_index][column]
            for position, (_, candidate, model) in enumerate(groups):
                for offset, (asset, color) in zip(
                    (-0.14, 0.14), (("AAPL", SERIES[0]), ("GOOGL", SERIES[1]))
                ):
                    values = index[(candidate, model, asset, metric)]
                    if not values:
                        continue
                    ax.scatter(
                        [position + offset] * len(values),
                        values,
                        s=36,
                        color=color,
                        edgecolor=SURFACE,
                        linewidth=1.5,
                        zorder=3,
                        label=asset if position == 2 else None,
                    )
                    ax.plot(
                        [position + offset - 0.1, position + offset + 0.1],
                        [statistics.fmean(values)] * 2,
                        color=INK,
                        linewidth=2,
                        zorder=4,
                    )
            ax.grid(axis="x", visible=False)
            if row_index == 0:
                ax.axhline(LN2, color=INK_2, linestyle=(0, (4, 3)), linewidth=1)
                ax.text(
                    -0.45,
                    LN2 + 0.01,
                    "ln 2 (coin flip)",
                    color=INK_2,
                    fontsize=9,
                    va="bottom",
                )
                ax.set_title(title, loc="left", fontsize=11)
            else:
                ax.set_ylim(-0.03, 1.05)
                ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=9)
    axes[0][0].set_ylabel(
        "Test log-loss, per-context inference\n(nats, lower is better)"
    )
    axes[1][0].set_ylabel("Final fidelity F\n(baselines: not applicable)")
    axes[1][1].legend(loc="lower left")
    fig.suptitle(
        "Keeping the context register fixes inference: order 1 matches the classical tables"
        " on MA5; larger orders often fail to converge",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        -0.01,
        f"Dots = seeds, black tick = mean. {SCOPE}.",
        color=INK_2,
        fontsize=8,
    )
    _save(fig, "fig5_context_preserving", outputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    _style()
    plot_context_preserving(
        load_rows(args.sweep_dir), [args.sweep_dir / "figures", args.results]
    )
    print(f"figure written to {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
