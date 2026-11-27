"""Figures for sweep S1: readout gap, context retention, training curves, paired differences.

Reads only the sweep's structured artifacts (``sweep_status.json``,
``metrics.json``, ``losses.csv``, ``params.npz``) and writes PNG figures to
``<sweep_dir>/figures`` and ``--results``.

Usage::

    python papers/QNN_STOCK/utils/plot_phase3_s1.py <sweep_dir> --results results/phase3_s1/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR.parents[1]))
sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(PAPER_DIR / "utils"))

from context_retention import retention_rows  # noqa: E402
from summarize_sweep import load_rows, paired  # noqa: E402

# Reference categorical palette (fixed order) and text tokens, light mode.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SCOPE = "substitute-data · seeds 42–46 · final epoch"
LN2 = math.log(2.0)
RETENTION_LABELS = {
    "e3_qstl_original": "QSTL 4L\nraw",
    "e3_qstl_ma5": "QSTL 4L\nMA5",
    "e3_qstl_hea_loader": "QSTL 4L\nHEA loader",
    "e3_qstl_l2": "QSTL 2L\nraw",
    "e3_qstl_l2_ma5": "QSTL 2L\nMA5",
    "e3_qstl_l2_e250": "QSTL 2L\n250 ep raw",
    "e3_qstl_l2_e250_ma5": "QSTL 2L\n250 ep MA5",
    "e6_qstl_mse_original": "QSTL 4L\nMSE loss",
    "e8_qstl_d4": "QSTL d=4\nT=2 (1/16)",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _save(fig, name: str, outputs: list[Path]) -> None:
    for out in outputs:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_readout_gap(rows: list[dict], outputs: list[Path]) -> None:
    """Test log-loss per seed: classical tables vs the two quantum readouts."""

    index = defaultdict(list)
    for row in rows:
        index[(row["candidate"], row["model"], row["asset"], row["metric"])].append(
            row["value"]
        )
    panels = (
        (
            "Raw daily moves",
            "e3_baselines_classical",
            "e3_qstl_original",
            "e4_qmtl_original",
        ),
        ("5-day moving average", "e3_baselines_ma5", "e3_qstl_ma5", "e4_qmtl_ma5"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (title, base, qstl, qmtl) in zip(axes, panels):
        groups = (
            ("B0 frequency\ntable", base, "B0_frequency", "inference"),
            ("B2 logistic", base, "B2_logistic", "inference"),
            ("QSTL\njoint readout", qstl, "QSTL", "output"),
            ("QSTL\nper-context\ninference", qstl, "QSTL", "inference"),
            ("QMTL\nper-context\ninference", qmtl, "QMTL", "inference"),
        )
        for position, (_, candidate, model, readout) in enumerate(groups):
            for offset, (asset, color) in zip(
                (-0.14, 0.14), (("AAPL", SERIES[0]), ("GOOGL", SERIES[1]))
            ):
                values = index[(candidate, model, asset, f"{readout}/test/log_loss")]
                xs = [position + offset] * len(values)
                ax.scatter(
                    xs,
                    values,
                    s=36,
                    color=color,
                    edgecolor=SURFACE,
                    linewidth=1.5,
                    zorder=3,
                    label=asset if position == 0 else None,
                )
                ax.plot(
                    [position + offset - 0.1, position + offset + 0.1],
                    [statistics.fmean(values)] * 2,
                    color=INK,
                    linewidth=2,
                    zorder=4,
                )
        ax.axhline(LN2, color=INK_2, linestyle=(0, (4, 3)), linewidth=1)
        ax.text(
            -0.45,
            LN2 + 0.005,
            "ln 2 (coin flip)",
            color=INK_2,
            va="bottom",
            ha="left",
            fontsize=9,
        )
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test log-loss (nats, lower is better)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), title=None)
    fig.suptitle(
        "Same trained circuit, two readouts: the joint state matches the frequency table,"
        " per-context inference does not",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        -0.1,
        f"Dots = seeds, black tick = mean. {SCOPE}.",
        color=INK_2,
        fontsize=8,
    )
    _save(fig, "fig1_readout_gap", outputs)


def plot_context_retention(retention: list[dict], outputs: list[Path]) -> None:
    """P(context register unchanged | input |c>|0>) per config, all seeds and contexts."""

    order = [
        "e3_qstl_original",
        "e3_qstl_ma5",
        "e3_qstl_hea_loader",
        "e3_qstl_l2",
        "e3_qstl_l2_ma5",
        "e3_qstl_l2_e250",
        "e3_qstl_l2_e250_ma5",
        "e6_qstl_mse_original",
        "e8_qstl_d4",
    ]
    by_config = defaultdict(list)
    for row in retention:
        by_config[row["candidate"]].append(float(row["p_context_kept"]))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for position, config in enumerate(order):
        values = by_config[config]
        jitter = [position + ((i % 9) - 4) * 0.035 for i in range(len(values))]
        ax.scatter(
            jitter,
            values,
            s=12,
            color=SERIES[0],
            alpha=0.45,
            edgecolor="none",
            zorder=2,
        )
        ax.plot(
            [position - 0.25, position + 0.25],
            [statistics.fmean(values)] * 2,
            color=INK,
            linewidth=2,
            zorder=3,
        )
        chance = 1 / 16 if config == "e8_qstl_d4" else 1 / 8
        ax.plot(
            [position - 0.35, position + 0.35],
            [chance] * 2,
            color=SERIES[1],
            linewidth=1.5,
            linestyle=(0, (3, 2)),
            zorder=3,
        )
    ax.axhline(1.0, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(
        len(order) - 0.6,
        1.0,
        "Eq. 5 assumption: context kept with probability 1",
        ha="right",
        va="bottom",
        color=INK_2,
        fontsize=9,
    )
    ax.plot([], [], color=INK, linewidth=2, label="mean over seeds and contexts")
    ax.plot(
        [],
        [],
        color=SERIES[1],
        linewidth=1.5,
        linestyle=(0, (3, 2)),
        label="chance (1 / number of contexts)",
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93))
    ax.set_xticks(range(len(order)), [RETENTION_LABELS[c] for c in order], fontsize=8.5)
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel("P(context unchanged)")
    ax.grid(axis="x", visible=False)
    ax.set_title(
        "Trained circuits scramble the context register: retention sits at chance",
        loc="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        -0.16,
        f"Each dot = one (seed, context) for one asset. {SCOPE}.",
        color=INK_2,
        fontsize=8,
    )
    _save(fig, "fig2_context_retention", outputs)


def _fidelity_curves(
    sweep_dir: Path, status: dict, config: str, column: str
) -> list[list[tuple[int, float]]]:
    curves = []
    for run in status["runs"]:
        if run["candidate"]["config"] != config:
            continue
        losses = Path(sweep_dir / run["run_dir"] / "losses.csv")
        with losses.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            points = []
            for row in reader:
                if config.startswith("e4") or row["stage"].endswith("_AAPL"):
                    if row.get(column):
                        points.append((int(row["epoch"]), float(row[column])))
        # QSTL runs log one stage per asset; keep the AAPL stage only.
        curves.append(points)
    return curves


def plot_training_curves(sweep_dir: Path, outputs: list[Path]) -> None:
    """Exact fidelity per epoch on AAPL raw data, mean and range over seeds."""

    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    series = (
        ("QSTL 4 layers (32 params), 3000 epochs", "e3_qstl_original", "fidelity"),
        ("QSTL 2 layers (16), 3000 epochs", "e3_qstl_l2", "fidelity"),
        ("QSTL 2 layers (16), 250 epochs", "e3_qstl_l2_e250", "fidelity"),
        (
            "QMTL 1+1 layers (16 per asset), AAPL then GOOGL",
            "e4_qmtl_original",
            "fidelity_AAPL",
        ),
        (
            "QSTL 4 layers, MSE loss, lr 1e-5, 10,000 epochs",
            "e6_qstl_mse_original",
            "fidelity",
        ),
    )
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for color, (label, config, column) in zip(SERIES, series):
        curves = [c for c in _fidelity_curves(sweep_dir, status, config, column) if c]
        length = min(len(c) for c in curves)
        epochs = [p[0] for p in curves[0][:length]]
        stacked = [[c[i][1] for c in curves] for i in range(length)]
        ax.fill_between(
            epochs,
            [min(v) for v in stacked],
            [max(v) for v in stacked],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            epochs,
            [statistics.fmean(v) for v in stacked],
            color=color,
            linewidth=2,
            label=label,
        )
    ax.axvline(250, color=INK_2, linewidth=1, linestyle=(0, (2, 2)))
    ax.text(
        260,
        0.5,
        "QMTL switches to GOOGL;\nshared block updates,\nAAPL fidelity drops",
        color=INK_2,
        fontsize=8.5,
        va="center",
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("SPSA epoch (log scale)")
    ax.set_ylabel("Fidelity to joint target, F")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        "Training: only the long 4- and 2-layer QSTL runs reach F ≈ 1 (AAPL, raw)",
        loc="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        -0.04,
        "Line = mean over seeds, band = min–max. Exact F from the statevector. substitute-data · seeds 42–46.",
        color=INK_2,
        fontsize=8,
    )
    _save(fig, "fig3_training_curves", outputs)


def plot_paired(rows: list[dict], outputs: list[Path]) -> None:
    """Paired-by-seed differences with 95 % intervals for the protocol comparisons."""

    table = paired(rows)
    labels = [
        ("A_raw", "A: 2 vs 4 layers (raw)"),
        ("A_ma5", "A: 2 vs 4 layers (MA5)"),
        ("B_raw", "B: QMTL vs 2-layer QSTL, 250 ep (raw)"),
        ("B_ma5", "B: QMTL vs 2-layer QSTL, 250 ep (MA5)"),
        ("paper_STL_vs_MTL_raw", "Paper: QMTL vs QSTL (raw)"),
        ("paper_STL_vs_MTL_ma5", "Paper: QMTL vs QSTL (MA5)"),
        ("loader_hea_vs_exact", "HEA loader vs exact"),
        ("mse_vs_fidelity", "MSE vs fidelity loss"),
        ("QSTL_vs_B0_raw", "QSTL vs B0 (raw)"),
        ("QSTL_vs_B0_ma5", "QSTL vs B0 (MA5)"),
        ("QMTL_vs_B4_raw", "QMTL vs B4 (raw)"),
        ("QMTL_vs_B4_ma5", "QMTL vs B4 (MA5)"),
    ]
    metrics = (
        ("inference/test/log_loss", "Test log-loss difference (primary)"),
        ("output/train/kl_joint", "Train joint KL difference (paper-style)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        for row_index, (key, _) in enumerate(labels):
            for offset, (asset, color) in zip(
                (-0.15, 0.15), (("AAPL", SERIES[0]), ("GOOGL", SERIES[1]))
            ):
                match = [
                    r
                    for r in table
                    if r["comparison"] == key
                    and r["asset"] == asset
                    and r["metric"] == metric
                ]
                if not match:
                    continue
                r = match[0]
                y = row_index + offset
                ax.plot(
                    [r["ci95_low"], r["ci95_high"]], [y, y], color=color, linewidth=2
                )
                ax.scatter(
                    [r["mean_diff"]],
                    [y],
                    s=40,
                    color=color,
                    edgecolor=SURFACE,
                    linewidth=1.5,
                    zorder=3,
                    label=asset if row_index == 0 else None,
                )
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_xscale("symlog", linthresh=0.05)
        ticks = [-1, -0.1, 0, 0.1, 1]
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("candidate − reference (> 0: candidate worse)")
    axes[0].set_yticks(range(len(labels)), [label for _, label in labels], fontsize=9)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    axes[1].text(
        0.97,
        0.22,
        "baselines have no joint\nquantum state: no train\njoint KL",
        transform=axes[1].transAxes,
        ha="right",
        color=INK_2,
        fontsize=8.5,
    )
    fig.suptitle(
        "Paired differences, mean and 95 % t interval over 5 seeds; an interval crossing 0 is not distinguishable",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        -0.02,
        f"Symmetric-log x axis (linear within ±0.05). {SCOPE}.",
        color=INK_2,
        fontsize=8,
    )
    _save(fig, "fig4_paired_differences", outputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    _style()
    outputs = [args.sweep_dir / "figures", args.results]
    rows = load_rows(args.sweep_dir)
    plot_readout_gap(rows, outputs)
    plot_context_retention(retention_rows(args.sweep_dir), outputs)
    plot_training_curves(args.sweep_dir, outputs)
    plot_paired(rows, outputs)
    print(f"figures written to {', '.join(str(o) for o in outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
