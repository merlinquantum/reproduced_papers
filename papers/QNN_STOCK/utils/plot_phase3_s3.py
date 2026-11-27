"""Sweep S3 (step F): test log-loss against training-set size.

Reads the sweep's ``sweep_status.json`` and ``metrics.json`` files (full
precision) and writes ``size_curves.csv`` (mean and std per size),
``size_paired.csv`` (paired-by-seed differences with 95 % t intervals) and
``fig6_train_size.png`` to ``--out`` and ``<sweep_dir>/figures``.

Usage::

    python papers/QNN_STOCK/utils/plot_phase3_s3.py <sweep_dir> --out results/phase3_s3
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR / "utils"))

from plot_phase3_s1 import INK, INK_2, LN2, SERIES, _save, _style, plt  # noqa: E402
from summarize_sweep import T_975, load_rows  # noqa: E402

METRIC = "inference/test/log_loss"
SCOPE = "substitute-data · seeds 42–51 · final epoch · context-preserving circuits are an extension"
FULL = 8023  # AAPL full training split; used only to place "full" on the x axis
# (label, config stem, model, color index)
MODELS = (
    ("B2 logistic (4 params)", "e3_baselines_classical", "B2_logistic", 0),
    ("B0 frequency table (8)", "e3_baselines_classical", "B0_frequency", 1),
    ("Context-preserving order 1 (4)", "e11_qstl_cc1", "QSTL", 2),
    ("Context-preserving order 3 (8)", "e11_qstl_cc3", "QSTL", 3),
    ("HEA, paper circuit (32)", "e3_qstl_original", "QSTL", 4),
)
PAIRS = (
    ("CC1 vs B2", ("e11_qstl_cc1", "QSTL"), ("e3_baselines_classical", "B2_logistic")),
    ("CC3 vs B0", ("e11_qstl_cc3", "QSTL"), ("e3_baselines_classical", "B0_frequency")),
    ("CC1 vs B0", ("e11_qstl_cc1", "QSTL"), ("e3_baselines_classical", "B0_frequency")),
    (
        "B2 vs B0",
        ("e3_baselines_classical", "B2_logistic"),
        ("e3_baselines_classical", "B0_frequency"),
    ),
)


def _split(candidate: str) -> tuple[str, str, str]:
    """``e11_qstl_cc1_ma5__n50`` -> (``e11_qstl_cc1``, ``ma5``, ``n50``)."""

    stem, size = candidate.split("__")
    data = "ma5" if stem.endswith("_ma5") else "raw"
    # The MA5 configs of the paper HEA and of the baselines have no raw-style
    # suffix to strip (e3_qstl_ma5, e3_baselines_ma5), so map them explicitly.
    raw_names = {
        "e3_qstl_ma5": "e3_qstl_original",
        "e3_baselines_ma5": "e3_baselines_classical",
    }
    stem = raw_names.get(stem, stem.removesuffix("_ma5"))
    return stem, data, size


def _size(label: str) -> int:
    return FULL if label == "nfull" else int(label[1:])


def collect(rows: list[dict]) -> dict:
    values = defaultdict(dict)
    for row in rows:
        if row["metric"] != METRIC:
            continue
        stem, data, size = _split(row["candidate"])
        values[(stem, row["model"], data, row["asset"], _size(size))][row["seed"]] = (
            row["value"]
        )
    return values


def tables(values: dict) -> tuple[list[dict], list[dict]]:
    curves = []
    for (stem, model, data, asset, size), by_seed in sorted(values.items()):
        v = list(by_seed.values())
        curves.append(
            {
                "model": f"{stem}:{model}",
                "data": data,
                "asset": asset,
                "train_size": size,
                "n_seeds": len(v),
                "mean": statistics.fmean(v),
                "std": statistics.stdev(v),
                "median": statistics.median(v),
            }
        )
    paired = []
    sizes = sorted({k[4] for k in values})
    for label, cand, ref in PAIRS:
        for data in ("raw", "ma5"):
            for asset in ("AAPL", "GOOGL"):
                for size in sizes:
                    a = values.get((*cand, data, asset, size), {})
                    b = values.get((*ref, data, asset, size), {})
                    seeds = sorted(set(a) & set(b))
                    if len(seeds) < 2:
                        continue
                    diffs = [a[s] - b[s] for s in seeds]
                    mean, std = statistics.fmean(diffs), statistics.stdev(diffs)
                    half = T_975[len(diffs) - 1] * std / math.sqrt(len(diffs))
                    low, high = mean - half, mean + half
                    paired.append(
                        {
                            "comparison": label,
                            "data": data,
                            "asset": asset,
                            "train_size": size,
                            "n_pairs": len(diffs),
                            "mean_diff": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "verdict": "not distinguishable"
                            if low <= 0 <= high
                            else (
                                "candidate lower" if high < 0 else "candidate higher"
                            ),
                        }
                    )
    return curves, paired


def plot(values: dict, outputs: list[Path]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row_index, data in enumerate(("raw", "ma5")):
        for col_index, asset in enumerate(("AAPL", "GOOGL")):
            ax = axes[row_index][col_index]
            for label, stem, model, color in MODELS:
                points = sorted(
                    (
                        size,
                        statistics.median(by_seed.values()),
                        min(by_seed.values()),
                        max(by_seed.values()),
                    )
                    for (s, m, d, a, size), by_seed in values.items()
                    if (s, m, d, a) == (stem, model, data, asset)
                )
                xs = [p[0] for p in points]
                ax.plot(
                    xs,
                    [p[1] for p in points],
                    color=SERIES[color],
                    linewidth=2,
                    marker="o",
                    markersize=5,
                    label=label,
                )
                ax.fill_between(
                    xs,
                    [p[2] for p in points],
                    [p[3] for p in points],
                    color=SERIES[color],
                    alpha=0.10,
                    linewidth=0,
                )
            ax.axhline(LN2, color=INK_2, linestyle=(0, (4, 3)), linewidth=1)
            ax.set_xscale("log")
            ax.set_ylim(0.4, 1.4)
            ax.set_title(
                f"{asset}, {'raw daily moves' if data == 'raw' else '5-day moving average'}",
                loc="left",
                fontsize=11,
            )
            if row_index == 1:
                ax.set_xlabel("Training windows (log scale; rightmost = full split)")
            if col_index == 0:
                ax.set_ylabel("Test log-loss, per-context inference")
    axes[0][0].text(
        26, LN2 + 0.01, "ln 2 (coin flip)", color=INK_2, fontsize=9, va="bottom"
    )
    axes[0][1].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(
        "Step F: test log-loss vs training-set size (line = median over 10 seeds, band = min–max;"
        " y clipped at 1.4)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.text(0.01, -0.01, SCOPE + ".", color=INK_2, fontsize=8)
    _save(fig, "fig6_train_size", outputs)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    _style()
    values = collect(load_rows(args.sweep_dir))
    curves, paired = tables(values)
    args.out.mkdir(parents=True, exist_ok=True)
    _write(args.out / "size_curves.csv", curves)
    _write(args.out / "size_paired.csv", paired)
    plot(values, [args.out / "figures", args.sweep_dir / "figures"])
    print(f"{len(curves)} curve rows, {len(paired)} paired rows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
