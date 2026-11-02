#!/usr/bin/env python3
"""Fig. 3(c, d) analogue: observed vs predicted AQI over the test window.

For each requested candidate of a sweep, picks the seed whose test RMSE is the
median over seeds (so the panel is representative, not the best run), reads
its ``predictions.csv``, and plots observed vs predicted test AQI. Panels share
the y-axis. Saves ``test_predictions.png`` to the sweep directory and to
``papers/QK_LSTM/results/<sweep_name>/``.

Usage::

    .venv/bin/python papers/QK_LSTM/utils/plot_test_predictions.py <SWEEP_DIR> \
        --candidates B0_persistence B1_lstm_1873 QB_quantum_209 --label "paper-accurate"
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
OBSERVED = "#2b2b29"
PREDICTED = "#2a78d6"
MUTED = "#6b6b66"
GRID = "#e4e3de"


def _median_seed_run(
    sweep_dir: Path, status: dict, candidate: str
) -> tuple[dict, dict]:
    runs = [
        r
        for r in status["runs"]
        if r["candidate"] == candidate and r["status"] == "COMPLETED"
    ]
    if not runs:
        raise ValueError(f"No completed runs for {candidate}")
    scored = []
    for run in runs:
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        scored.append(
            (metrics["splits"]["test"]["observed_targets"]["rmse"], run, metrics)
        )
    scored.sort(key=lambda item: item[0])
    median_rmse = statistics.median_low([s[0] for s in scored])
    _, run, metrics = next(item for item in scored if item[0] == median_rmse)
    return run, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--split",
        choices=["test", "all"],
        default="test",
        help="test window only (Fig. 3c,d) or the full 2015-2020 series (Fig. 3a,b)",
    )
    args = parser.parse_args(argv)
    sweep_dir = args.sweep_dir.resolve()
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    plan = json.loads((sweep_dir / "plan.json").read_text())

    fig, axes = plt.subplots(
        len(args.candidates),
        1,
        figsize=(11, 2.4 * len(args.candidates) + 0.8),
        sharex=True,
        sharey=True,
    )
    axes = [axes] if len(args.candidates) == 1 else list(axes)
    for axis, candidate in zip(axes, args.candidates):
        run, metrics = _median_seed_run(sweep_dir, status, candidate)
        frame = pd.read_csv(
            sweep_dir / run["run_dir"] / "predictions.csv", parse_dates=["date"]
        )
        shown = frame if args.split == "all" else frame[frame["split"] == "test"]
        shown = shown.sort_values("date")
        scored = shown["target_observed"].astype(bool)
        axis.plot(
            shown["date"],
            shown["aqi_true"].where(scored),
            color=OBSERVED,
            lw=1.0 if args.split == "all" else 1.5,
            label="observed AQI",
        )
        axis.plot(
            shown["date"],
            shown["aqi_pred"],
            color=PREDICTED,
            lw=1.0 if args.split == "all" else 2,
            label="prediction (next day)",
        )
        if args.split == "all":
            test_start = frame.loc[frame["split"] == "test", "date"].min()
            axis.axvline(test_start, color=MUTED, lw=1.5, ls="--")
            axis.text(
                test_start,
                axis.get_ylim()[1],
                " test",
                va="top",
                fontsize=8,
                color=MUTED,
            )
        m = metrics["splits"]["test"]["observed_targets"]
        axis.set_title(
            f"{candidate} - seed {run['seed']} (median-RMSE seed): RMSE {m['rmse']:.2f}, MAE {m['mae']:.2f}, R² {m['r2']:.3f}",
            fontsize=9,
            loc="left",
            color=OBSERVED,
        )
        axis.set_ylabel("AQI", color=MUTED, fontsize=9)
        axis.grid(axis="y", color=GRID, lw=1)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(
        loc="upper left" if args.split == "all" else "upper right",
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    fig.suptitle(
        f"{status['sweep_id']} - {'2015-2020 (train + test)' if args.split == 'all' else 'test window'}, {args.label}",
        fontsize=10,
        x=0.01,
        ha="left",
        color=OBSERVED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    results_dir = PROJECT_DIR / "results" / plan["sweep_name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    for directory in (sweep_dir, results_dir):
        name = (
            "full_period_predictions.png"
            if args.split == "all"
            else "test_predictions.png"
        )
        fig.savefig(directory / name, dpi=150)
    print(results_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
