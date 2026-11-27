"""Summarize the C5 shared learning-rate sweep (MSE vs SWAP-test fidelity loss).

Reads only completed runs' ``metrics.json`` from a ``phase5_c5_lr_grid`` sweep
directory and applies the decision rule frozen in ``LOG.md`` (Amendment
2026-11-26T04:20:00Z): per loss and asset, pick the learning rate with the
lowest seed-mean train-target ``kl_joint``; compare the two picks paired by
seed with a 95 % t interval.

Usage (repository root)::

    python papers/QNN_STOCK/utils/compare_c5_grid.py \
        papers/QNN_STOCK/outdir/sweeps/<phase5_c5_lr_grid_dir> \
        --out papers/QNN_STOCK/results/phase5_c5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LOSSES = {"e6_qstl_mse_original": "MSE", "e3_qstl_original": "fidelity"}
PRIMARY = "output/train/kl_joint"
METRICS = {
    "output/train/kl_joint": ("output", "train", "kl_joint"),
    "output/train/kl_conditional_sum": ("output", "train", "kl_conditional_sum"),
    "inference/test/log_loss": ("inference", "test", "log_loss"),
}
EXPECTED_RUNS = 50
RUN_PATTERN = re.compile(r"^(?P<config>.+)__lr(?P<lr>[^_]+)_s(?P<seed>\d+)$")


def load_runs(sweep_dir: Path) -> pd.DataFrame:
    """One row per (run, asset, metric) from completed runs only."""

    rows = []
    for run_root in sorted((sweep_dir / "runs").iterdir()):
        match = RUN_PATTERN.match(run_root.name)
        if match is None:
            raise ValueError(f"Unexpected run directory name: {run_root.name}")
        (run_dir,) = list(run_root.glob("run_*"))
        status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
        if status["status"] != "COMPLETED":
            raise RuntimeError(f"{run_root.name} is {status['status']}, not COMPLETED")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        for asset, entry in metrics["assets"].items():
            model = entry["models"]["QSTL"]
            values = {"final_fidelity": model["final_fidelity"]}
            for name, (readout, split, key) in METRICS.items():
                values[name] = model[readout][split][key]
            for name, value in values.items():
                if not np.isfinite(value):
                    raise ValueError(f"{run_root.name} {asset} {name} is not finite")
                rows.append(
                    {
                        "loss": LOSSES[match["config"]],
                        "lr": float(match["lr"]),
                        "seed": int(match["seed"]),
                        "asset": asset,
                        "metric": name,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def paired(a: pd.Series, b: pd.Series) -> dict[str, float]:
    differences = (a - b).dropna()
    n = len(differences)
    mean = differences.mean()
    half = stats.t.ppf(0.975, n - 1) * differences.std(ddof=1) / np.sqrt(n)
    low, high = mean - half, mean + half
    reading = (
        "not distinguishable"
        if low <= 0 <= high
        else ("lower" if high < 0 else "higher")
    )
    return {
        "n_pairs": n,
        "mean_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "fidelity_vs_mse": reading,
    }


def plot_grid(long: pd.DataFrame, path: Path) -> None:
    """Train-target KL and state fidelity against the learning rate, per loss."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"MSE": "#d62728", "fidelity": "#1f77b4"}
    markers = {"AAPL": "o", "GOOGL": "s"}
    panels = [
        (PRIMARY, "train-target KL of the joint readout (log scale)", True),
        ("final_fidelity", "state fidelity F to the target", False),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (metric, label, log_y) in zip(axes, panels):
        rows = long[long.metric == metric]
        stats_by_lr = (
            rows.groupby(["loss", "asset", "lr"])["value"]
            .agg(["mean", "min", "max"])
            .reset_index()
        )
        for (loss, asset), group in stats_by_lr.groupby(["loss", "asset"]):
            ax.errorbar(
                group.lr,
                group["mean"],
                yerr=[group["mean"] - group["min"], group["max"] - group["mean"]],
                color=colors[loss],
                marker=markers[asset],
                capsize=2,
                label=f"{loss} loss, {asset}",
            )
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("SPSA learning rate (shared grid; paper: MSE 1e-5, fidelity 0.1)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "C5: MSE vs SWAP-test fidelity loss on a shared learning-rate grid "
        "(raw data, 10,000 epochs, mean and min-max over 5 seeds)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    long = load_runs(args.sweep_dir)
    n_runs = long.groupby(["loss", "lr", "seed"]).ngroups
    if n_runs != EXPECTED_RUNS:
        raise RuntimeError(f"{n_runs} completed runs, expected {EXPECTED_RUNS}")
    args.out.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out / "metrics_long.csv", index=False)

    aggregate = (
        long.groupby(["metric", "asset", "loss", "lr"])["value"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    aggregate.to_csv(args.out / "aggregate.csv", index=False)

    primary = long[long.metric == PRIMARY]
    means = primary.groupby(["asset", "loss", "lr"])["value"].mean()
    selected = (
        means.groupby(["asset", "loss"])
        .idxmin()
        .map(lambda key: key[2])
        .rename("selected_lr")
        .reset_index()
    )
    selected.to_csv(args.out / "selected_lr.csv", index=False)

    def series(asset: str, loss: str, lr: float, metric: str = PRIMARY) -> pd.Series:
        rows = long[
            (long.asset == asset)
            & (long.loss == loss)
            & (long.lr == lr)
            & (long.metric == metric)
        ]
        return rows.set_index("seed")["value"]

    comparisons = []
    for asset in sorted(long.asset.unique()):
        picks = selected[selected.asset == asset].set_index("loss")["selected_lr"]
        for label, mse_lr, fid_lr in (
            ("selected lr per loss", picks["MSE"], picks["fidelity"]),
            ("paper lrs (MSE 1e-5, fidelity 0.1)", 1e-5, 0.1),
        ):
            for metric in (PRIMARY, "inference/test/log_loss"):
                comparisons.append(
                    {
                        "asset": asset,
                        "comparison": label,
                        "metric": metric,
                        "mse_lr": mse_lr,
                        "fidelity_lr": fid_lr,
                        "mse_mean": series(asset, "MSE", mse_lr, metric).mean(),
                        "fidelity_mean": series(
                            asset, "fidelity", fid_lr, metric
                        ).mean(),
                        **paired(
                            series(asset, "fidelity", fid_lr, metric),
                            series(asset, "MSE", mse_lr, metric),
                        ),
                    }
                )
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(args.out / "pairs.csv", index=False)

    decisive = comparisons[
        (comparisons.comparison == "selected lr per loss")
        & (comparisons.metric == PRIMARY)
    ]
    lower = int((decisive.fidelity_vs_mse == "lower").sum())
    if lower == len(decisive):
        outcome = "SUPPORTED"
    elif lower == 0:
        outcome = "UNSUPPORTED"
    else:
        outcome = "UNRESOLVED"
    verdict = {
        "claim": "C5",
        "outcome": outcome,
        "rule": "fidelity lower than MSE (95 % interval excludes 0) on train kl_joint at each loss's selected lr, on all assets",
        "assets_fidelity_lower": lower,
        "assets": len(decisive),
        "runs": n_runs,
    }
    (args.out / "verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    plot_grid(long, args.out / "fig_c5_lr_grid.png")

    pd.set_option("display.width", 200)
    print(means.unstack("lr").round(4).to_string())
    print(selected.to_string(index=False))
    print(comparisons.round(4).to_string(index=False))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
