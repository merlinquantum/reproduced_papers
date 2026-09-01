"""Plot the reduced-compute live sweep results (independent reimplementation).

Reads ``results/sweep_summary.csv`` (produced by utils/sweep.py) and writes:
  * sweep_mse_by_model.png   grouped bar of median test MSE per model/problem
  * sweep_training_curves.png overlaid validation-MSE curves (seed 0)
  * sweep_table.md           median +/- MAD test MSE table

These are a REDUCED (V2) independent check: single representative
hyperparameter per model, fixed 400-epoch budget, 3 seeds.  They are not a
substitute for the grid-optimised authors' results (see plot_paper_figures.py).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT = PROJECT_DIR / "results"
SWEEP = PROJECT_DIR / "results" / "sweep"

QUANTUM = ["d-QNN", "ru-QNN", "QRNN", "QLSTM", "le-QLSTM"]
PHOTONIC = ["photonic"]
CLASSICAL = ["MLP", "RNN", "LSTM"]
ORDER = QUANTUM + PHOTONIC + CLASSICAL
COLORS = {
    "d-QNN": "#CCB974",
    "ru-QNN": "#8172B2",
    "QRNN": "#C44E52",
    "QLSTM": "#64B5CD",
    "le-QLSTM": "#4C72B0",
    "photonic": "#000000",
    "MLP": "#E377C2",
    "RNN": "#55A868",
    "LSTM": "#8C564B",
}
PROBLEM_TAGS = {
    (("henon_1000"), 1): "Henon k=1 (easy)",
    (("henon_1000"), 4): "Henon k=4 (hard)",
}


def _mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))


def load() -> pd.DataFrame:
    df = pd.read_csv(OUT / "sweep_summary.csv")
    return df


def plot_bars(df: pd.DataFrame) -> pd.DataFrame:
    probs = sorted(
        df.groupby(["data_label", "prediction_step"]).groups.keys(), key=lambda t: t[1]
    )
    fig, axes = plt.subplots(
        1, len(probs), figsize=(6 * len(probs), 4.5), squeeze=False
    )
    rows = []
    for ax, (data, pred) in zip(axes[0], probs):
        sub = df[(df.data_label == data) & (df.prediction_step == pred)]
        med = sub.groupby("display")["mse_test"].median()
        mad = sub.groupby("display")["mse_test"].apply(_mad)
        models = [m for m in ORDER if m in med.index]
        vals = [med[m] for m in models]
        errs = [mad[m] for m in models]
        colors = [COLORS[m] for m in models]
        ax.bar(range(len(models)), vals, yerr=errs, color=colors, capsize=3)
        ax.set_yscale("log")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_title(PROBLEM_TAGS.get((data, pred), f"{data} k={pred}"))
        ax.set_ylabel("Median test MSE (3 seeds)")
        for m in models:
            rows.append(
                {
                    "problem": f"{data} k={pred}",
                    "model": m,
                    "kind": (
                        "photonic"
                        if m in PHOTONIC
                        else "quantum"
                        if m in QUANTUM
                        else "classical"
                    ),
                    "mse_median": med[m],
                    "mse_mad": mad[m],
                }
            )
    plt.suptitle(
        "Reduced live reproduction: test MSE per model "
        "(400-epoch budget, 3 seeds, fixed version)"
    )
    plt.tight_layout()
    plt.savefig(OUT / "sweep_mse_by_model.png", dpi=130)
    plt.close()
    return pd.DataFrame(rows)


def plot_curves(df: pd.DataFrame) -> None:
    probs = sorted(
        df.groupby(["data_label", "prediction_step"]).groups.keys(), key=lambda t: t[1]
    )
    fig, axes = plt.subplots(
        1, len(probs), figsize=(6 * len(probs), 4.5), squeeze=False
    )
    for ax, (data, pred) in zip(axes[0], probs):
        for m in ORDER:
            # find a seed-0 run directory for this model/problem
            hits = list(SWEEP.glob(f"{m}_*_s0_*/losses.csv"))
            hits = [h for h in hits if _matches(h, data, pred)]
            if not hits:
                continue
            lc = pd.read_csv(hits[0])
            ax.plot(lc["epoch"], lc["val_mse"], label=m, color=COLORS[m], lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE")
        ax.set_title(PROBLEM_TAGS.get((data, pred), f"{data} k={pred}"))
        ax.legend(fontsize=7, ncol=2)
    plt.suptitle("Reduced live reproduction: validation-MSE training curves (seed 0)")
    plt.tight_layout()
    plt.savefig(OUT / "sweep_training_curves.png", dpi=130)
    plt.close()


def _matches(losses_path: Path, data: str, pred: int) -> bool:
    mj = losses_path.parent / "metrics.json"
    if not mj.exists():
        return False
    import json

    d = json.loads(mj.read_text())
    return d["data_label"] == data and d["prediction_step"] == pred


def write_table(table: pd.DataFrame) -> None:
    lines = [
        "# Reduced live reproduction — median test MSE (3 seeds, 400 epochs)\n",
        "Single representative hyperparameter per model; fixed epoch budget "
        "(NOT the paper's grid search). Fixed (bug-corrected) model versions.\n",
    ]
    for prob, g in table.groupby("problem"):
        lines.append(f"\n## {prob}\n")
        g = g.sort_values("mse_median")
        lines.append(
            g[["model", "kind", "mse_median", "mse_mad"]].to_markdown(
                index=False, floatfmt=".5g"
            )
        )
    (OUT / "sweep_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    df = load()
    table = plot_bars(df)
    plot_curves(df)
    write_table(table)
    print(f"\nSweep figures -> {OUT}")


if __name__ == "__main__":
    main()
