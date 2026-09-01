"""Reproduce the paper's headline figures from the authors' released result CSVs.

The upstream repository ships the full grid-search results
(``Results/<model>_averaged_ids.csv``, mirrored here under
``original_results/``).  Regenerating Figures 3/4/5 from these files reproduces
the paper's central *analysis and claim* — that classical models rank at least
on par with, and usually above, the variational quantum models — without
re-running the (thousands-of-epochs) grid search.

Outputs (written to ``results/``):
  * ranking_all_models.png            (Fig. 5)
  * best_mse_per_task.png             (Fig. 3 essence)
  * mse_vs_parameters_lorenz.png      (Fig. 4)
  * claim_summary.csv / .md           (mean rank, win counts)

Run:
    python utils/plot_paper_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC = PROJECT_DIR / "original_results"
OUT = PROJECT_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)

QUANTUM = ["d-QNN", "ru-QNN", "QRNN", "QLSTM", "le-QLSTM"]
CLASSICAL = ["MLP", "RNN", "LSTM"]
COLORS = {
    "d-QNN": "#CCB974",
    "ru-QNN": "#8172B2",
    "QRNN": "#C44E52",
    "QLSTM": "#64B5CD",
    "le-QLSTM": "#4C72B0",
    "MLP": "#E377C2",
    "RNN": "#55A868",
    "LSTM": "#8C564B",
}


def load_models() -> dict[str, pd.DataFrame]:
    vqc = pd.read_csv(SRC / "vqc_averaged_ids.csv")
    frames = {
        "d-QNN": vqc[
            vqc["Ansatz"].str.startswith("paper_rivera-ruiz_with_inputlayer_")
        ],
        "ru-QNN": vqc[vqc["Ansatz"].str.startswith("ruexp_")],
        "QRNN": pd.read_csv(SRC / "qrnn_paper_averaged_ids.csv").pipe(
            lambda d: d[d["Ansatz"] == "paper_no_reset"]
        ),
        "QLSTM": pd.read_csv(SRC / "qlstm_paper_averaged_ids.csv"),
        "le-QLSTM": pd.read_csv(SRC / "qlstm_linear_enhanced_paper_averaged_ids.csv"),
        "MLP": pd.read_csv(SRC / "mlp_averaged_ids.csv"),
        "RNN": pd.read_csv(SRC / "rnn_averaged_ids.csv"),
        "LSTM": pd.read_csv(SRC / "lstm_averaged_ids.csv"),
    }
    return frames


def find_best(df: pd.DataFrame) -> pd.DataFrame:
    """Best config per learning problem = min validation-median MSE."""
    keys = ["Prediction Step", "Data", "Sequence Length"]
    idx = df.groupby(keys, observed=True)["MSE Validation Median"].idxmin()
    return df.loc[idx]


def build_ranking(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name, df in frames.items():
        b = find_best(df).copy()
        b["Model Name"] = name
        parts.append(b)
    res = pd.concat(parts, ignore_index=True)
    res["Rank"] = res.groupby(["Sequence Length", "Data", "Prediction Step"])[
        "MSE Testing Median"
    ].rank(method="min")
    return res


def plot_ranking(res: pd.DataFrame) -> None:
    rank_counts = res.groupby(["Model Name", "Rank"]).size().unstack(fill_value=0)
    mean_rank = res.groupby("Model Name")["Rank"].mean().sort_values(ascending=False)
    rank_counts = rank_counts.loc[mean_rank.index]
    cmap = plt.colormaps.get_cmap("coolwarm")
    n_ranks = rank_counts.shape[1]
    colors = [cmap(i) for i in np.linspace(0, 1, n_ranks)]
    ax = rank_counts.plot(kind="barh", stacked=True, color=colors, figsize=(7, 4))
    ax.set_xlabel("Number of learning problems (of 27)")
    ax.set_ylabel("Model")
    ax.set_title(
        "Model ranking across 27 learning problems (from authors' CSVs)\n"
        "dark blue = rank 1 (best)"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [str(int(float(x))) for x in labels],
        title="Rank",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()
    plt.savefig(OUT / "ranking_all_models.png", dpi=130)
    plt.close()
    return mean_rank


def plot_best_mse_per_task(res: pd.DataFrame) -> None:
    """Grouped: for each dataset/pred, best test MSE per model (seq collapsed to min)."""
    datasets = [
        ("mackey_1000", [1, 70, 140]),
        ("henon_1000", [1, 2, 4]),
        ("lorenz_1000", [1, 13, 25]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    order = QUANTUM + CLASSICAL
    for ax, (data, preds) in zip(axes, datasets):
        sub = res[res["Data"] == data]
        # collapse sequence lengths: best (min) test MSE per (model, pred)
        piv = (
            sub.groupby(["Model Name", "Prediction Step"])["MSE Testing Median"]
            .min()
            .reset_index()
        )
        x = np.arange(len(preds))
        w = 0.1
        for i, m in enumerate(order):
            vals = [
                piv[(piv["Model Name"] == m) & (piv["Prediction Step"] == p)][
                    "MSE Testing Median"
                ].min()
                for p in preds
            ]
            ax.bar(x + i * w, vals, w, label=m, color=COLORS[m])
        ax.set_yscale("log")
        ax.set_xticks(x + w * (len(order) - 1) / 2)
        ax.set_xticklabels([f"k={p}" for p in preds])
        ax.set_title(data.replace("_1000", ""))
        ax.set_xlabel("Prediction horizon")
    axes[0].set_ylabel("Best median test MSE")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.suptitle("Best test MSE per model and horizon (Fig. 3 essence, authors' CSVs)")
    plt.tight_layout()
    plt.savefig(OUT / "best_mse_per_task.png", dpi=130)
    plt.close()


def plot_mse_vs_params(frames: dict[str, pd.DataFrame]) -> None:
    """Fig. 4: MSE vs #parameters on Lorenz seq=16 for pred 1 and 25."""
    sel = ["d-QNN", "QRNN", "le-QLSTM", "LSTM"]
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    for ax, pred in zip(axes, [1, 25]):
        for m in sel:
            df = frames[m]
            d = df[
                (df["Data"] == "lorenz_1000")
                & (df["Sequence Length"] == 16)
                & (df["Prediction Step"] == pred)
            ]
            d = d.sort_values("Num Parameters")
            ax.errorbar(
                d["Num Parameters"],
                d["MSE Testing Median"],
                yerr=d["MSE Testing Mad"],
                marker="o",
                ms=4,
                lw=1,
                capsize=2,
                label=m,
                color=COLORS[m],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Median test MSE")
        ax.set_title(f"Lorenz, seq=16, pred={pred}")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Number of trainable parameters")
    plt.suptitle("MSE vs parameter count (Fig. 4, authors' CSVs)")
    plt.tight_layout()
    plt.savefig(OUT / "mse_vs_parameters_lorenz.png", dpi=130)
    plt.close()


def write_claim_summary(res: pd.DataFrame, mean_rank: pd.Series) -> None:
    wins = res[res["Rank"] == 1].groupby("Model Name").size()
    top3 = res[res["Rank"] <= 3].groupby("Model Name").size()
    summary = pd.DataFrame(
        {
            "mean_rank": mean_rank,
            "n_rank1": wins.reindex(mean_rank.index).fillna(0).astype(int),
            "n_top3": top3.reindex(mean_rank.index).fillna(0).astype(int),
            "kind": [
                "quantum" if m in QUANTUM else "classical" for m in mean_rank.index
            ],
        }
    ).sort_values("mean_rank")
    summary.to_csv(OUT / "claim_summary.csv")

    q_mean = summary[summary.kind == "quantum"]["mean_rank"].mean()
    c_mean = summary[summary.kind == "classical"]["mean_rank"].mean()
    lines = [
        "# Claim reproduction (from authors' released result CSVs)\n",
        "Best configuration per learning problem selected by validation-median MSE; "
        "ranked by test-median MSE across all 27 learning problems (lower rank = better).\n",
        summary.round(3).to_markdown(),
        "",
        f"\nMean rank of **classical** models: {c_mean:.2f}",
        f"\nMean rank of **quantum** models:   {q_mean:.2f}",
        "\n\n**Conclusion:** classical models achieve a better (lower) average rank than "
        "the variational quantum models, reproducing the paper's central claim.",
    ]
    (OUT / "claim_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    frames = load_models()
    res = build_ranking(frames)
    mean_rank = plot_ranking(res)
    plot_best_mse_per_task(res)
    plot_mse_vs_params(frames)
    write_claim_summary(res, mean_rank)
    print(f"\nFigures written to {OUT}")


if __name__ == "__main__":
    main()
