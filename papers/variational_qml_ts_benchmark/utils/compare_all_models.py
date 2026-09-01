"""Single side-by-side comparison of every model in this reproduction.

The live results live in two files because they were produced by two different
sweeps with different scopes:

* ``results/sweep_summary.csv``     -- the 5 gate-based quantum models + 3 classical
  baselines + the photonic d-QNN, on the 2 Henon problems, fixed 400 epochs.
* ``results/reservoir_summary.csv`` -- classical + photonic d-QNN + the 3 photonic
  reservoirs, on 6 problems, at both the fixed and the convergence budget.

They overlap on MLP/RNN/LSTM/photonic-dQNN at Henon k=1/k=4, fixed budget, and
agree there to the last digit (runs are deterministic given the seed), so merging
them is safe.  This script does the merge and writes:

* ``results/all_models_comparison.md``  -- ranked tables + an explicit coverage matrix
* ``results/all_models_comparison.csv`` -- the merged long-form table

Coverage is deliberately *not* filled in by guessing: cells that were never run
are shown as "not run", with the reason.  The gate-based quantum models were not
run live beyond Henon because a single 400-epoch run costs 2-29 CPU-minutes each
(le-QLSTM ~29 min, QLSTM ~20 min, d-QNN ~16 min), i.e. ~13 CPU-hours to extend
them to the other four problems; the authors' grid-search CSVs cover all 27 tasks
and are quoted instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

KIND_COLOR = {
    "classical": "#55A868",
    "quantum (gate, variational)": "#8172B2",
    "photonic (variational)": "#000000",
    "photonic (reservoir, non-variational)": "#DD8452",
}

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_DIR / "results"
SRC = PROJECT_DIR / "original_results"

TASKS = {
    "henon_1000_p1": "Hénon k=1",
    "henon_1000_p4": "Hénon k=4",
    "mackey_1000_p1": "Mackey-Glass k=1",
    "mackey_1000_p140": "Mackey-Glass k=140",
    "lorenz_1000_p1": "Lorenz k=1",
    "lorenz_1000_p25": "Lorenz k=25",
}

KIND = {
    "MLP": "classical",
    "RNN": "classical",
    "LSTM": "classical",
    "d-QNN": "quantum (gate, variational)",
    "ru-QNN": "quantum (gate, variational)",
    "QRNN": "quantum (gate, variational)",
    "QLSTM": "quantum (gate, variational)",
    "le-QLSTM": "quantum (gate, variational)",
    "photonic-dQNN": "photonic (variational)",
    "photonic-RC": "photonic (reservoir, non-variational)",
    "photonic-seqRC": "photonic (reservoir, non-variational)",
    "photonic-memRC": "photonic (reservoir, non-variational)",
}
ORDER = list(KIND)

AUTHOR_FILES = {
    "d-QNN": ("vqc_averaged_ids.csv", "paper_rivera-ruiz_with_inputlayer_"),
    "ru-QNN": ("vqc_averaged_ids.csv", "ruexp_"),
    "QRNN": ("qrnn_paper_averaged_ids.csv", None),
    "QLSTM": ("qlstm_paper_averaged_ids.csv", None),
    "le-QLSTM": ("qlstm_linear_enhanced_paper_averaged_ids.csv", None),
    "MLP": ("mlp_averaged_ids.csv", None),
    "RNN": ("rnn_averaged_ids.csv", None),
    "LSTM": ("lstm_averaged_ids.csv", None),
}


def load_live() -> pd.DataFrame:
    """Merge the two live sweeps into one long-form table."""
    s = pd.read_csv(RESULTS / "sweep_summary.csv")
    s["task"] = s.data_label + "_p" + s.prediction_step.astype(str)
    s["display"] = s.display.replace({"photonic": "photonic-dQNN"})
    s["arm"] = "fixed"

    r = pd.read_csv(RESULTS / "reservoir_summary.csv")
    r["task"] = r.data_label + "_p" + r.prediction_step.astype(str)

    keep = [
        "display",
        "task",
        "arm",
        "mse_test",
        "num_parameters",
        "epochs",
        "total_time_s",
    ]
    both = pd.concat([s[keep], r[keep]], ignore_index=True)
    # Overlapping cells are identical; drop the duplicates from the older sweep.
    return both.drop_duplicates(subset=["display", "task", "arm", "mse_test"])


def summarise(live: pd.DataFrame) -> pd.DataFrame:
    g = (
        live.groupby(["arm", "task", "display"])
        .agg(
            mse=("mse_test", "median"),
            params=("num_parameters", "max"),
            epochs=("epochs", "median"),
            n_seeds=("mse_test", "size"),
        )
        .reset_index()
    )
    return g


def authors_grid_best() -> pd.DataFrame:
    rows = []
    for name, (fname, prefix) in AUTHOR_FILES.items():
        df = pd.read_csv(SRC / fname)
        if prefix is not None:
            df = df[df["Ansatz"].str.startswith(prefix)]
        for task, _ in TASKS.items():
            ds, pred = task.rsplit("_p", 1)
            sub = df[
                (df["Data"] == ds)
                & (df["Sequence Length"] == 4)
                & (df["Prediction Step"] == int(pred))
            ]
            if sub.empty:
                continue
            row = sub.loc[sub["MSE Validation Median"].idxmin()]
            rows.append(
                {
                    "display": name,
                    "task": task,
                    "mse": float(row["MSE Testing Median"]),
                    "params": int(row["Num Parameters"]),
                }
            )
    return pd.DataFrame(rows)


def _fmt(v) -> str:
    return (
        "—"
        if v is None or (isinstance(v, float) and not np.isfinite(v))
        else f"{v:.4g}"
    )


def write_report(g: pd.DataFrame, auth: pd.DataFrame) -> None:
    L = [
        "# All-model comparison",
        "",
        "Every model in this reproduction, side by side. Median test MSE over 3 seeds",
        "of the best-validation model (the paper's own metric). Merged from",
        "`sweep_summary.csv` and `reservoir_summary.csv`; the models common to both",
        "agree to the last digit, so the merge introduces no inconsistency.",
        "",
        "Model kinds:",
        "",
        "| model | kind |",
        "|:---|:---|",
    ]
    L += [f"| {m} | {KIND[m]} |" for m in ORDER]
    L += [
        "",
        "The photonic reservoirs are an **extension**, not part of the reproduction:",
        "the paper benchmarks *variational* QML and a reservoir trains no circuit",
        "parameters. They are included here because they are evaluated in the same",
        "protocol.",
        "",
        "---",
        "",
        "## 1. Full live comparison — Hénon, fixed 400-epoch budget",
        "",
        "The only configuration in which *every* model was run live.",
        "",
        "![All 12 models](all_models_comparison.png)",
        "",
    ]
    for task in ["henon_1000_p1", "henon_1000_p4"]:
        sub = g[(g.arm == "fixed") & (g.task == task)].set_index("display")
        sub = sub.reindex([m for m in ORDER if m in sub.index]).sort_values("mse")
        L += [
            f"### {TASKS[task]}",
            "",
            "| rank | model | kind | params | test MSE |",
            "|---:|:---|:---|---:|---:|",
        ]
        for i, (m, row) in enumerate(sub.iterrows(), 1):
            L.append(f"| {i} | {m} | {KIND[m]} | {int(row.params)} | {_fmt(row.mse)} |")
        L.append("")

    L += [
        "---",
        "",
        "## 2. Convergence budget — 6 tasks (photonic + classical only)",
        "",
        "The paper's validation-plateau rule, capped at 3000 epochs. The gate-based",
        "quantum models were not re-run at this budget (see coverage below).",
        "",
        "The ordering is not stable across budgets — at 400 epochs the classical",
        "baselines are under-trained and the photonic d-QNN leads; under the plateau",
        "rule the LSTM overtakes it. Any modality claim read off the fixed-budget",
        "numbers alone is a training-budget artifact:",
        "",
        "![Budget effect](budget_effect.png)",
        "",
        "| model | " + " | ".join(TASKS[t] for t in TASKS) + " | mean rank |",
        "|:---|" + "---:|" * (len(TASKS) + 1),
    ]
    conv = g[g.arm == "conv"].pivot_table(index="display", columns="task", values="mse")
    conv = conv.reindex(columns=list(TASKS))
    ranks = conv.rank()
    for m in ORDER:
        if m not in conv.index:
            continue
        cells = " | ".join(_fmt(conv.loc[m, t]) for t in TASKS)
        L.append(f"| {m} | {cells} | {ranks.loc[m].mean():.2f} |")
    L.append("")

    L += [
        "---",
        "",
        "## 3. Authors' grid-best reference (all 8 paper models, same 6 tasks)",
        "",
        "Best configuration from the authors' full grid search over 10 seeds, from",
        "`original_results/`. This is the authoritative comparison for the paper's",
        "claim and a demanding yardstick — our live runs use one hyperparameter",
        "setting and 3 seeds, so they are expected to sit above these numbers.",
        "",
        "| model | " + " | ".join(TASKS[t] for t in TASKS) + " | mean rank |",
        "|:---|" + "---:|" * (len(TASKS) + 1),
    ]
    ap = auth.pivot_table(index="display", columns="task", values="mse").reindex(
        columns=list(TASKS)
    )
    aranks = ap.rank()
    for m in ORDER:
        if m not in ap.index:
            continue
        cells = " | ".join(_fmt(ap.loc[m, t]) for t in TASKS)
        L.append(f"| {m} | {cells} | {aranks.loc[m].mean():.2f} |")
    L.append("")

    L += [
        "---",
        "",
        "## 4. Coverage — what was actually run",
        "",
        "`F` = fixed 400-epoch arm, `C` = convergence arm, `·` = not run.",
        "",
        "| model | " + " | ".join(TASKS[t] for t in TASKS) + " |",
        "|:---|" + "---:|" * len(TASKS),
    ]
    for m in ORDER:
        cells = []
        for t in TASKS:
            has_f = not g[(g.arm == "fixed") & (g.task == t) & (g.display == m)].empty
            has_c = not g[(g.arm == "conv") & (g.task == t) & (g.display == m)].empty
            cells.append(("F" if has_f else "") + ("C" if has_c else "") or "·")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += [
        "",
        "**Why the gaps.** The 5 gate-based quantum models were run live only on the",
        "two Hénon problems at the fixed budget. Extending them to the other four",
        "problems costs roughly 13 CPU-hours (le-QLSTM ~29 min, QLSTM ~20 min, d-QNN",
        "~16 min per 400-epoch run, x 4 problems x 3 seeds), and re-running them at the",
        "convergence budget would cost several times that. The authors' grid CSVs",
        "(section 3) already cover all 27 tasks for those models, so the live runs add",
        "little beyond confirming our reimplementation reproduces the ordering — which",
        "`results/sweep_table.md` already establishes.",
        "",
        "**Consequence for reading section 2.** Because the gate-based quantum models",
        "are absent there, section 2 ranks the photonic models against *classical*",
        "baselines only. For a photonic-vs-gate-quantum comparison use section 1",
        "(same budget, all models) or section 3 (grid-optimised, all paper models).",
    ]
    (RESULTS / "all_models_comparison.md").write_text("\n".join(L))


def plot_all_models(g: pd.DataFrame) -> None:
    """Every model side by side on the two tasks where all 12 were run live."""
    tasks = ["henon_1000_p1", "henon_1000_p4"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, task in zip(axes, tasks):
        sub = g[(g.arm == "fixed") & (g.task == task)].set_index("display")
        sub = sub.reindex([m for m in ORDER if m in sub.index]).sort_values(
            "mse", ascending=False
        )
        colors = [KIND_COLOR[KIND[m]] for m in sub.index]
        ax.barh(
            range(len(sub)), sub.mse, color=colors, edgecolor="black", linewidth=0.6
        )
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(
            [f"{m}  ({int(p)}p)" for m, p in zip(sub.index, sub.params)], fontsize=9
        )
        ax.set_xscale("log")
        ax.set_xlabel("median test MSE (log)")
        ax.set_title(TASKS[task], fontsize=11)
        ax.grid(axis="x", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.6)
        for c in KIND_COLOR.values()
    ]
    fig.legend(
        handles, list(KIND_COLOR), loc="upper center", ncol=2, frameon=False, fontsize=9
    )
    fig.suptitle(
        "All 12 models, fixed 400-epoch budget, 3 seeds "
        "(the only configuration in which every model was run live)",
        y=0.02,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])
    fig.savefig(RESULTS / "all_models_comparison.png", dpi=150)
    print("wrote results/all_models_comparison.png")


def plot_budget_effect(g: pd.DataFrame) -> None:
    """Mean rank at the fixed vs the convergence budget -- the ordering reverses."""
    rows = []
    for arm in ["fixed", "conv"]:
        p = g[g.arm == arm].pivot_table(index="display", columns="task", values="mse")
        p = p.reindex(columns=list(TASKS)).dropna(axis=0, how="any")
        rows.append(p.rank().mean(axis=1).rename(arm))
    both = pd.concat(rows, axis=1).dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    for m, row in both.iterrows():
        c = KIND_COLOR[KIND[m]]
        ax.plot([0, 1], [row["fixed"], row["conv"]], "-o", color=c, lw=2, ms=7)
        ax.annotate(
            m,
            (0, row["fixed"]),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            color=c,
        )
        ax.annotate(
            f"{row['conv']:.2f}",
            (1, row["conv"]),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color=c,
        )
    ax.set_xlim(-0.55, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["fixed\n400 epochs", "convergence\n(plateau rule, cap 3000)"])
    ax.invert_yaxis()
    ax.set_ylabel("mean rank over 6 tasks (lower = better)")
    ax.set_title(
        "Training budget, not modality, drives the ordering\n"
        "the photonic d-QNN's lead at a fixed budget is an under-training artifact",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "budget_effect.png", dpi=150)
    print("wrote results/budget_effect.png")


def main() -> None:
    live = load_live()
    g = summarise(live)
    auth = authors_grid_best()
    plot_all_models(g)
    plot_budget_effect(g)
    g.assign(source="live").to_csv(RESULTS / "all_models_comparison.csv", index=False)
    write_report(g, auth)
    print("wrote results/all_models_comparison.md")
    print("wrote results/all_models_comparison.csv")


if __name__ == "__main__":
    main()
