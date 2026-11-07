"""Aggregate and plot the photonic-reservoir extension.

Reads ``results/reservoir_summary.csv`` (written by ``utils/run_reservoirs.py``)
and writes:

* ``results/reservoir_table.md``          -- median test MSE per task, both budget arms
* ``results/reservoir_mse_by_task.png``   -- per-task comparison against the
  authors' grid-optimised reference numbers

The reference numbers come from the authors' released grid-search CSVs
(``original_results/``), restricted to the same learning problems.  They are the
*best* configuration the authors found per task over their full grid and 10
seeds, so they are a demanding yardstick: our runs use one hyperparameter
setting and 3 seeds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC = PROJECT_DIR / "original_results"
RESULTS = PROJECT_DIR / "results"

# (tag -> (dataset, sequence length, prediction step, pretty label))
TASKS = {
    "henon_p1": ("henon_1000", 4, 1, "Hénon k=1"),
    "henon_p4": ("henon_1000", 4, 4, "Hénon k=4"),
    "mackey_p1": ("mackey_1000", 4, 1, "Mackey-Glass k=1"),
    "mackey_p140": ("mackey_1000", 4, 140, "Mackey-Glass k=140"),
    "lorenz_p1": ("lorenz_1000", 4, 1, "Lorenz k=1"),
    "lorenz_p25": ("lorenz_1000", 4, 25, "Lorenz k=25"),
}

ORDER = [
    "MLP",
    "RNN",
    "LSTM",
    "photonic-dQNN",
    "photonic-RC",
    "photonic-seqRC",
    "photonic-memRC",
]
COLORS = {
    "MLP": "#E377C2",
    "RNN": "#55A868",
    "LSTM": "#8C564B",
    "photonic-dQNN": "#000000",
    "photonic-RC": "#DD8452",
    "photonic-seqRC": "#937860",
    "photonic-memRC": "#4C72B0",
}
QUANTUM_FILES = {
    "d-QNN": ("vqc_averaged_ids.csv", "paper_rivera-ruiz_with_inputlayer_"),
    "ru-QNN": ("vqc_averaged_ids.csv", "ruexp_"),
    "QRNN": ("qrnn_paper_averaged_ids.csv", None),
    "QLSTM": ("qlstm_paper_averaged_ids.csv", None),
    "le-QLSTM": ("qlstm_linear_enhanced_paper_averaged_ids.csv", None),
}
CLASSICAL_FILES = {
    "MLP": "mlp_averaged_ids.csv",
    "RNN": "rnn_averaged_ids.csv",
    "LSTM": "lstm_averaged_ids.csv",
}


def _grid_best(fname: str, prefix: str | None, ds: str, seq: int, pred: int):
    """Authors' best-validation config for one learning problem."""
    df = pd.read_csv(SRC / fname)
    if prefix is not None:
        df = df[df["Ansatz"].str.startswith(prefix)]
    sub = df[
        (df["Data"] == ds)
        & (df["Sequence Length"] == seq)
        & (df["Prediction Step"] == pred)
    ]
    if sub.empty:
        return np.nan, np.nan
    row = sub.loc[sub["MSE Validation Median"].idxmin()]
    return float(row["MSE Testing Median"]), float(row["Num Parameters"])


def authors_reference() -> pd.DataFrame:
    rows = []
    for tag, (ds, seq, pred, _) in TASKS.items():
        for name, fname in CLASSICAL_FILES.items():
            mse, npar = _grid_best(fname, None, ds, seq, pred)
            rows.append(
                {
                    "tag": tag,
                    "model": name,
                    "kind": "classical",
                    "mse": mse,
                    "params": npar,
                }
            )
        for name, (fname, prefix) in QUANTUM_FILES.items():
            mse, npar = _grid_best(fname, prefix, ds, seq, pred)
            rows.append(
                {
                    "tag": tag,
                    "model": name,
                    "kind": "quantum",
                    "mse": mse,
                    "params": npar,
                }
            )
    return pd.DataFrame(rows)


def load_runs() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "reservoir_summary.csv")
    g = (
        df.groupby(["arm", "tag", "display"])
        .agg(
            mse_median=("mse_test", "median"),
            mse_mad=("mse_test", lambda s: float(np.median(np.abs(s - s.median())))),
            params=("num_parameters", "max"),
            epochs=("epochs", "median"),
            time_s=("total_time_s", "mean"),
            n=("mse_test", "size"),
        )
        .reset_index()
    )
    return g


def write_table(runs: pd.DataFrame, ref: pd.DataFrame) -> None:
    lines = [
        "# Photonic reservoir extension — results",
        "",
        "**Not part of the paper reproduction.** These models are non-variational",
        "(frozen photonic circuit, trainable linear readout only), so they fall outside",
        "the paper's stated scope. They are evaluated inside the paper's protocol so the",
        "numbers are directly comparable.",
        "",
        "Median test MSE over 3 seeds of the best-validation model (the paper's metric).",
        "",
        "- **fixed** — 400 epochs for every model, comparable to `results/sweep_table.md`.",
        "- **conv** — the paper's validation-plateau convergence rule, capped at 3000 epochs.",
        "- **authors' grid-best** — best of the authors' full grid search over 10 seeds",
        "  for that task, from `original_results/` (a demanding reference: our runs use one",
        "  hyperparameter setting and 3 seeds).",
        "",
    ]
    for tag, (_, _, _, label) in TASKS.items():
        lines += [f"## {label}", ""]
        r = ref[ref.tag == tag]
        best_cl = r[r.kind == "classical"].nsmallest(1, "mse")
        best_qu = r[r.kind == "quantum"].nsmallest(1, "mse")
        header = (
            "| model | params | fixed (400 ep) | conv (≤3000 ep) | epochs (conv) |\n"
            "|:---|---:|---:|---:|---:|"
        )
        lines.append(header)
        for m in ORDER:
            f = runs[(runs.arm == "fixed") & (runs.tag == tag) & (runs.display == m)]
            c = runs[(runs.arm == "conv") & (runs.tag == tag) & (runs.display == m)]
            if f.empty and c.empty:
                continue
            pf = int(f["params"].iloc[0]) if not f.empty else int(c["params"].iloc[0])
            fv = f"{f['mse_median'].iloc[0]:.4g}" if not f.empty else "—"
            cv = f"{c['mse_median'].iloc[0]:.4g}" if not c.empty else "—"
            ce = f"{int(c['epochs'].iloc[0])}" if not c.empty else "—"
            lines.append(f"| {m} | {pf} | {fv} | {cv} | {ce} |")
        lines.append("")
        if not best_cl.empty:
            lines.append(
                f"Authors' grid-best **classical**: {best_cl['model'].iloc[0]} "
                f"= {best_cl['mse'].iloc[0]:.4g} ({int(best_cl['params'].iloc[0])} params)"
            )
        if not best_qu.empty:
            lines.append(
                f"Authors' grid-best **quantum**: {best_qu['model'].iloc[0]} "
                f"= {best_qu['mse'].iloc[0]:.4g} ({int(best_qu['params'].iloc[0])} params)"
            )
        lines.append("")
    (RESULTS / "reservoir_table.md").write_text("\n".join(lines))


def plot(runs: pd.DataFrame, ref: pd.DataFrame, arm: str = "conv") -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (tag, (_, _, _, label)) in zip(axes.ravel(), TASKS.items()):
        sub = runs[(runs.arm == arm) & (runs.tag == tag)]
        models = [m for m in ORDER if m in set(sub.display)]
        vals = [sub[sub.display == m]["mse_median"].iloc[0] for m in models]
        ax.bar(
            range(len(models)),
            vals,
            color=[COLORS[m] for m in models],
            edgecolor="black",
            linewidth=0.6,
        )
        r = ref[ref.tag == tag]
        cl = r[r.kind == "classical"]["mse"].min()
        qu = r[r.kind == "quantum"]["mse"].min()
        if np.isfinite(cl):
            ax.axhline(
                cl, ls="--", color="grey", lw=1.4, label="authors' grid-best classical"
            )
        if np.isfinite(qu):
            ax.axhline(
                qu, ls=":", color="purple", lw=1.4, label="authors' grid-best quantum"
            )
        ax.set_yscale("log")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("test MSE (log)")
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    budget = "convergence budget" if arm == "conv" else "fixed 400-epoch budget"
    fig.suptitle(
        f"Photonic reservoirs vs classical baselines, {budget} "
        "(extension — non-variational, outside the paper's scope)",
        y=0.02,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(RESULTS / "reservoir_mse_by_task.png", dpi=150)
    print("wrote results/reservoir_mse_by_task.png")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        default="auto",
        help="Budget arm to plot: fixed | conv | auto (prefer conv when complete).",
    )
    args = ap.parse_args()

    runs = load_runs()
    ref = authors_reference()
    ref.to_csv(RESULTS / "reservoir_reference.csv", index=False)
    write_table(runs, ref)
    print("wrote results/reservoir_table.md")

    arm = args.arm
    if arm == "auto":
        n_conv = len(runs[runs.arm == "conv"])
        n_fixed = len(runs[runs.arm == "fixed"])
        arm = "conv" if n_conv >= n_fixed > 0 else "fixed"
        print(f"plotting arm={arm} (conv cells={n_conv}, fixed cells={n_fixed})")
    plot(runs, ref, arm=arm)


if __name__ == "__main__":
    main()
