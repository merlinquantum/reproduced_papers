"""Write up the reservoir hyperparameter search.

Reads ``results/reservoir_search_summary.csv`` (the search) and
``results/reservoir_summary.csv`` (the a-priori baseline) and produces
``results/reservoir_search.md`` + ``results/reservoir_search.png``.

The comparison that matters is the **convergence arm** (stage 4, extended by
stage 5 where available). At a flat epoch budget the classical baselines are
under-trained, so a tuned reservoir can look better than it is -- the artifact
documented in README §4. Both arms are reported here precisely so that gap is
visible rather than hidden.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_DIR / "results"
TASKS = ["henon_p1", "henon_p4", "mackey_p1", "mackey_p140", "lorenz_p1", "lorenz_p25"]
PRETTY = {
    "henon_p1": "Hénon k=1",
    "henon_p4": "Hénon k=4",
    "mackey_p1": "MG k=1",
    "mackey_p140": "MG k=140",
    "lorenz_p1": "Lorenz k=1",
    "lorenz_p25": "Lorenz k=25",
}
TUNED_NAME = {
    "tunedRC": "photonic-RC",
    "tunedSeqRC": "photonic-seqRC",
    "tunedMemRC": "photonic-memRC",
}


def _piv(df, values="mse_test"):
    p = df.pivot_table(index="cfg", columns="tag", values=values, aggfunc="median")
    return p.reindex(columns=[t for t in TASKS if t in p.columns])


def _effective_convergence_runs(
    search_results: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Overlay cap-specific retries on the original stage-4 results.

    Parameters
    ----------
    search_results : pandas.DataFrame
        Collected staged reservoir-search results.

    Returns
    -------
    tuple[pandas.DataFrame, int]
        One convergence result per configuration, task, and seed, plus the
        largest cap represented by an extended retry.

    Raises
    ------
    ValueError
        If no original stage-4 results are available.
    """
    stage4_results = search_results[search_results.stage == "s4"].copy()
    if stage4_results.empty:
        raise ValueError("Cannot report convergence results without stage 4.")
    stage4_results["_cap"] = 3000

    extended_caps = search_results.stage.astype(str).str.extract(
        r"^s4_cap([0-9]+)$", expand=False
    )
    extended_results = search_results[extended_caps.notna()].copy()
    if extended_results.empty:
        return stage4_results.drop(columns="_cap"), 3000

    extended_results["_cap"] = extended_caps[extended_caps.notna()].astype(int)
    combined_results = pd.concat([stage4_results, extended_results], ignore_index=True)
    combined_results = combined_results.sort_values("_cap").drop_duplicates(
        subset=["cfg", "tag", "seed"], keep="last"
    )
    largest_cap = int(combined_results["_cap"].max())
    return combined_results.drop(columns="_cap"), largest_cap


def main() -> None:
    s = pd.read_csv(RESULTS / "reservoir_search_summary.csv")
    base = pd.read_csv(RESULTS / "reservoir_summary.csv")
    tuned_cfg = json.loads((RESULTS / "reservoir_search_best.json").read_text())
    convergence_runs, convergence_cap = _effective_convergence_runs(s)
    convergence_title = (
        "convergence budget (cap 3000)"
        if convergence_cap == 3000
        else f"convergence budget (extended cap {convergence_cap})"
    )

    L = [
        "# Photonic reservoir — hyperparameter search",
        "",
        "The reservoir numbers in `reservoir_table.md` used one a-priori configuration",
        "(6 modes, 3 photons, scale=pi, leak=0.5, 2 memristors), so their last-place",
        "ranking was only a *lower bound*. This is the search that tests how much of the",
        "gap was the topology and how much was an unlucky guess.",
        "",
        "**Selection is on validation MSE throughout**; test numbers are read out only",
        "afterwards. Tuning uses the two Hénon tasks; Mackey-Glass and Lorenz are held",
        "out, so stage 3/4 also test whether the tuning generalises.",
        "",
        f"**Tuned configuration:** {tuned_cfg['modes']} modes, {tuned_cfg['photons']} photons, "
        f"scale={tuned_cfg['scale']:.4g} ({tuned_cfg['scale'] / math.pi:.3g}·pi), "
        f"leak={tuned_cfg['leak']}, {tuned_cfg['mem']} memristors",
        "",
    ]

    # ---- Stage 1: geometry -------------------------------------------------
    s1 = _piv(s[s.stage == "s1"], "mse_val")
    if not s1.empty:
        s1 = s1.assign(mean_rank=s1.rank().mean(axis=1)).sort_values("mean_rank")
        L += [
            "## Stage 1 — optical geometry (modes x photons x encoding scale)",
            "",
            "Static reservoir, validation MSE, 32 configurations. Top 5 and the a-priori:",
            "",
            "| config | "
            + " | ".join(PRETTY[t] for t in s1.columns[:-1])
            + " | mean rank |",
            "|:---|" + "---:|" * len(s1.columns),
        ]
        show = list(s1.index[:5]) + [i for i in s1.index if i.startswith("m6p3s3.14")]
        for cfg in dict.fromkeys(show):
            r = s1.loc[cfg]
            cells = " | ".join(f"{r[c]:.4g}" for c in s1.columns[:-1])
            star = " *(a-priori)*" if cfg.startswith("m6p3s3.14") else ""
            L.append(f"| `{cfg}`{star} | {cells} | {r['mean_rank']:.1f} |")
        L += [
            "",
            "Every top configuration sits at the grid's largest scale — the coarse grid",
            "was **monotone in scale**, so its winner was an edge optimum, not an optimum.",
            "Stage 1b refines that axis.",
            "",
        ]

    # ---- Stage 1b: scale refinement ---------------------------------------
    s1b = s[s.stage == "s1b"]
    if not s1b.empty:
        sc = s1b.copy()
        sc["scale"] = sc.cfg.str.extract(r"s([0-9.]+)$").astype(float)
        g = sc.groupby("scale").agg(
            val=("mse_val", "median"), test=("mse_test", "median")
        )
        g["x_pi"] = (g.index / math.pi).round(1)
        L += [
            "## Stage 1b — refining the encoding scale",
            "",
            "1-D sweep at the winning geometry. The curve *does* turn over, so the",
            "refined optimum is interior:",
            "",
            "| scale | x pi | validation MSE | test MSE |",
            "|---:|---:|---:|---:|",
        ]
        best = g.val.idxmin()
        for sv, r in g.iterrows():
            mark = " **<- best**" if sv == best else ""
            L.append(
                f"| {sv:.4g} | {r['x_pi']:g} | {r['val']:.4g}{mark} | {r['test']:.4g} |"
            )
        L.append("")

    # ---- Stage 2: memristor ------------------------------------------------
    s2 = _piv(s[s.stage == "s2"], "mse_val")
    if not s2.empty:
        s2 = s2.assign(mean_rank=s2.rank().mean(axis=1)).sort_values("mean_rank")
        spread = s2.iloc[:, 0].max() / s2.iloc[:, 0].min()
        L += [
            "## Stage 2 — memristor dynamics (leak x number of memristors)",
            "",
            f"15 configurations. The spread across the *entire* grid is only "
            f"**{spread:.2f}x** on {PRETTY[s2.columns[0]]} — the memristor parameters",
            "barely matter. Note that `leak0.00` (zero memory retention) ranks near the",
            "top: switching the memory off costs nothing, an independent echo of the",
            "capacity-control result.",
            "",
            "| config | "
            + " | ".join(PRETTY[t] for t in s2.columns[:-1])
            + " | mean rank |",
            "|:---|" + "---:|" * len(s2.columns),
        ]
        for cfg in s2.index:
            r = s2.loc[cfg]
            cells = " | ".join(f"{r[c]:.4g}" for c in s2.columns[:-1])
            L.append(f"| `{cfg}` | {cells} | {r['mean_rank']:.1f} |")
        L.append("")

    # ---- Stage 3/4: tuned vs a-priori vs classical -------------------------
    comparisons = (
        (s[s.stage == "s3"], "fixed", "fixed 400-epoch budget"),
        (
            convergence_runs,
            "conv",
            convergence_title,
        ),
    )
    for comparison_runs, arm, title in comparisons:
        sub = _piv(comparison_runs)
        if sub.empty:
            continue
        bl = base[base.arm == arm].pivot_table(
            index="display", columns="tag", values="mse_test", aggfunc="median"
        )
        bl = bl.reindex(columns=[t for t in TASKS if t in bl.columns])
        L += [
            f"## Tuned vs a-priori — {title}",
            "",
            "| model | "
            + " | ".join(PRETTY[t] for t in sub.columns)
            + " | geo-mean gain |",
            "|:---|" + "---:|" * (len(sub.columns) + 1),
        ]
        for cfg, disp in TUNED_NAME.items():
            if cfg not in sub.index or disp not in bl.index:
                continue
            gain = bl.loc[disp] / sub.loc[cfg]
            gm = float(np.exp(np.log(gain.astype(float)).mean()))
            L.append(
                f"| {disp} **tuned** | "
                + " | ".join(f"{v:.4g}" for v in sub.loc[cfg])
                + f" | **{gm:.2f}x** |"
            )
            L.append(
                f"| {disp} a-priori | "
                + " | ".join(f"{v:.4g}" for v in bl.loc[disp])
                + " | — |"
            )
        L.append("")
        # ranking including the classical baselines
        keep = [m for m in ["MLP", "RNN", "LSTM", "photonic-dQNN"] if m in bl.index]
        allm = pd.concat(
            [
                bl.loc[keep],
                sub.rename(index=lambda c: TUNED_NAME.get(c, c) + " (tuned)"),
            ]
        )
        allm = allm.dropna(axis=0, how="any")
        rk = allm.rank().mean(axis=1).sort_values()
        L += [
            f"Mean rank over the six tasks, tuned reservoirs against the baselines "
            f"({title}):",
            "",
            "| model | mean rank |",
            "|:---|---:|",
        ]
        L += [f"| {m} | {v:.2f} |" for m, v in rk.items()]
        L.append("")

    (RESULTS / "reservoir_search.md").write_text("\n".join(L))
    print("wrote results/reservoir_search.md")

    # ---- figure: scale sweep + tuned-vs-apriori ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if not s1b.empty:
        axes[0].plot(
            g.index / math.pi, g.val, "-o", color="#DD8452", label="validation"
        )
        axes[0].plot(g.index / math.pi, g.test, "--s", color="#4C72B0", label="test")
        axes[0].axvline(
            tuned_cfg["scale"] / math.pi, color="grey", ls=":", label="selected"
        )
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("encoding scale / pi")
        axes[0].set_ylabel("MSE (log)")
        axes[0].set_title("Stage 1b — encoding scale dominates, then turns over")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
    s4 = _piv(convergence_runs)
    if not s4.empty:
        bl = base[base.arm == "conv"].pivot_table(
            index="display", columns="tag", values="mse_test", aggfunc="median"
        )
        bl = bl.reindex(columns=[t for t in TASKS if t in bl.columns])
        x = np.arange(len(TASKS))
        w = 0.2
        axes[1].bar(
            x - 1.5 * w, bl.loc["LSTM"], w, label="LSTM (classical)", color="#8C564B"
        )
        axes[1].bar(
            x - 0.5 * w,
            bl.loc["photonic-dQNN"],
            w,
            label="photonic-dQNN",
            color="black",
        )
        axes[1].bar(
            x + 0.5 * w,
            bl.loc["photonic-memRC"],
            w,
            label="memRC a-priori",
            color="#CCB974",
        )
        axes[1].bar(
            x + 1.5 * w, s4.loc["tunedMemRC"], w, label="memRC tuned", color="#4C72B0"
        )
        axes[1].set_yscale("log")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            [PRETTY[t] for t in TASKS], rotation=30, ha="right", fontsize=8
        )
        axes[1].set_ylabel("test MSE (log)")
        axes[1].set_title(
            "Convergence budget — tuning helps, but does not close the gap"
        )
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "reservoir_search.png", dpi=150)
    print("wrote results/reservoir_search.png")


if __name__ == "__main__":
    main()
