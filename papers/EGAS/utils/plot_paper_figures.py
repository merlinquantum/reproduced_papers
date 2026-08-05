"""Regenerate every figure of the paper (Figs 1-7 + Table I) from saved run artifacts.

Reads structured `metrics.json` (+ `accuracies.npz`) from each dataset run directory; never
hardcodes run dirs (globs `outdir/<DS>/run_*/`). Writes PNGs to `results/`.

Paper figure map:
  Table I -> table1_wasserstein.png        (W1 reproduced vs paper)
  Fig 1   -> fig1_tracedist_vs_w1.png       (trace distance vs input W1, ZZ map)
  Fig 2   -> fig2_egas_schematic.png        (EGAS training-loop schematic)
  Fig 3   -> fig3_deltaE_per_candidate.png  (bias-refinement ΔE per candidate, PW)
  Fig 4   -> fig4_deltaE_groups.png         (group-mean ΔE across datasets, G vs B)
  Fig 5   -> fig5_win_tie_loss.png          (W/T/L vs classical linear, per dataset)
  Fig 6   -> fig6_iqr.png                   (embedding-sensitivity IQR per dataset)
  Fig 7   -> fig7_accuracy_heatmap.png      (embedding-wise QKSVM accuracy heatmap)
"""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

PAPER_W1 = {
    "PW": 5.2380,
    "WDGV1": 5.1570,
    "DB": 13.9108,
    "WC": 10.8562,
    "WQ": 3.0112,
    "MGT": 3.3036,
    "EGSSD": 3.5619,
}
EGAS_DATASETS = ["PW", "WQ", "MGT"]  # datasets actually run (reduced scope)


def _latest(pattern):
    hits = sorted(glob(pattern))
    return hits[-1] if hits else None


def load_egas():
    """Return {ds: (metrics_dict, npz_or_None)} for each EGAS run."""
    out = {}
    for ds in EGAS_DATASETS:
        mj = _latest(str(ROOT / f"outdir/{ds}/run_*/metrics.json"))
        if not mj:
            continue
        m = json.load(open(mj))
        npz_path = Path(mj).with_name("accuracies.npz")
        npz = np.load(npz_path) if npz_path.exists() else None
        out[ds] = (m, npz)
    return out


def _save(fig, name):
    p = RESULTS / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.name)


# ---------------------------------------------------------------- Table I
def table1():
    mj = _latest(str(ROOT / "outdir/run_*/metrics.json"))
    # find the wasserstein run specifically
    for f in sorted(glob(str(ROOT / "outdir/run_*/metrics.json"))):
        d = json.load(open(f))
        if d.get("task") == "wasserstein":
            mj = f
    d = json.load(open(mj))["results"]
    names = [n for n in d if "w1" in d[n]]
    repro = [d[n]["w1"] for n in names]
    paper = [PAPER_W1.get(n, np.nan) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, repro, 0.4, label="reproduced")
    ax.bar(x + 0.2, paper, 0.4, label="paper (Table I)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("1-Wasserstein distance")
    ax.set_title("Table I — input-space class-conditional W1")
    ax.legend()
    _save(fig, "table1_wasserstein.png")


# ---------------------------------------------------------------- Fig 1
def fig1():
    f = None
    for c in sorted(glob(str(ROOT / "outdir/run_*/metrics.json"))):
        if json.load(open(c)).get("task") == "fig1":
            f = c
    if not f:
        print("Fig1: no fig1 run found")
        return
    res = json.load(open(f))["results"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, dd in res.items():
        ax.plot(dd["w1"], dd["trace_dist"], "o-", label=f"ZZ {key}")
    ax.set_xlabel("input W1 distance")
    ax.set_ylabel("trace distance")
    ax.set_title("Fig 1 — trace distance vs input W1 (saturating)")
    ax.legend()
    _save(fig, "fig1_tracedist_vs_w1.png")


# ---------------------------------------------------------------- Fig 2 (schematic)
def fig2():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    boxes = [
        (0.02, "GPT\n(θ)"),
        (0.19, "sample M\ntoken seqs"),
        (0.37, "embedding\ncircuits Φ_s"),
        (0.55, "pairwise-fidelity\nenergy E(s)"),
        (0.74, "replay buffer\n(top/mid/bottom)"),
        (0.90, "logit-matching\nupdate (Eq.10)"),
    ]
    for x, t in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, 0.4), 0.135, 0.3, fill=True, facecolor="#cfe8ff", edgecolor="black"
            )
        )
        ax.text(x + 0.0675, 0.55, t, ha="center", va="center", fontsize=8.5)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 0.135
        x1 = boxes[i + 1][0]
        ax.annotate(
            "", xy=(x1, 0.55), xytext=(x0, 0.55), arrowprops={"arrowstyle": "->"}
        )
    # feedback arrow update -> GPT
    ax.annotate(
        "",
        xy=(0.0875, 0.4),
        xytext=(0.9675, 0.4),
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=0.3",
            "color": "gray",
        },
    )
    ax.text(
        0.5,
        0.16,
        "update p_θ(s) ∝ exp(-γ·w_sum) toward low energy",
        ha="center",
        fontsize=8.5,
        color="gray",
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 0.85)
    ax.set_title("Fig 2 — EGAS training loop (schematic)")
    _save(fig, "fig2_egas_schematic.png")


# ---------------------------------------------------------------- Fig 3
def fig3(egas):
    if "PW" not in egas:
        print("Fig3: PW run missing")
        return
    # prefer a dedicated repeat-run artifact if present (with error bars)
    rep = _latest(str(ROOT / "outdir/fig3_PW/fig3_data.json"))
    fig, ax = plt.subplots(figsize=(9, 4))
    if rep:
        d = json.load(open(rep))
        labels = [f"G{i + 1}" for i in range(len(d["G_mean"]))] + [
            f"B{i + 1}" for i in range(len(d["B_mean"]))
        ]
        means = d["G_mean"] + d["B_mean"]
        stds = d["G_std"] + d["B_std"]
        ax.bar(
            range(len(means)),
            means,
            yerr=stds,
            capsize=3,
            color=["#1f77b4"] * len(d["G_mean"]) + ["#d62728"] * len(d["B_mean"]),
        )
        sub = f"(PW; mean ± std over {d.get('repeats', '?')} refinement repeats)"
    else:
        m = egas["PW"][0]
        g = m["delta_E"]["G"]
        b = m["delta_E"]["B"]
        labels = [f"G{i + 1}" for i in range(len(g))] + [
            f"B{i + 1}" for i in range(len(b))
        ]
        means = g + b
        ax.bar(
            range(len(means)), means, color=["#1f77b4"] * len(g) + ["#d62728"] * len(b)
        )
        sub = (
            "(PW; single refinement run, reduced — see fig3 repeats run for error bars)"
        )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel("ΔE  (E_before − E_after)")
    ax.set_title(f"Fig 3 — bias-refinement surrogate-energy reduction\n{sub}")
    _save(fig, "fig3_deltaE_per_candidate.png")


# ---------------------------------------------------------------- Fig 4
def fig4(egas):
    ds = [d for d in EGAS_DATASETS if d in egas]
    ds.sort(key=lambda d: egas[d][0]["w1"])
    gm = [np.mean(egas[d][0]["delta_E"]["G"]) for d in ds]
    gs = [np.std(egas[d][0]["delta_E"]["G"]) for d in ds]
    bm = [np.mean(egas[d][0]["delta_E"]["B"]) for d in ds]
    bs = [np.std(egas[d][0]["delta_E"]["B"]) for d in ds]
    x = np.arange(len(ds))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        x - 0.05, gm, yerr=gs, fmt="o", capsize=4, label="G group", color="#1f77b4"
    )
    ax.errorbar(
        x + 0.05, bm, yerr=bs, fmt="s", capsize=4, label="B group", color="#d62728"
    )
    ax.axhline(0, ls="--", color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ds)
    ax.set_ylabel("mean ΔE")
    ax.set_title("Fig 4 — group-wise mean surrogate-energy reduction across datasets")
    ax.legend()
    _save(fig, "fig4_deltaE_groups.png")


# ---------------------------------------------------------------- Fig 5
def fig5(egas):
    ds = [d for d in EGAS_DATASETS if d in egas]
    n = len(ds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, ds):
        m = egas[name][0]
        models = [
            ("G*", m["G"][m["G_star_idx"]]["wtl_vs_linear"]),
            ("G*(Bias)", m["G_bias"][m["G_bias_star_idx"]]["wtl_vs_linear"]),
            ("B*", m["B"][m["B_star_idx"]]["wtl_vs_linear"]),
            ("B*(Bias)", m["B_bias"][m["B_bias_star_idx"]]["wtl_vs_linear"]),
            ("NQE", m["baselines"]["NQE"]["wtl_vs_linear"]),
            ("ZZ", m["baselines"]["ZZ"]["wtl_vs_linear"]),
        ]
        labels = [a for a, _ in models]
        wins = [w["win"] for _, w in models]
        ties = [w["tie"] for _, w in models]
        loss = [w["lose"] for _, w in models]
        x = np.arange(len(labels))
        ax.bar(x, wins, color="#2c7fb8", label="win")
        ax.bar(x, ties, bottom=wins, color="#f4d03f", label="tie")
        ax.bar(
            x,
            loss,
            bottom=np.array(wins) + np.array(ties),
            color="#d62728",
            label="lose",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_title(f"{name} (W1={m['w1']:.2f})")
    axes[0].set_ylabel("# splits (of {})".format(egas[ds[0]][0]["n_splits"]))
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Fig 5 — Win/Tie/Loss vs classical linear SVM")
    _save(fig, "fig5_win_tie_loss.png")


# ---------------------------------------------------------------- Fig 6
def fig6(egas):
    ds = [d for d in EGAS_DATASETS if d in egas]
    ds.sort(key=lambda d: egas[d][0]["w1"])
    iqr = [egas[d][0]["embedding_sensitivity_IQR"] for d in ds]
    w1 = [egas[d][0]["w1"] for d in ds]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(np.arange(len(ds)), iqr, color="#7fbf7b")
    for b, w in zip(bars, w1):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.002,
            f"W1={w:.2f}",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(np.arange(len(ds)))
    ax.set_xticklabels(ds)
    ax.set_ylabel("IQR of mean test accuracy")
    ax.set_title("Fig 6 — embedding-sensitivity IQR per dataset (grows with W1)")
    _save(fig, "fig6_iqr.png")


# ---------------------------------------------------------------- Fig 7
def fig7(egas):
    ds = [d for d in EGAS_DATASETS if d in egas]
    ds.sort(key=lambda d: egas[d][0]["w1"])
    # build row labels from the first dataset that has an npz
    sample = next((egas[d][1] for d in ds if egas[d][1] is not None), None)
    if sample is None:
        print("Fig7: no npz arrays found")
        return
    nG = sample["accG"].shape[0]
    rows = (
        ["Classical(Linear)", "Classical(RBF)", "NQE", "ZZ"]
        + [f"G{i + 1:02d}(Bias)" for i in range(nG)]
        + [f"G{i + 1:02d}" for i in range(nG)]
        + [f"B{i + 1:02d}(Bias)" for i in range(nG)]
        + [f"B{i + 1:02d}" for i in range(nG)]
    )
    mat = np.full((len(rows), len(ds)), np.nan)
    for j, name in enumerate(ds):
        m, z = egas[name]
        col = [
            m["baselines"]["classical_linear"]["mean_acc"],
            m["baselines"]["classical_rbf"]["mean_acc"],
            m["baselines"]["NQE"]["mean_acc"],
            m["baselines"]["ZZ"]["mean_acc"],
        ]
        if z is not None:
            col += (
                list(z["accGb"].mean(1))
                + list(z["accG"].mean(1))
                + list(z["accBb"].mean(1))
                + list(z["accB"].mean(1))
            )
        mat[: len(col), j] = col
    fig, ax = plt.subplots(figsize=(1.6 * len(ds) + 3, 0.34 * len(rows) + 1))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.45, vmax=1.0)
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels(ds)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    for i in range(len(rows)):
        for j in range(len(ds)):
            if not np.isnan(mat[i, j]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if mat[i, j] < 0.8 else "black",
                )
    fig.colorbar(im, ax=ax, label="mean test accuracy")
    ax.set_title("Fig 7 — embedding-wise QKSVM test accuracy")
    _save(fig, "fig7_accuracy_heatmap.png")


if __name__ == "__main__":
    egas = load_egas()
    print("loaded EGAS datasets:", list(egas))
    table1()
    fig1()
    fig2()
    fig3(egas)
    fig4(egas)
    fig5(egas)
    fig6(egas)
    fig7(egas)
    print("done.")
