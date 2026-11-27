"""Fig. 3b / 3c / 3d / 3e / 3f reproductions: one dot plot of test MMD^2 per comparison sweep.

Usage::

    python papers/photonic_MMD_generative/utils/plot_comparison.py <figure> <sweep_dir> [<sweep_dir> ...]

``<figure>`` is one of fig3b, fig3cd, fig3e, fig3f. Several sweep folders are
merged (fig3b: main sweep + butterfly retry; fig3cd: Haar + butterfly sweeps).
The y value is the sample-based test MMD^2 with the run's kernel
(``walsh_mmd``), the same estimator for the QCBM and every baseline; models
that cannot be sampled (non-unitary, D3) are shown with the paper's estimator
value and a hollow marker. Points: mean ± std over seeds; small dots: seeds.
Baselines are drawn as horizontal reference lines per panel (mean over the
runs that share the panel's kernel).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from plot_common import SERIES, TEXT, load_sweep_runs, plt, save, style_axes

TITLES = {
    "fig3b": "Fig. 3b reproduction — sushi m=100, n=10, Gaussian σ=3, near-identity init, by ansatz",
    "fig3cd": "Fig. 3c/3d reproduction — movies m=100, n=10 (rebuilt, unverified), by initialization and σ\n"
    "Partial comparison: Haar ran on gpu-reproduced-papers-2, both butterflies on gpu-reproduced-papers-1 (same L4 model); "
    "code identity across these launches is not verifiable from artifacts",
    "fig3e": "Fig. 3e reproduction — movies m=100, n=10 (rebuilt, unverified), Haar, by kernel",
    "fig3f": "Fig. 3f reproduction — bioinformatics m=100, n=10 (rebuilt, unverified), Haar, weak-unbiased init, by kernel",
}
BASELINE_ORDER = ("rbm", "marginal", "random")


def model_value(run):
    sample = run["metrics"].get("sample_metrics", {})
    if "model" in sample:
        return sample["model"]["walsh_mmd"], True
    return run["metrics"]["test_mmd_estimator"], False


def panel_key(figure, candidate):
    if figure == "fig3cd":
        variant = candidate.get("model.butterfly_variant")
        ansatz = "Haar" if variant is None else f"Butterfly ({variant})"
        return (
            ansatz,
            candidate["label"].rsplit("_sigma", 1)[0],
            candidate["label"].rsplit("_sigma", 1)[1],
        )
    return "all", candidate["label"], None


def main() -> None:
    """Plot one comparison figure from one or more sweeps."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("figure", choices=sorted(TITLES))
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    runs = [r for d in args.sweep_dirs for r in load_sweep_runs(d.resolve())]
    points = defaultdict(lambda: defaultdict(list))  # panel -> (label, sigma) -> values
    deployable = {}
    baselines = defaultdict(
        lambda: defaultdict(list)
    )  # panel/sigma -> baseline -> values
    for run in runs:
        panel, label, sigma = panel_key(args.figure, run["candidate"])
        value, ok = model_value(run)
        points[panel][(label, sigma)].append(value)
        deployable[(panel, label, sigma)] = ok
        for name in BASELINE_ORDER:
            entry = run["metrics"].get("sample_metrics", {}).get(name)
            if entry:
                key = (
                    (panel, sigma)
                    if args.figure == "fig3cd"
                    else (panel, label if args.figure in ("fig3e", "fig3f") else None)
                )
                baselines[key][name].append(entry["walsh_mmd"])
    panels = sorted(points)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.2 * len(panels) + 1.5, 4.4),
        sharey=True,
        squeeze=False,
    )
    for ax, panel in zip(axes[0], panels, strict=True):
        style_axes(ax)
        keys = sorted(points[panel], key=lambda k: (k[1] or "", k[0]))
        for x, key in enumerate(keys):
            values = np.array(points[panel][key])
            ok = deployable[(panel, *key)]
            color = SERIES["model"][1]
            ax.scatter(
                np.full(values.size, x),
                np.clip(values, 1e-7, None),
                s=9,
                color=color,
                alpha=0.35,
                linewidths=0,
            )
            ax.errorbar(
                x,
                max(values.mean(), 1e-7),
                yerr=values.std(ddof=1) if values.size > 1 else None,
                fmt="o",
                ms=7,
                color=color,
                mfc=color if ok else "white",
                mec=color,
                mew=1.5,
                elinewidth=1.5,
                capsize=0,
            )
            for name in BASELINE_ORDER:
                bkey = (
                    (panel, key[1])
                    if args.figure == "fig3cd"
                    else (panel, key[0] if args.figure in ("fig3e", "fig3f") else None)
                )
                if bkey in baselines and baselines[bkey].get(name):
                    level = float(np.mean(baselines[bkey][name]))
                    ax.hlines(
                        level, x - 0.35, x + 0.35, color=SERIES[name][1], linewidth=2
                    )
        labels = [k[0] if k[1] is None else f"{k[0]}\nσ={k[1]}" for k in keys]
        ax.set_xticks(
            range(len(keys)),
            labels,
            rotation=60 if len(keys) > 6 else 20,
            ha="right",
            fontsize=7,
        )
        ax.set_yscale("log")
        if panel != "all":
            ax.set_title(panel, fontsize=9, color=TEXT)
    axes[0][0].set_ylabel("Test MMD² (lower is better)", color=TEXT, fontsize=9)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color=SERIES["model"][1],
            linestyle="",
            label="QCBM (sample-based)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color=SERIES["model"][1],
            mfc="white",
            linestyle="",
            label="QCBM, not unitary: paper estimator only",
        ),
    ]
    present = {n for per_key in baselines.values() for n, v in per_key.items() if v}
    handles += [
        plt.Line2D([], [], color=SERIES[n][1], linewidth=2, label=SERIES[n][0])
        for n in BASELINE_ORDER
        if n in present
    ]
    rotated = max(len(points[p]) for p in panels) > 6
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.28 if rotated else -0.12),
    )
    fig.suptitle(
        TITLES[args.figure]
        + "\nPaper settings · mean ± std over seeds · baselines as horizontal bars "
        "(absolute values are not comparable across kernels)",
        fontsize=9,
        color=TEXT,
        y=1.05,
    )
    for path in save(fig, args.sweep_dirs[0].resolve(), f"{args.figure}_comparison"):
        print(path)


if __name__ == "__main__":
    main()
