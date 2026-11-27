"""Fig. 3a reproduction: test MMD^2 of the QCBM and baselines on boson-sampling data.

Usage::

    python papers/photonic_MMD_generative/utils/plot_fig3a.py <sweep_dir of fig3a_bs>

Three panels, one per metric (the paper does not say which scored each model):
Walsh features (what the QCBM is trained on), exact Gaussian kernel on raw
counts (the paper's sample-based metric for RBM/random), Gaussian kernel on
binarized samples. Points: mean ± std over seeds; small dots: seeds. Gray band:
test-to-test noise level (mean ± std). Log y axis.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from plot_common import NEUTRAL, SERIES, TEXT, load_sweep_runs, plt, save, style_axes

METRICS = (
    ("walsh_mmd", "Walsh (parity) features — training metric"),
    ("gaussian_mmd_raw", "Gaussian kernel, raw photon counts"),
    ("gaussian_mmd_binarized", "Gaussian kernel, binarized samples"),
)
ORDER = ("model", "b5", "rbm", "marginal", "random")


def collect(runs):
    """values[metric][series][sigma] -> list over seeds."""
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run in runs:
        sigma = run["candidate"]["kernel.sigma"]
        sample = run["metrics"]["sample_metrics"]
        is_b5 = run["candidate"]["label"].startswith("b5")
        for name, entry in sample.items():
            series = "b5" if (is_b5 and name == "model") else name
            for metric, _ in METRICS:
                if metric in entry:
                    values[metric][series][sigma].append(entry[metric])
    return values


def main() -> None:
    """Plot the Fig. 3a sweep."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    sweep_dir = parser.parse_args().sweep_dir.resolve()
    values = collect(load_sweep_runs(sweep_dir))
    sigmas = sorted(values["walsh_mmd"]["model"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    offsets = np.linspace(-0.24, 0.24, len(ORDER))
    for ax, (metric, title) in zip(axes, METRICS, strict=True):
        style_axes(ax)
        noise = values[metric]["test_to_test"]
        for i, sigma in enumerate(sigmas):
            if sigma in noise:
                level = abs(np.mean(noise[sigma])) + np.std(noise[sigma], ddof=1)
                ax.fill_between(
                    [i - 0.4, i + 0.4],
                    1e-7,
                    level,
                    color=NEUTRAL,
                    alpha=0.18,
                    linewidth=0,
                )
        for offset, series in zip(offsets, ORDER, strict=True):
            if series not in values[metric]:
                continue
            label, color = SERIES[series]
            for i, sigma in enumerate(sigmas):
                seeds = np.array(values[metric][series].get(sigma, []))
                if seeds.size == 0:
                    continue
                x = i + offset
                ax.scatter(
                    np.full(seeds.size, x),
                    np.clip(seeds, 1e-7, None),
                    s=9,
                    color=color,
                    alpha=0.35,
                    linewidths=0,
                )
                ax.errorbar(
                    x,
                    max(seeds.mean(), 1e-7),
                    yerr=seeds.std(ddof=1) if seeds.size > 1 else None,
                    fmt="o",
                    ms=6,
                    color=color,
                    ecolor=color,
                    elinewidth=1.5,
                    capsize=0,
                    mec="white",
                    mew=1.2,
                    label=label if i == 0 else None,
                )
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 2e-2)
        ax.set_xticks(range(len(sigmas)), [f"σ = {s:g}" for s in sigmas])
        ax.set_title(title, fontsize=9, color=TEXT)
    axes[0].set_ylabel("Test MMD² (lower is better)", color=TEXT, fontsize=9)
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=NEUTRAL, alpha=0.18))
    labels.append("Test-to-test noise level")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(
        "Fig. 3a reproduction — boson sampling m=256, n=16, Haar, near-identity init\n"
        "Paper settings · BS data regenerated (new unitary, 60% of samples with collisions) · "
        "mean ± std over 5 seeds (error bars that reach the axis floor have std > mean)",
        fontsize=9,
        color=TEXT,
        y=1.04,
    )
    for path in save(fig, sweep_dir, "fig3a_bs_benchmarks"):
        print(path)


if __name__ == "__main__":
    main()
