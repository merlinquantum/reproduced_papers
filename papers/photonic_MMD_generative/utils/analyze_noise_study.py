"""Phase 04 M3: aggregate and plot the MerLin noise / detector study.

Usage::

    python papers/photonic_MMD_generative/utils/analyze_noise_study.py <noise_study sweep_dir>

Reads every run's metrics.json. For each size (m, n), trained model (QCBM, B5),
indistinguishability V, transmittance T and detector (pnr, threshold):
mean ± std over seeds of the exact test MMD² and of the postselection rate.
References from the same runs (same kernel, same detector reading):
- B5 native: the distinguishable twin evaluated at V = 0 (its own physics);
- classical baselines: marginal sampler (B4) and uniform random, sample-based
  (gaussian_mmd_raw for PNR, gaussian_mmd_binarized for threshold);
- test-to-test noise level.
Writes results/phase4_noise_study.json, .md and .png/.pdf (and copies the
figure into the sweep folder).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from plot_common import (
    RESULTS_DIR,
    SERIES,
    TEXT,
    load_sweep_runs,
    plt,
    save,
    style_axes,
)


def stats(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [float(v) for v in arr],
    }


def main() -> None:
    """Aggregate, tabulate and plot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    sweep_dir = parser.parse_args().sweep_dir.resolve()
    grid = defaultdict(lambda: defaultdict(list))
    refs = defaultdict(lambda: defaultdict(list))
    for run in load_sweep_runs(sweep_dir):
        label = run["candidate"]["label"]  # qcbm_m12, b5_m16, ...
        model, size = label.split("_")
        for row in run["metrics"]["noise_study"]:
            key = (
                size,
                model,
                row["indistinguishability"],
                row["transmittance"],
                row["detector"],
            )
            grid[key]["exact_mmd"].append(row["exact_mmd"])
            grid[key]["postselection_rate"].append(row["postselection_rate"])
        sample = run["metrics"]["sample_metrics"]
        for name in ("marginal", "random"):
            refs[(size, name, "pnr")]["v"].append(sample[name]["gaussian_mmd_raw"])
            refs[(size, name, "threshold")]["v"].append(
                sample[name]["gaussian_mmd_binarized"]
            )
        refs[(size, "test_to_test", "pnr")]["v"].append(
            sample["test_to_test"]["gaussian_mmd_raw"]
        )
    table = [
        {
            "size": k[0],
            "model": k[1],
            "indistinguishability": k[2],
            "transmittance": k[3],
            "detector": k[4],
            "exact_mmd": stats(v["exact_mmd"]),
            "postselection_rate": stats(v["postselection_rate"]),
        }
        for k, v in sorted(grid.items())
    ]
    references = [
        {"size": k[0], "reference": k[1], "detector": k[2], **stats(v["v"])}
        for k, v in sorted(refs.items())
    ]
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "phase4_noise_study.json").write_text(
        json.dumps({"grid": table, "references": references}, indent=2)
    )

    lines = [
        "# Phase 04 M3 — MerLin noise / detector study",
        "",
        f"Source: `{sweep_dir.name}` (exact MerLin SLOS, Gaussian sigma=1, 5 seeds). Exact test MMD²: median [min–max] over seeds",
        "(seeds are bimodal: some runs end in a poor local optimum; a Phase 07 rerun without early stopping does not rescue them). Transmittance T",
        "changes only the postselection rate: uniform loss followed by postselection on n photons leaves the",
        "distribution unchanged.",
        "",
    ]
    for size in sorted({r["size"] for r in table}):
        lines += [
            f"## {size}",
            "",
            "| Model | T | Detector | "
            + " | ".join(
                f"V={v:g}"
                for v in sorted(
                    {r["indistinguishability"] for r in table}, reverse=True
                )
            )
            + " | Postselection (V=1) |",
            "|---|---|---|"
            + "---|" * (len({r["indistinguishability"] for r in table}) + 1),
        ]
        for model in ("qcbm", "b5"):
            for T in sorted({r["transmittance"] for r in table}, reverse=True):
                for det in ("pnr", "threshold"):
                    rows = {
                        r["indistinguishability"]: r
                        for r in table
                        if (r["size"], r["model"], r["transmittance"], r["detector"])
                        == (size, model, T, det)
                    }
                    cells = [
                        f"{rows[v]['exact_mmd']['median']:.2e} [{rows[v]['exact_mmd']['min']:.1e}–{rows[v]['exact_mmd']['max']:.1e}]"
                        for v in sorted(rows, reverse=True)
                    ]
                    lines.append(
                        f"| {model} | {T:g} | {det} | "
                        + " | ".join(cells)
                        + f" | {rows[1.0]['postselection_rate']['mean']:.3f} |"
                    )
        lines.append("")
        for ref in (r for r in references if r["size"] == size):
            lines.append(
                f"- {ref['reference']} ({ref['detector']}): {ref['mean']:.2e} ± {ref['std']:.1e}"
            )
        lines.append("")
    (RESULTS_DIR / "phase4_noise_study.md").write_text("\n".join(lines))

    sizes = sorted({r["size"] for r in table})
    fig, axes = plt.subplots(
        2, len(sizes), figsize=(6 * len(sizes), 7.5), sharex=True, squeeze=False
    )
    for col, size in enumerate(sizes):
        for row_i, det in enumerate(("pnr", "threshold")):
            ax = axes[row_i][col]
            style_axes(ax)
            rows = sorted(
                (
                    r
                    for r in table
                    if (r["size"], r["model"], r["transmittance"], r["detector"])
                    == (size, "qcbm", 1.0, det)
                ),
                key=lambda r: r["indistinguishability"],
            )
            V = [r["indistinguishability"] for r in rows]
            per_seed = np.array(
                [r["exact_mmd"]["values"] for r in rows]
            )  # (len(V), seeds)
            for seed_curve in per_seed.T:
                ax.plot(
                    V, seed_curve, color=SERIES["model"][1], linewidth=0.8, alpha=0.45
                )
            ax.plot(
                V,
                np.median(per_seed, axis=1),
                color=SERIES["model"][1],
                linewidth=2.5,
                label="QCBM (median; thin: seeds)",
            )
            b5 = next(
                r
                for r in table
                if (
                    r["size"],
                    r["model"],
                    r["indistinguishability"],
                    r["transmittance"],
                    r["detector"],
                )
                == (size, "b5", 0.0, 1.0, det)
            )
            ax.axhline(
                b5["exact_mmd"]["median"],
                color=SERIES["b5"][1],
                linewidth=2,
                label="B5 twin (native, V=0; median)",
            )
            for name in ("marginal", "random"):
                ref = next(
                    r
                    for r in references
                    if (r["size"], r["reference"], r["detector"]) == (size, name, det)
                )
                ax.axhline(
                    ref["mean"],
                    color=SERIES[name][1],
                    linewidth=1.5,
                    label=SERIES[name][0],
                )
            ax.set_yscale("log")
            ax.set_title(
                f"{size.replace('m', 'm=')} · {'photon-number-resolving' if det == 'pnr' else 'threshold (collision-free postselection)'}",
                fontsize=9,
                color=TEXT,
            )
            if row_i == 1:
                ax.set_xlabel("Source indistinguishability V", color=TEXT, fontsize=9)
        axes[0][0].set_ylabel(
            "Exact test MMD² (lower is better)", color=TEXT, fontsize=9
        )
        axes[1][0].set_ylabel(
            "Exact test MMD² (lower is better)", color=TEXT, fontsize=9
        )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Phase 04 M3 (MerLin adaptation, reduced scope) — trained QCBM under partial distinguishability and loss\n"
        "Boson-sampling data m=12 n=3 and m=16 n=4 · exact MerLin SLOS · 5 seeds · postselected on n detections · "
        "curves for T = 1, 0.9, 0.7 coincide after postselection (only the event rate changes)",
        fontsize=9,
        color=TEXT,
        y=1.03,
    )
    for ax in axes.flat:
        ax.tick_params(labelsize=8)
    paths = save(fig, sweep_dir, "phase4_noise_study")
    print(RESULTS_DIR / "phase4_noise_study.md", paths[-1])


if __name__ == "__main__":
    main()
