"""Build the Phase 03 result tables from run artifacts and the authors' cached arrays.

Usage (from papers/power_of_QNN):
    python utils/summarize_results.py --runs runs.json --output results/summary.md

``runs.json`` maps experiment names to run directories, for example
``{"effdim_d40_original": "outdir/run_20261021-072045", ...}``. Every listed run must
be COMPLETED. Values are read at full precision from ``metrics.json`` and the
``.npz`` arrays; the markdown output only rounds for display.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from figure_io import (
    PAPER_TABLE1_FISHER_RAO,
    PAPER_TABLE1_LOSS,
    UPSTREAM,
    load_metrics,
    paper_fig3a_curves,
    paper_fig6_points,
)

N_COLUMNS = (0, 3, 6, 10)  # n = 1e3, 1e4, 1e5, 1e6 in the paper grid


def _fmt(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def effective_dimension_table(metrics: dict, title: str) -> list[str]:
    n_values = metrics["n_values"]
    header = " | ".join(f"n={n_values[i]:.0e}" for i in N_COLUMNS)
    lines = [
        f"### {title}",
        "",
        f"| Model | d | {header} | n={n_values[-1]:.0e} | mean rank | frac eig < 1e-10 | mean tr F (unnormalized) |",
        "|---|---|" + "---|" * (len(N_COLUMNS) + 4),
    ]
    for name, entry in metrics["models"].items():
        mean, std = (
            entry["normalized_effective_dimension_mean"],
            entry["normalized_effective_dimension_std"],
        )
        cells = " | ".join(_fmt(mean[i], std[i]) for i in N_COLUMNS)
        lines.append(
            f"| {name} | {entry['d']} | {cells} | {_fmt(mean[-1], std[-1])} | {entry['mean_rank_mean']:.2f} | "
            f"{entry['fraction_near_zero_mean']:.3f} | {entry['mean_trace_mean']:.4g} |"
        )
    lines.append(
        f"\nDraw seeds: {metrics['fisher']['draw_seeds']} (mean ± std over draws); theta ~ U[{metrics['fisher']['theta_low']}, {metrics['fisher']['theta_high']}].\n"
    )
    return lines


def paper_fig3a_table() -> list[str]:
    curves = paper_fig3a_curves()
    lines = [
        "### Paper Fig. 3a source data (authors' cache)",
        "",
        "| Curve | "
        + " | ".join(f"n={curves['paper QNN'][0][i]:.0e}" for i in N_COLUMNS)
        + " |",
        "|---|" + "---|" * len(N_COLUMNS),
    ]
    for label, (_, values) in curves.items():
        lines.append(
            f"| {label} | " + " | ".join(f"{values[i]:.4f}" for i in N_COLUMNS) + " |"
        )
    return lines + [""]


def spectrum_shape_table(
    run_dirs: dict[str, Path], metrics_by_run: dict[str, dict]
) -> list[str]:
    """Normalized-spectrum shape per model (seed-0 F_hat, eigenvalues pooled over theta)."""
    lines = [
        "### F_hat spectrum shape (draw seed 0, eigenvalues pooled over the 100 theta)",
        "",
        "| Run | Model | d | frac eig < 0.1 | frac eig < 0.01 | 99th pct | max | mean tr F / d (unnormalized) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run_name, metrics in metrics_by_run.items():
        for name, entry in metrics["models"].items():
            with np.load(run_dirs[run_name] / entry["arrays"]) as arrays:
                eigenvalues = np.clip(
                    np.linalg.eigvalsh(arrays["f_hat_seed0"]), 0.0, None
                ).ravel()
            lines.append(
                f"| {run_name} | {name} | {entry['d']} | {np.mean(eigenvalues < 0.1):.3f} | {np.mean(eigenvalues < 0.01):.3f} | "
                f"{np.percentile(eigenvalues, 99):.2f} | {eigenvalues.max():.2f} | {entry['mean_trace_mean'] / entry['d']:.3e} |"
            )
    return lines + [""]


def topology_table(metrics: dict, title: str) -> list[str]:
    lines = [
        f"### {title}",
        "",
        f"Selected (predeclared rule): {metrics['selected']}",
        "",
        "| Topology | mean rank | ed/d at n=1e5 |",
        "|---|---|---|",
    ]
    for candidate in sorted(
        metrics["candidates"], key=lambda c: (-c["mean_rank"], -c["selection_value"])
    ):
        lines.append(
            f"| {candidate['sizes']} | {candidate['mean_rank']:.2f} | {candidate['selection_value']:.4f} |"
        )
    return lines + [""]


def sensitivity_table(metrics: dict) -> list[str]:
    sizes = next(iter(metrics["models"].values()))["sample_sizes"]
    lines = [
        "### Fig. 8 sensitivity: ed/d at n=1e6 vs N_theta = N_x (mean ± std over 10 draws)",
        "",
        "| Model | d | " + " | ".join(str(s) for s in sizes) + " |",
        "|---|---|" + "---|" * len(sizes),
    ]
    for name, entry in metrics["models"].items():
        mean = np.asarray(entry["normalized_effective_dimension_mean"])[:, 10]
        std = np.asarray(entry["normalized_effective_dimension_std"])[:, 10]
        lines.append(
            f"| {name} | {entry['d']} | "
            + " | ".join(_fmt(m, s, 3) for m, s in zip(mean, std))
            + " |"
        )
    return lines + [""]


def training_table(metrics: dict, title: str) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Model | d | optimizer / gradient / init | final loss (mean ± std) | paper final loss | final train acc | frac reaching CE ≤ 0.3 | median iters to 0.3 | Fisher-Rao norm | paper FR norm | mean ‖θ‖² |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in metrics["models"].items():
        training = entry["effective_training"]
        init = training["init"]
        fr = (
            "n/a"
            if entry["fisher_rao_norm_mean"] is None
            else _fmt(entry["fisher_rao_norm_mean"], entry["fisher_rao_norm_std"], 2)
        )
        median = entry["median_iterations_to_threshold"]
        lines.append(
            f"| {name} | {entry['d']} | {training['optimizer']} / {training['gradient']} / U[{init['low']:.3g}, {init['high']:.3g}] | "
            f"{_fmt(entry['final_loss_mean'], entry['final_loss_std'])} | {PAPER_TABLE1_LOSS.get(name, 'n/a')} | {entry['final_accuracy_mean']:.3f} | "
            f"{entry['fraction_reaching_threshold']:.2f} | {'n/a' if median is None else f'{median:.0f}'} | {fr} | {PAPER_TABLE1_FISHER_RAO.get(name, 'n/a')} | "
            f"{entry['final_theta_norm_squared_mean']:.2f} |"
        )
    lines.append(
        f"\nSeeds: {metrics['seeds'][0]}..{metrics['seeds'][-1]} ({len(metrics['seeds'])}); full batch; {metrics['training']['iterations']} iterations; lr {metrics['training']['lr']}.\n"
    )
    return lines


def generalisation_table(metrics: dict) -> list[str]:
    fractions, paper_mean, paper_std = paper_fig6_points()
    lines = [
        "### Fig. 6: effective dimension at the trained theta vs label randomization",
        "",
        "| Fraction overwritten | realized flip fraction | ed/d (ours, mean ± std) | ed/d (authors' cache, first 10 runs) | final train loss | test loss | test acc |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry, paper_m, paper_s in zip(metrics["fractions"], paper_mean, paper_std):
        lines.append(
            f"| {entry['fraction']:.1f} | {entry['realized_flip_fraction_mean']:.3f} | {_fmt(entry['normalized_effective_dimension_mean'], entry['normalized_effective_dimension_std'])} | "
            f"{_fmt(paper_m, paper_s)} | {entry['final_train_loss_mean']:.4f} | {entry['test_loss_mean']:.3f} | {entry['test_accuracy_mean']:.3f} |"
        )
    return lines + [
        f"\nRuns per fraction: {metrics['training']['seeds']['count']}; {metrics['training']['iterations']} Adam iterations, lr {metrics['training']['lr']}.\n"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = {
        name: Path(path)
        for name, path in json.loads(args.runs.read_text(encoding="utf-8")).items()
    }
    metrics = {name: load_metrics(path) for name, path in run_dirs.items()}
    lines = [
        "# power_of_QNN Phase 03 result tables",
        "",
        f"Generated by `utils/summarize_results.py --runs {args.runs}` from: "
        + ", ".join(f"`{name}` = `{path}`" for name, path in run_dirs.items()),
        f"Upstream reference cache: `{UPSTREAM.relative_to(UPSTREAM.parents[3])}`.",
        "",
    ]
    effective = {
        "effdim_d40_original": "Fig. 2 / 3a models, theta ~ U[0,1] (authors' code)",
        "effdim_d40_paper_domain": "A-01: theta ~ U[-1,1] (paper text)",
        "effdim_d40_easy_ry": "A-03: easy model encodings",
        "spectrum_scaling_original": "Fig. 9: s_in = 6, 8, 10",
    }
    for name, title in effective.items():
        if name in metrics:
            lines += effective_dimension_table(metrics[name], f"{title} (`{name}`)")
    lines += paper_fig3a_table()
    spectral = {
        name: metrics[name]
        for name in ("effdim_d40_original", "spectrum_scaling_original")
        if name in metrics
    }
    if spectral:
        lines += spectrum_shape_table(run_dirs, spectral)
    for name in ("topology_search_d40_classical", "topology_search_d40_relu_classical"):
        if name in metrics:
            lines += topology_table(metrics[name], f"Topology search `{name}`")
    if "sensitivity_classical" in metrics:
        lines += sensitivity_table(metrics["sensitivity_classical"])
    for name in (
        "training_iris_original",
        "training_iris_finite_difference",
        "training_iris_paper_domain",
    ):
        if name in metrics:
            lines += training_table(metrics[name], f"Training `{name}`")
    if "generalisation_original" in metrics:
        lines += generalisation_table(metrics["generalisation_original"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
