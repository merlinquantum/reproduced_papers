"""E1 summary table and paper Fig. 2 analogue from the four Pareto-search runs.

Usage (from the paper folder)::

    python utils/plot_pareto.py --runs outdir/pareto_n6/run_A outdir/pareto_n8/run_B ... \
        --scope "original size; our search budget"

Reads each run's ``metrics.json`` and ``trials.csv`` (status must be COMPLETED) and writes
``results/pareto_summary.{json,md}`` and ``results/pareto_fig2.png`` (copies in each run
directory). The support rule was declared before launch in LOG.md: Table I is "near the
front" in a projection when its Pareto rank is <= 1 or its normalised distance to the
front is <= 0.1.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
PROJECTIONS = (
    "fig2a_narma_noisy_vs_bond",
    "fig2a_text_definition",
    "fig2b_variability_vs_bond",
    "all_objectives",
)
# Fig. 2 Table I stars, read off the plot with calibrated log axes (LOG.md).
PAPER_STARS = {
    6: {"narma_noisy": 3.35, "bond_dimension": 8, "variability": 6.8e-3},
    8: {"narma_noisy": 4.45, "bond_dimension": 14, "variability": 6.7e-3},
    10: {"narma_noisy": 5.3, "bond_dimension": 28, "variability": 4.0e-3},
    12: {"narma_noisy": 6.25, "bond_dimension": 50, "variability": 4.0e-3},
}
COLORS = {6: "#C9B23F", 8: "#2E8B3E", 10: "#E15B6E", 12: "#4477AA"}


def _supported(entry: dict) -> bool:
    return entry["reference_rank"] <= 1 or entry["reference_front_distance"] <= 0.1


def _front(points: np.ndarray) -> np.ndarray:
    """Indices of non-dominated points (maximisation of both columns), sorted by x."""
    keep = [
        i
        for i, point in enumerate(points)
        if not np.any(np.all(points >= point, axis=1) & np.any(points > point, axis=1))
    ]
    return np.array(sorted(keep, key=lambda i: points[i, 0]))


def _cell(projections: dict, name: str) -> str:
    entry = projections[name]
    label = "near" if entry["near_front"] else "not near"
    return (
        f"{entry['reference_rank']}, {entry['reference_front_distance']:.3f} ({label})"
    )


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    status = json.loads((run_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{run_dir} is {status['status']}")
    metrics = json.loads((run_dir / status["metrics_path"]).read_text())
    with (run_dir / metrics["trials_path"]).open() as handle:
        rows = [
            {
                key: (value if key == "is_table_I" else float(value))
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]
    if len(rows) != metrics["n_trials"] or not all(
        np.isfinite(r["narma_noisy"]) for r in rows
    ):
        raise ValueError(f"{run_dir}: trial count or values invalid")
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--outdir", type=Path, default=PAPER_DIR / "results")
    args = parser.parse_args()

    summary: dict = {"scope": args.scope, "sizes": {}}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for run_dir in args.runs:
        metrics, rows = load_run(run_dir)
        n = int(metrics["n_qubits"])
        analysis = metrics["analysis"]
        summary["sizes"][n] = {
            "run": str(run_dir),
            "n_layers": metrics["n_layers"],
            "n_trials": metrics["n_trials"],
            "table_I": metrics["table_I_objectives"],
            "paper_star": PAPER_STARS.get(n),
            "projections": {
                name: {**analysis[name], "near_front": _supported(analysis[name])}
                for name in PROJECTIONS
            },
            "single_objective_narma_noisy": analysis["single_objective_narma_noisy"],
        }
        bond = np.array([r["bond_dimension"] for r in rows])
        narma = np.array([r["narma_noisy"] for r in rows])
        variability = np.array([r["variability"] for r in rows])
        for ax, x in ((axes[0], narma), (axes[1], variability)):
            ax.scatter(x, bond, s=8, alpha=0.25, color=COLORS[n])
            points = np.column_stack([x, bond])
            front = _front(points)
            ax.plot(
                x[front], bond[front], "-o", color=COLORS[n], ms=4, label=f"{n} qubits"
            )
            reference = next(i for i, r in enumerate(rows) if r["is_table_I"] == "True")
            ax.scatter(
                x[reference],
                bond[reference],
                marker="*",
                s=260,
                color="yellow",
                edgecolor="black",
                zorder=6,
            )
        star = PAPER_STARS[n]
        axes[0].scatter(
            star["narma_noisy"],
            star["bond_dimension"],
            marker="X",
            s=90,
            color=COLORS[n],
            edgecolor="black",
            zorder=7,
        )
        axes[1].scatter(
            star["variability"],
            star["bond_dimension"],
            marker="X",
            s=90,
            color=COLORS[n],
            edgecolor="black",
            zorder=7,
        )
        run_dir.mkdir(exist_ok=True)

    axes[0].set(
        xlabel="Cumulative NARMA-n score, n = 1..N (Gaussian noise 10^-1.5)",
        ylabel="Bond dimension",
        yscale="log",
        title="Fig. 2a analogue",
    )
    axes[1].set(xlabel="Variability", xscale="log", title="Fig. 2b analogue")
    axes[0].scatter(
        [],
        [],
        marker="*",
        s=200,
        color="yellow",
        edgecolor="black",
        label="Table I (ours)",
    )
    axes[0].scatter(
        [],
        [],
        marker="X",
        s=80,
        color="gray",
        edgecolor="black",
        label="Table I (paper star)",
    )
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Multi-objective search, 500 NSGA-II trials per size ({args.scope})")

    args.outdir.mkdir(parents=True, exist_ok=True)
    outputs = [args.outdir] + list(args.runs)
    for directory in outputs:
        fig.savefig(directory / "pareto_fig2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    lines = [
        f"# E1 Pareto summary ({args.scope})",
        "",
        "Near front = Pareto rank <= 1 or normalised front distance <= 0.1 (declared before launch).",
        "",
        "| N | layers | Table I noisy NARMA / bond / variability | paper star | Fig. 2a rank, dist | text-def rank, dist | Fig. 2b rank, dist | 4-D rank | NARMA-only optimum |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n, entry in sorted(summary["sizes"].items()):
        t, p, proj = entry["table_I"], entry["paper_star"], entry["projections"]

        lines.append(
            f"| {n} | {entry['n_layers']} | {t['narma_noisy']:.2f} / {t['bond_dimension']:.1f} / {t['variability']:.1e} | "
            f"{p['narma_noisy']} / {p['bond_dimension']} / {p['variability']:.1e} | {_cell(proj, 'fig2a_narma_noisy_vs_bond')} | "
            f"{_cell(proj, 'fig2a_text_definition')} | {_cell(proj, 'fig2b_variability_vs_bond')} | {proj['all_objectives']['reference_rank']} | "
            f"{entry['single_objective_narma_noisy']['best_value']:.2f} |"
        )
    (args.outdir / "pareto_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.outdir / "pareto_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
