"""NARMA R^2 versus order (paper Fig. 3a analogue, E2) and the concentration grid (E1b).

Usage (from the paper folder)::

    python utils/plot_narma_concentration.py \
        --narma-runs outdir/narma_n8/run_A outdir/narma_n12/run_B outdir/narma_n16/run_C \
        --concentration-runs outdir/concentration/run_D outdir/concentration_scrambling/run_E \
        --scope partial

Writes ``results/narma_r2_vs_order.png``, ``results/concentration_variability.png`` and
``results/narma_concentration_summary.json``. Every value comes from ``metrics.json`` of
COMPLETED runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
NARMA_MODELS = (
    ("qelm|shadow_1e5_shots", "QELM, shadow noise 1e5 shots", "-o"),
    ("qelm|exact", "QELM, exact", ":"),
    ("degree2_window|none", "ridge, degree-2 window", "--s"),
    ("raw_window|none", "ridge, raw window", "--^"),
    ("esn|none", "ESN-400", "-.d"),
)
# Concentration settings excluded from figures and summaries: the J = pi/4 "scrambling" point
# of concentration.json is non-entangling under the Fig. 1 convention (LOG.md).
EXCLUDED_SETTINGS = {"scrambling"}


def _metrics(run_dir: Path) -> dict:
    status = json.loads((run_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{run_dir} is {status['status']}")
    return json.loads((run_dir / status["metrics_path"]).read_text())


def plot_narma(runs: list[Path], scope: str, outdir: Path) -> dict:
    summaries = {}
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), sharey=True)
    for ax, run in zip(axes, runs, strict=True):
        metrics = _metrics(run)
        orders = metrics["orders"]
        summaries[metrics["n_qubits"]] = metrics["summary"]
        for key, label, style in NARMA_MODELS:
            entry = metrics["summary"][key]
            ax.errorbar(
                orders,
                entry["r2_mean"],
                yerr=entry["r2_std"],
                fmt=style,
                ms=4,
                capsize=2,
                label=f"{label} (cum {entry['cumulative_r2_mean']:.2f})",
            )
        ax.axvline(metrics["window"], color="k", linestyle=":", linewidth=1)
        ax.set(
            title=f"{metrics['n_qubits']} qubits, {metrics['n_layers']} layers, L = {metrics['window']}",
            xlabel="NARMA order n",
            ylim=(-0.2, 1.05),
        )
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("Test R^2 (mean ± std over sequences x reservoirs)")
    fig.suptitle(
        f"NARMA-n vs order (Fig. 3a analogue; {scope}); dotted line: window L = N/2"
    )
    fig.savefig(outdir / "narma_r2_vs_order.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summaries


def plot_concentration(runs: list[Path], scope: str, outdir: Path) -> dict:
    grid: dict = {}
    for run in runs:
        for key, entry in _metrics(run)["summary"].items():
            setting, n_part, layer_part = key.split("|")
            if setting in EXCLUDED_SETTINGS:
                continue
            n, layers = int(n_part.split("=")[1]), int(layer_part.split("=")[1])
            grid.setdefault(setting, {}).setdefault(n, {})[layers] = entry
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, weight in zip(axes, ("w1", "w2"), strict=True):
        for setting, style in zip(sorted(grid), ("-o", "--s"), strict=False):
            for n, by_layer in sorted(grid[setting].items()):
                layers = sorted(by_layer)
                ax.errorbar(
                    layers,
                    [
                        by_layer[depth][f"{weight}_variability_mean_avg"]
                        for depth in layers
                    ],
                    yerr=[
                        by_layer[depth][f"{weight}_variability_mean_std"]
                        for depth in layers
                    ],
                    fmt=style,
                    capsize=2,
                    label=f"{setting}, N = {n}",
                )
        ax.set(yscale="log", xlabel="layers", title=f"weight-{weight[1]} Paulis")
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("mean Var_u[<R_i>], u ~ Unif(-1, 1)")
    fig.suptitle(f"Output variability vs depth and size (E1b; {scope})")
    fig.savefig(outdir / "concentration_variability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--narma-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--concentration-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--outdir", type=Path, default=PAPER_DIR / "results")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": args.scope,
        "narma_runs": [str(r) for r in args.narma_runs],
        "concentration_runs": [str(r) for r in args.concentration_runs],
        "narma": plot_narma(args.narma_runs, args.scope, args.outdir),
        "concentration": plot_concentration(
            args.concentration_runs, args.scope, args.outdir
        ),
    }
    (args.outdir / "narma_concentration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
