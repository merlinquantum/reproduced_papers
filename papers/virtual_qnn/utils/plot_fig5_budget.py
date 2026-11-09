"""Fig. 5 reproduction: VQEM Alg. 2 squared HS error vs sampling-overhead budget.

Usage::

    python3 papers/virtual_qnn/utils/plot_fig5_budget.py \
        papers/virtual_qnn/outdir/sweeps/e3_vqem_budget_noisy \
        papers/virtual_qnn/outdir/sweeps/e3_vqem_budget_ideal --results papers/virtual_qnn/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR.parent))

from virtual_qnn.lib import paper_values as paper  # noqa: E402
from virtual_qnn.utils import plot_style as style  # noqa: E402


def _curve(summary: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted(summary["candidates"], key=lambda c: c["grid_point"]["--vqem-budget"])
    budget = np.array([c["grid_point"]["--vqem-budget"] for c in rows])
    mean = np.array([c["aggregates"]["hs_squared"]["mean"] for c in rows])
    sd = np.array([c["aggregates"]["hs_squared"]["std"] for c in rows])
    return budget, mean, sd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("noisy_sweep_dir", type=Path)
    parser.add_argument("ideal_sweep_dir", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    style.apply_style()
    fig, ax = style.plt.subplots(figsize=(5.6, 3.8))
    for index, (label, sweep_dir, reference) in enumerate(
        (
            ("Noisy branches", args.noisy_sweep_dir, paper.FIG5_HS_NOISY),
            ("Noiseless branches", args.ideal_sweep_dir, paper.FIG5_HS_IDEAL),
        )
    ):
        budget, mean, sd = _curve(style.load_summary(sweep_dir))
        color, marker = style.SERIES[index + 1], style.MARKERS[index]
        ax.fill_between(
            budget,
            np.clip(mean - sd, 0, None),
            mean + sd,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            budget,
            mean,
            color=color,
            marker=marker,
            label=f"{label}: reproduction (3 seeds)",
        )
        ax.plot(
            paper.FIG5_BUDGET,
            reference,
            linestyle="none",
            marker=marker,
            markerfacecolor="none",
            markeredgecolor=color,
            markersize=9,
            markeredgewidth=1.4,
            label=f"{label}: paper",
        )
    ax.axvline(1.375, color=style.NEUTRAL, linestyle="--", linewidth=1.2)
    ax.text(1.385, 0.118, "SDP minimum 1.375", fontsize=8, color=style.TEXT_SECONDARY)
    ax.set_xlabel("Sampling-overhead budget γ")
    ax.set_ylabel("Squared HS distance to Φ⁺")
    ax.set_title("VQEM with a fixed budget, depolarizing p = 0.2 (Fig. 5)", loc="left")
    ax.legend(fontsize=8)
    style.scope_note(
        ax,
        "Scope: paper-accurate (reference hyperparameters), 3 seeds, exact simulation; band = ±1 std.",
    )
    paths = style.save(
        fig,
        "fig5_budget_sweep",
        [args.noisy_sweep_dir, args.ideal_sweep_dir, args.results],
    )
    print("\n".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
