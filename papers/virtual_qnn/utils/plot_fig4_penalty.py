"""Fig. 4 reproduction: VQEM Alg. 1 penalty sweep (amplitude damping q = 0.3).

Usage::

    python3 papers/virtual_qnn/utils/plot_fig4_penalty.py \
        papers/virtual_qnn/outdir/sweeps/e2_vqem_penalty_sweep --results papers/virtual_qnn/results
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    style.apply_style()
    summary = style.load_summary(args.sweep_dir)
    rows = sorted(summary["candidates"], key=lambda c: c["grid_point"]["--penalty"])
    alpha = np.array([c["grid_point"]["--penalty"] for c in rows], dtype=float)
    hs = np.array([c["aggregates"]["hs_squared"]["mean"] for c in rows])
    hs_sd = np.array([c["aggregates"]["hs_squared"]["std"] for c in rows])
    gamma = np.array([c["aggregates"]["overhead"]["mean"] for c in rows])
    gamma_sd = np.array([c["aggregates"]["overhead"]["std"] for c in rows])

    fig, (top, bottom) = style.plt.subplots(
        2, 1, figsize=(5.2, 5.6), sharex=True, height_ratios=[1, 1.3]
    )
    top.errorbar(
        alpha,
        hs,
        yerr=hs_sd,
        color=style.SERIES[0],
        marker="o",
        markersize=4,
        label="Reproduction (mean ± std, 3 seeds)",
    )
    top.plot(
        paper.FIG4_PENALTY,
        paper.FIG4_HS,
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        markeredgecolor=style.REFERENCE,
        markersize=11,
        markeredgewidth=1.3,
        zorder=5,
        label="Paper (reference data)",
    )
    top.set_xscale("log")
    top.set_yscale("log")
    top.set_ylabel("Squared HS distance")
    top.set_title("VQEM penalty sweep, amplitude damping q = 0.3 (Fig. 4)", loc="left")
    top.legend(fontsize=8)

    bottom.errorbar(
        alpha,
        gamma,
        yerr=gamma_sd,
        color=style.SERIES[0],
        marker="o",
        markersize=4,
        label="Reproduction",
    )
    bottom.plot(
        paper.FIG4_PENALTY,
        paper.FIG4_OVERHEAD,
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        markeredgecolor=style.REFERENCE,
        markersize=11,
        markeredgewidth=1.3,
        zorder=5,
        label="Paper (reference data)",
    )
    bottom.axhline(1.3 / 0.7, color=style.NEUTRAL, linestyle="--", linewidth=1.2)
    bottom.text(
        1.1,
        1.3 / 0.7 + 0.01,
        "SDP minimum 1.857 (E0)",
        fontsize=8,
        color=style.TEXT_SECONDARY,
        va="bottom",
    )
    bottom.set_xscale("log")
    bottom.set_xlabel("Penalty strength α")
    bottom.set_ylabel("Sampling overhead γ = ‖c‖₁")
    style.scope_note(
        bottom,
        "Scope: paper-accurate (reference hyperparameters), 3 seeds, exact simulation.",
    )
    paths = style.save(fig, "fig4_penalty_sweep", [args.sweep_dir, args.results])
    print("\n".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
