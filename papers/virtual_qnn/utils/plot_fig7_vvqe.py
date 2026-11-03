"""Fig. 7 reproduction: HeH+ energies from VVQE vs noisy VQE, with FCI and HF references.

Usage::

    python3 papers/virtual_qnn/utils/plot_fig7_vvqe.py \
        papers/virtual_qnn/outdir/sweeps/e5_vvqe \
        papers/virtual_qnn/outdir/sweeps/e5_vqe_noiseless --results papers/virtual_qnn/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR.parent))

from virtual_qnn.lib import data  # noqa: E402
from virtual_qnn.lib import paper_values as paper  # noqa: E402
from virtual_qnn.utils import plot_style as style  # noqa: E402

BUDGETS = (1.0, 1.5, 2.0, 2.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("vvqe_sweep_dir", type=Path)
    parser.add_argument("noiseless_sweep_dir", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    style.apply_style()
    summary = style.load_summary(args.vvqe_sweep_dir)
    noiseless = style.load_summary(args.noiseless_sweep_dir)
    grid = np.linspace(0.3, 2.5, 89)
    references = [data.heh_tapered_hamiltonian(float(round(d, 4))) for d in grid]

    fig, ax = style.plt.subplots(figsize=(6.6, 4.4))
    ax.plot(
        grid,
        [r["fci_energy"] for r in references],
        color=style.REFERENCE,
        linestyle="--",
        linewidth=1.3,
        label="Exact (FCI, tapered STO-3G)",
    )
    ax.plot(
        grid,
        [r["hf_energy"] for r in references],
        color=style.NEUTRAL,
        linestyle=":",
        linewidth=1.6,
        label="Hartree–Fock (the paper's 'error-free' curve)",
    )
    nl = sorted(
        noiseless["candidates"], key=lambda c: c["grid_point"]["--half-distance"]
    )
    ax.plot(
        [c["grid_point"]["--half-distance"] for c in nl],
        [c["aggregates"]["energy"]["mean"] for c in nl],
        color=style.NEUTRAL,
        linewidth=1.2,
        marker="x",
        markersize=5,
        label="Noiseless VQE, same ansatz (ceiling)",
    )
    for index, budget in enumerate(BUDGETS):
        rows = sorted(
            (
                c
                for c in summary["candidates"]
                if c["grid_point"]["--vvqe-budget"] == budget
            ),
            key=lambda c: c["grid_point"]["--half-distance"],
        )
        d = np.array([c["grid_point"]["--half-distance"] for c in rows])
        mean = np.array([c["aggregates"]["energy"]["mean"] for c in rows])
        sd = np.array([c["aggregates"]["energy"]["std"] for c in rows])
        color, marker = style.SERIES[index], style.MARKERS[index]
        name = "VQE (γ = 1)" if budget == 1.0 else f"VVQE γ = {budget:g}"
        ax.fill_between(d, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)
        ax.plot(d, mean, color=color, marker=marker, label=f"{name}: reproduction")
        ax.plot(
            paper.FIG7_HALF_DISTANCE,
            paper.FIG7_ENERGY[budget],
            linestyle="none",
            marker=marker,
            markerfacecolor="none",
            markeredgecolor=color,
            markersize=9,
            markeredgewidth=1.4,
        )
    ax.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        markeredgecolor=style.TEXT_SECONDARY,
        markersize=9,
        label="Hollow markers: paper (reference data)",
    )
    ax.set_xlabel(
        "Reference 'distance' d (bohr; nuclear separation = 2d bohr ≈ 1.058·d Å)"
    )
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("HeH⁺ ground-state energy with noisy circuits (Fig. 7)", loc="left")
    ax.legend(fontsize=7.5, loc="upper right", ncol=1)
    style.scope_note(
        ax,
        "Scope: paper-accurate (reference hyperparameters), 3 seeds, exact simulation; band = ±1 std.",
    )
    paths = style.save(
        fig,
        "fig7_heh_energy",
        [args.vvqe_sweep_dir, args.noiseless_sweep_dir, args.results],
    )
    print("\n".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
