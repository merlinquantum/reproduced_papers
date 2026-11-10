"""Figures of the Phase 4 photonic adaptation (MerLin) and its Kerr ablation.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/plot_p4.py learning \
        papers/fermi_dirac_neurons/outdir/sweeps/p4_learning_v1
    python3 papers/fermi_dirac_neurons/utils/plot_p4.py trainability \
        papers/fermi_dirac_neurons/outdir/sweeps/p4_trainability_v1
    python3 papers/fermi_dirac_neurons/utils/plot_p4.py kerr \
        papers/fermi_dirac_neurons/outdir/sweeps/p4_kerr_v1

Each figure is saved to ``<sweep>/figures/`` and ``results/figures/`` with
the result-scope label as the legend title.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))

from fermi_dirac_neurons.lib.plotting import (  # noqa: E402
    COLOURS,
    GRID,
    SweepReader,
    apply_style,
    band,
    finish,
    save,
)

PHOTONIC_ROLES = {
    "photonic_tanh": ("quantum", "photonic neuron (MerLin mesh + PNR, tanh)"),
    "photon_count_tanh": ("commuting", "commuting photon-count neuron"),
    "correlation_tanh": ("b2", "classical neuron on <a†a> (B2 counterpart)"),
    "photonic_linear": ("linear", "photonic linear Tr[Hρ]"),
}
INPUT_COLOURS = {
    "fock_first": (COLOURS["quantum"], "Fock |1..1 0..0>"),
    "haar_interferometer": (COLOURS["b2"], "Haar interferometer × Fock"),
    "haar_full": (COLOURS["linear"], "Haar on full N-photon space"),
}
KERR_COLOURS = {0.0: COLOURS["quantum"], 0.5: COLOURS["b2"], 2.0: COLOURS["commuting"]}


def learning(sweep_dir: Path) -> None:
    """Training-loss curves of the four models, haar and fock arms, three sizes."""
    import matplotlib.pyplot as plt

    reader = SweepReader(sweep_dir)
    sizes = [(4, 2), (6, 3), (8, 4)]
    figure, axes = plt.subplots(
        2, 3, figsize=(11, 6.4), sharex=True, layout="constrained"
    )
    handles = {}
    for row, arm in enumerate(("haar", "fock")):
        for column, (m, n) in enumerate(sizes):
            axis = axes[row, column]
            config = f"p4_learn_m{m}_n{n}_{arm}"
            for model, (role, label) in PHOTONIC_ROLES.items():
                epochs, curves = reader.histories(config, model)
                band(axis, epochs, curves, role, label)
            axis.set_yscale("log")
            axis.set_title(
                f"N={n} photons, m={m} modes · "
                + ("interferometer data" if arm == "haar" else "Fock-basis data")
            )
            if column == 0:
                axis.set_ylabel("training squared loss")
            if row == 1:
                axis.set_xlabel("epoch")
            for line in axis.get_lines():
                handles.setdefault(line.get_label(), line)
    finish(
        figure,
        list(handles.values()),
        list(handles),
        "Phase 4 (MerLin adaptation): photonic Fermi–Dirac neuron vs baselines, paper optimizer (GD, η = 0.1)",
        "Simulation, exact PNR probabilities · 10 seeds, median and min–max band · photonic counterpart of Fig. 7, not the paper's data",
        columns=2,
    )
    save(figure, "p4_learning_curves", sweep_dir)


def _variance_panel(axis, points: dict, colour: str, label: str) -> None:
    dimensions = np.array(sorted(points))
    variances = np.array([points[d]["variance"] for d in dimensions])
    low = np.array([points[d]["variance_ci95"][0] for d in dimensions])
    high = np.array([points[d]["variance_ci95"][1] for d in dimensions])
    axis.fill_between(dimensions, low, high, color=colour, alpha=0.15, linewidth=0)
    axis.plot(dimensions, variances, "o-", color=colour, label=label, markersize=3)


def _reference(axis, dimensions: list[int], anchor: float) -> None:
    grid = np.array([min(dimensions), max(dimensions)], dtype=float)
    axis.plot(
        grid,
        anchor * grid[0] / grid,
        linestyle="--",
        color=GRID,
        linewidth=1.2,
        label="∝ 1/d (Haar rate)",
    )


def trainability(sweep_dir: Path) -> None:
    """Var(∂f) vs Fock dimension, half-filling family, T = 2, a = 0.5."""
    import matplotlib.pyplot as plt

    reader = SweepReader(sweep_dir)
    cells: dict = {}
    for config, _seed in reader.runs:
        metrics = reader.metrics(config)[0]
        if 2 * metrics["n_photons"] != metrics["n_modes"]:
            continue
        for record in metrics["records"]:
            if (
                record["quantity"] == "output"
                and record["temperature"] == 2.0
                and record["energy_range"] == 0.5
            ):
                cells.setdefault((record["parameter"], record["input"]), {})[
                    metrics["fock_dimension"]
                ] = record
    # The last mesh phase only sets an output phase (zero gradient), so the
    # third panel shows the commuting photon-count neuron as the reference.
    parameters = [
        ("eps0", "energy ε₁"),
        ("phase_first", "first mesh phase"),
        ("w0", "w₁ of the commuting photon-count neuron"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    handles = {}
    for axis, (parameter, title) in zip(axes, parameters):
        for kind, (colour, label) in INPUT_COLOURS.items():
            _variance_panel(axis, cells[(parameter, kind)], colour, label)
        dimensions = sorted(cells[(parameter, "haar_full")])
        _reference(
            axis, dimensions, cells[(parameter, "haar_full")][dimensions[0]]["variance"]
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Fock dimension d (m = 4, 6, 8, 10; N = m/2)")
        axis.set_title(f"∂f/∂{title}")
        for line in axis.get_lines():
            handles.setdefault(line.get_label(), line)
    axes[0].set_ylabel("Var over random parameters")
    finish(
        figure,
        list(handles.values()),
        list(handles),
        "Phase 4 E13: gradient variance of the photonic neuron (MerLin), T = 2, energies ~ U[−0.5, 0.5]",
        "Simulation, exact · seed 0, 200 parameter draws per point, 95 % bootstrap band · extension X2, not in the paper",
        columns=4,
    )
    save(figure, "p4_trainability", sweep_dir)


def kerr(sweep_dir: Path) -> None:
    """Var(∂f/∂θ_n0) vs Fock dimension for κ = 0, 0.5, 2, half-filling family."""
    import matplotlib.pyplot as plt

    reader = SweepReader(sweep_dir)
    cells: dict = {}
    for config, _seed in reader.runs:
        metrics = reader.metrics(config)[0]
        if 2 * metrics["n_photons"] != metrics["n_modes"]:
            continue
        for record in metrics["records"]:
            if record["parameter"] == "n0":
                cells.setdefault((record["input"], record["kerr"]), {})[
                    metrics["fock_dimension"]
                ] = record
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    handles = {}
    for axis, (kind, (_, title)) in zip(axes, INPUT_COLOURS.items()):
        for kappa, colour in KERR_COLOURS.items():
            _variance_panel(
                axis, cells[(kind, kappa)], colour, f"cross-Kerr κ = {kappa:g}"
            )
        dimensions = sorted(cells[(kind, 0.0)])
        _reference(axis, dimensions, cells[(kind, 0.0)][dimensions[0]]["variance"])
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Fock dimension d (m = 4, 6, 8, 10; N = m/2)")
        axis.set_title(f"input: {title}")
        for line in axis.get_lines():
            handles.setdefault(line.get_label(), line)
    axes[0].set_ylabel("Var(∂f/∂θ of n₁)")
    finish(
        figure,
        list(handles.values()),
        list(handles),
        "Phase 4 E13-K: passive H plus cross-Kerr chain, T = 2 (NOT MerLin: dense Fock-space numerics)",
        "Simulation, exact · seed 0, 200 parameter draws per point, 95 % bootstrap band · extension X2(c), not in the paper",
        columns=4,
    )
    save(figure, "p4_kerr", sweep_dir)


def main() -> None:
    """Draw the requested figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["learning", "trainability", "kerr"])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    apply_style()
    {"learning": learning, "trainability": trainability, "kerr": kerr}[args.part](
        args.sweep_dir
    )


if __name__ == "__main__":
    main()
