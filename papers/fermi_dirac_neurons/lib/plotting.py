"""Shared helpers for the reproduction figures (``utils/plot_*.py``).

Sweep runs are read only through ``sweep_status.json`` (COMPLETED runs) and
their ``history.csv`` / ``metrics.json`` artifacts. Every figure is saved to
``<sweep_dir>/figures/`` and to the paper's ``results/figures/``.

Colours follow the entity, never the panel: each model family keeps one
categorical slot in every figure (fixed order of the validated reference
palette of the dataviz guidance). Line style encodes the source: solid for
our 10-seed median (band: min-max over seeds), dashed for the paper.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_FIGURES = PROJECT_DIR / "results" / "figures"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Categorical slots 1-4 in fixed order: quantum neuron, commuting classical
# neuron, fair classical baseline B2, quantum linear model.
COLOURS = {
    "quantum": "#2a78d6",
    "commuting": "#eb6834",
    "b2": "#1baf7a",
    "linear": "#eda100",
}
ROLE_LABELS = {
    "quantum": "quantum neuron",
    "commuting": "commuting (classical) neuron",
    "b2": "B2: classical neuron on same Pauli features (paper η, untuned)",
    "linear": "quantum linear Tr[H(θ)ρ]",
}


def model_role(model: str) -> str:
    """Return the colour role of a model name used in the configs."""
    if model.startswith("b2_"):
        return "b2"
    if model.startswith("b3_") or model.endswith("_linear"):
        return "linear"
    if model.startswith("classical"):
        return "commuting"
    if model.startswith("quantum"):
        return "quantum"
    raise ValueError(f"No colour role for model {model!r}")


def apply_style() -> None:
    """Set recessive axes and grid, thin lines and the chart surface."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "font.size": 9,
        }
    )


class SweepReader:
    """Read per-seed artifacts of a COMPLETED sweep.

    Raises
    ------
    ValueError
        If the sweep is not COMPLETED or a requested run is missing.
    """

    def __init__(self, sweep_dir: Path) -> None:
        self.sweep_dir = Path(sweep_dir)
        status = json.loads((self.sweep_dir / "sweep_status.json").read_text())
        if status["status"] != "COMPLETED":
            raise ValueError(
                f"Sweep {self.sweep_dir} is {status['status']}, not COMPLETED"
            )
        self.sweep_id = status["sweep_id"]
        self.seeds = sorted(set(status["spec"]["seeds"]))
        self.runs = {(run["config"], run["seed"]): run for run in status["runs"]}

    def _run_dir(self, config: str, seed: int) -> Path:
        run = self.runs.get((config, seed))
        if run is None or run["status"] != "COMPLETED":
            raise ValueError(f"No COMPLETED run for {config} seed {seed}")
        return self.sweep_dir / run["run_dir"]

    def histories(self, config: str, model: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(epochs, losses[seed, epoch])`` of one model's training loss."""
        curves, epochs = [], None
        for seed in self.seeds:
            with (self._run_dir(config, seed) / "history.csv").open(
                encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            epochs = np.array([int(row["epoch"]) for row in rows])
            curves.append([float(row[f"{model}_train_loss"]) for row in rows])
        return epochs, np.array(curves)

    def metrics(self, config: str) -> list[dict]:
        """Return the ``metrics.json`` of every seed, in seed order."""
        return [
            json.loads((self._run_dir(config, seed) / "metrics.json").read_text())
            for seed in self.seeds
        ]


def band(axis, epochs: np.ndarray, curves: np.ndarray, role: str, label: str) -> None:
    """Draw the seed median (solid) and the min-max band of one model."""
    colour = COLOURS[role]
    axis.fill_between(
        epochs, curves.min(0), curves.max(0), color=colour, alpha=0.15, linewidth=0
    )
    axis.plot(epochs, np.median(curves, 0), color=colour, label=label)


def finish(
    figure, handles: list, labels: list, title: str, scope: str, columns: int = 3
) -> None:
    """Add title, shared legend and scope label in an order constrained layout can stack."""
    figure.suptitle(title)
    # The scope label is the legend's title: one layout-managed box at the
    # bottom, so neither can overlap the suptitle or the axes.
    legend = figure.legend(
        handles, labels, loc="outside lower center", ncol=columns, title=scope
    )
    legend.get_title().set_fontsize(7)
    legend.get_title().set_color(TEXT_SECONDARY)


def save(figure, name: str, sweep_dir: Path) -> list[Path]:
    """Save a figure as PNG to the sweep's ``figures/`` and to ``results/figures/``."""
    paths = []
    for directory in (Path(sweep_dir) / "figures", RESULTS_FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        paths.append(path)
    return paths
