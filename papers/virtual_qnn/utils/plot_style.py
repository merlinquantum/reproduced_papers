"""Shared matplotlib styling and I/O for the reproduction figures.

Colors are the first slots of the validated reference categorical palette
(light mode), assigned in fixed order; every series also has its own marker so
identity never relies on color alone. Paper values are always drawn as hollow
markers, reproduction values as filled markers with a +/- 1 std band.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
REFERENCE = "#0b0b0b"
NEUTRAL = "#8a8984"


def apply_style() -> None:
    """Set recessive axes/grid and readable type for static figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def load_summary(sweep_dir: Path) -> dict:
    """Load a sweep summary; refuse partial sweeps for paper-accurate figures."""
    summary = json.loads((Path(sweep_dir) / "summary.json").read_text())
    if summary["coverage"] != "COMPLETE SWEEP":
        raise ValueError(f"{sweep_dir} is {summary['coverage']}")
    return summary


def save(fig, name: str, targets: list[Path]) -> list[Path]:
    """Save ``fig`` as ``<name>.png`` into every target directory."""
    paths = []
    for directory in targets:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.png"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def scope_note(ax, text: str) -> None:
    """Write the result-scope label (validity policy) under the axes."""
    ax.text(
        0.0,
        -0.22,
        text,
        transform=ax.transAxes,
        fontsize=8,
        color=TEXT_SECONDARY,
        va="top",
    )
