"""Shared plotting helpers: palette, sweep loading, saving to the sweep folder and results/.

Colors: the dataviz reference categorical palette (light mode) in fixed slot
order; test-to-test is a neutral reference, never a series color.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PAPER_DIR / "results"
SERIES = {
    "model": ("QCBM", "#2a78d6"),
    "b5": ("Distinguishable twin (B5)", "#eb6834"),
    "rbm": ("RBM", "#1baf7a"),
    "marginal": ("Marginal sampler (B4)", "#eda100"),
    "random": ("Uniform random", "#e87ba4"),
}
NEUTRAL = "#8a8a85"
TEXT = "#222222"
GRID = "#e6e6e3"


def style_axes(ax) -> None:
    """Recessive grid and spines, text in ink colors."""
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NEUTRAL)
    ax.tick_params(colors=TEXT, labelsize=8)


def load_sweep_runs(sweep_dir: Path) -> list[dict]:
    """Return one dict per COMPLETED run: candidate, seed, metrics, run_dir."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    runs = []
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        runs.append(
            {
                "candidate": run["candidate"],
                "seed": run["seed"],
                "run_dir": sweep_dir / run["run_dir"],
                "metrics": json.loads((sweep_dir / run["metrics_path"]).read_text()),
            }
        )
    return runs


def save(fig, sweep_dir: Path, name: str) -> list[Path]:
    """Save PNG + PDF to the sweep folder and to results/."""
    paths = []
    RESULTS_DIR.mkdir(exist_ok=True)
    for folder in (sweep_dir, RESULTS_DIR):
        for ext in ("png", "pdf"):
            path = folder / f"{name}.{ext}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            paths.append(path)
    plt.close(fig)
    return paths
