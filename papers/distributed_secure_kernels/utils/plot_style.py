"""Shared matplotlib style for the reproduction figures (validated light palette)."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OURS = "#2a78d6"  # blue: this reproduction
PAPER = "#eb6834"  # orange: paper / authors' logs
REFERENCE = "#6b6b66"  # gray: classical and majority-class reference lines
INK, MUTED, GRID = "#1f1f1d", "#6b6b66", "#e4e3de"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def style_axes(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def save(fig, name: str, run_dir: Path) -> list[Path]:
    """Save ``name``.png in ``run_dir`` and copy it to ``results/``."""

    fig.patch.set_facecolor("#fcfcfb")
    target = Path(run_dir) / f"{name}.png"
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    RESULTS_DIR.mkdir(exist_ok=True)
    copy = RESULTS_DIR / target.name
    shutil.copyfile(target, copy)
    return [target, copy]
