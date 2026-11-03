"""Shared helpers for the ``utils/plot_*.py`` scripts: artifact loading, scope labels, saving."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_completed_metrics(run_dir: Path, experiment: str) -> dict:
    """Return ``metrics.json`` of a COMPLETED run of the given experiment kind.

    Raises
    ------
    ValueError
        If the run did not complete or is of another experiment kind.
    """
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] != "COMPLETED":
        raise ValueError(f"{run_dir} is not COMPLETED: {status['status']}")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics["experiment"] != experiment:
        raise ValueError(
            f"{run_dir} is a {metrics['experiment']!r} run, expected {experiment!r}"
        )
    return metrics


def stamp_scope(figure: plt.Figure, scope: str, source: str) -> None:
    """Write the result-scope label and the source runs in the figure footer."""
    figure.text(
        0.01,
        0.003,
        f"[{scope}]  source: {source}",
        fontsize=7,
        color="0.35",
        ha="left",
        va="bottom",
    )


def save_to_run_and_results(
    figure: plt.Figure, run_dir: Path, filename: str
) -> list[Path]:
    """Save ``figure`` as ``filename`` in ``run_dir`` and in ``results/``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [run_dir / filename, RESULTS_DIR / filename]
    for path in paths:
        figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return paths
