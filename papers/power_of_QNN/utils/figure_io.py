"""Shared helpers for the plotting scripts: paths, metrics loading, and paper references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for path in (REPO_ROOT, PROJECT_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

RESULTS_DIR = PROJECT_DIR / "results"
UPSTREAM = REPO_ROOT / "data" / "power_of_QNN" / "sources" / "effective_dimension"
PAPER_N = [
    1000,
    2000,
    8000,
    10000,
    40000,
    60000,
    100000,
    150000,
    200000,
    500000,
    1000000,
]
# Table 1 of arXiv:2011.00027v1: average final training loss over 100 trials.
PAPER_TABLE1_LOSS = {"qnn": 0.2314, "easy_qnn": 0.4305, "classical_41112": 0.3790}
PAPER_TABLE1_FISHER_RAO = {"qnn": 117.84, "easy_qnn": 104.89, "classical_41112": 46.45}


def parse_arguments(description: str) -> argparse.Namespace:
    """Parse ``--run-dir``, ``--scope`` and ``--name`` (the results/ file stem)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Run directory with metrics.json"
    )
    parser.add_argument(
        "--scope", required=True, help="Result-scope label, e.g. paper-accurate"
    )
    parser.add_argument("--name", required=True, help="File stem under results/")
    return parser.parse_args()


def load_metrics(run_dir: Path) -> dict:
    """Load ``metrics.json`` and require a COMPLETED run status."""
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] != "COMPLETED":
        raise ValueError(
            f"{run_dir} has status {status['status']}; plot only completed runs"
        )
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def outputs(run_dir: Path, name: str) -> tuple[Path, Path]:
    """Return the run-directory and results/ paths of a figure."""
    RESULTS_DIR.mkdir(exist_ok=True)
    return run_dir / f"{name}.png", RESULTS_DIR / f"{name}.png"


def paper_fig3a_curves() -> dict[str, tuple[list[float], list[float]]]:
    """Published Fig. 3a source data (rows: QNN, classical, easy), from the upstream cache."""
    rows = np.loadtxt(
        UPSTREAM
        / "Effective_dimension_plots/plot_fig/source_data/effective_dimension.txt"
    )
    return {
        "paper QNN": (PAPER_N, rows[0].tolist()),
        "paper classical (d=28 array)": (PAPER_N, rows[1].tolist()),
        "paper easy": (PAPER_N, rows[2].tolist()),
    }


def paper_fig6_points() -> tuple[list[float], list[float], list[float]]:
    """Authors' Fig. 6 statistic: first 10 cached runs per randomization level, d=880."""
    means, stds = [], []
    for level in range(5):
        values = []
        for run in range(10):
            entry = np.load(
                UPSTREAM / f"Generalisation_plots/data/data_s{run}_e{level}.npy",
                allow_pickle=True,
            )
            values.append(float(np.ravel(entry[1])[0]) / 880)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
    return [0.0, 0.1, 0.2, 0.3, 0.4], means, stds
