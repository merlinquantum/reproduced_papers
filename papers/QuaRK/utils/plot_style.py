"""Shared figure style and readers for the QuaRK plotting scripts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

# Reference categorical palette, slots 1-4 in fixed order (validated for the light
# surface; slots 3-4 are below 3:1 contrast, so every line is also direct-labeled).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
TASK_COLORS = {
    "current_point_readout": "#2a78d6",
    "exp_fading": "#eb6834",
    "volterra": "#1baf7a",
    "one_step_forecast": "#eda100",
}
TASK_LABELS = {
    "current_point_readout": "Current-point readout (F1 as run)",
    "exp_fading": "Exponential fading (F2)",
    "volterra": "Volterra (F3)",
    "one_step_forecast": "One-step forecast (F1 as written)",
}
PAPER_TASKS = ("current_point_readout", "exp_fading", "volterra")
TASK_ORDER = (*PAPER_TASKS, "one_step_forecast")


def ordered_tasks(metrics: dict) -> list[str]:
    """Tasks of a run in the fixed display order (paper tasks first)."""
    return [task for task in TASK_ORDER if task in metrics["tasks"]]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "font.size": 10,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
        }
    )


def load_run(run_dir: Path) -> tuple[dict, dict]:
    """Return ``(metrics, arrays)`` of a run; arrays merge every ``*.npz`` artifact."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    arrays = {}
    for name in ("regularization_sweep", "training_size_sweep", "fit_examples"):
        path = run_dir / f"{name}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                arrays[name] = {k: data[k] for k in data.files}
    return metrics, arrays


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """The authors' ``train_size_sweep.pkl`` references only these two globals."""

    allowed = {("numpy._core.numeric", "_frombuffer"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.allowed:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_release(release_run: Path) -> dict:
    """Authors' run values: regularization sweep and training-size sweep (3 tasks)."""
    with np.load(release_run / "reg_sweep.npz", allow_pickle=False) as data:
        sweep = {k: data[k] for k in data.files}
    with (release_run / "artifacts" / "train_size_sweep.pkl").open("rb") as handle:
        sizes = _NumpyOnlyUnpickler(handle).load()
    ordered = sorted(sizes)
    # Fig. 4 uses one ridge per task at every N; check it rather than assume it.
    ridges = np.array([sizes[n]["lambda_regs"] for n in ordered])
    if not np.all(ridges == ridges[0]):
        raise ValueError("the release's Fig. 4 ridge changes with N")
    return {
        "reg_grid": sweep["reg_grid"],
        "mse_train": sweep["mse_train"],
        "sizes": np.array(ordered),
        "size_mse_test": np.array(
            [sizes[n]["mse_test"] for n in ordered]
        ).T,  # (3, sizes)
        "lambda_regs": [float(r) for r in ridges[0]],
    }


def load_sweep(sweep_dir: Path) -> list[dict]:
    """Every run of a completed sweep: ``{candidate, seed, run_dir, metrics}``."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    runs = []
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            raise SystemExit(f"{sweep_dir}: run {run['run_id']} is {run['status']}")
        run_dir = (sweep_dir / run["metrics_path"]).parent
        runs.append(
            {
                "candidate": run["candidate"]["name"],
                "seed": int(run["seed"]),
                "run_dir": run_dir,
                "metrics": json.loads((run_dir / "metrics.json").read_text()),
            }
        )
    return runs


def quark_runs(*sweeps: list[dict]) -> dict:
    """QuaRK runs of several sweeps keyed by their recorded settings.

    Returns ``{(estimator, shots, n, R): {seed: metrics}}``; shots is None for exact
    features.
    """
    index = {}
    for runs in sweeps:
        for run in runs:
            metrics = run["metrics"]
            if metrics["model"]["kind"] != "quark":
                continue
            quark = metrics["model"]["quark"]
            key = (
                quark["estimator"],
                None if quark["estimator"] == "exact" else int(quark["shots"]),
                int(quark["num_qubits"]),
                int(quark["num_reservoirs"]),
            )
            if run["seed"] in index.setdefault(key, {}):
                raise SystemExit(f"duplicate QuaRK run {key}, seed {run['seed']}")
            index[key][run["seed"]] = metrics
    return index


def selected(metrics: dict, task: str, key: str):
    """A metric at the ridge chosen on validation data."""
    return metrics["tasks"][task]["selections"]["validation"][key]


def spread_log_labels(positions: dict, min_ratio: float = 1.13) -> dict:
    """Push end-label heights apart on a log axis so that neighbours differ by ``min_ratio``."""
    out, previous = {}, None
    for key, y in sorted(positions.items(), key=lambda item: item[1]):
        y = y if previous is None else max(y, previous * min_ratio)
        out[key], previous = y, y
    return out


def save(fig, run_dir: Path, name: str) -> list[Path]:
    """Save a figure as PNG and PDF in ``<run>/figures`` and in ``results/``."""
    paths = []
    for folder in (run_dir / "figures", RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            path = folder / f"{name}.{suffix}"
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
            paths.append(path)
    plt.close(fig)
    return paths
