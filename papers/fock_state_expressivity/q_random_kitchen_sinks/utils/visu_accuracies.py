from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.visualization import plot_accuracy_heatmap  # noqa: E402

from runtime_lib import run_from_project  # noqa: E402


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_dir(previous_run: str | None) -> Path:
    if previous_run:
        return Path(previous_run).expanduser().resolve()
    return run_from_project(PROJECT_DIR)


def _load_sweep_values(run_dir: Path) -> tuple[list[int], list[int]]:
    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_snapshot.json at {config_path}")
    config = _load_json(config_path)
    sweep = config.get("sweep", {})
    r_values = sweep.get("r_values", [1])
    gamma_values = sweep.get("gamma_values", list(range(1, 11)))
    return r_values, gamma_values


def _plot_accuracy_bar(metrics: list[dict], save_path: Path) -> None:
    lookup = {"quantum": [], "classical": []}
    for entry in metrics:
        method = entry.get("method")
        if method in lookup:
            lookup[method].append(entry.get("accuracy", 0.0))
    if not lookup["quantum"] or not lookup["classical"]:
        raise ValueError("Missing accuracy entries for quantum/classical methods.")
    q_avg = sum(lookup["quantum"]) / len(lookup["quantum"])
    c_avg = sum(lookup["classical"]) / len(lookup["classical"])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Quantum", "Classical"], [q_avg, c_avg], color=["#1f77b4", "#d62728"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison")
    for idx, value in enumerate([q_avg, c_avg]):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot accuracy summaries.")
    parser.add_argument(
        "--previous-run",
        dest="previous_run",
        help="Path to a previous run directory to reuse metrics.json.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.previous_run)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json at {metrics_path}")
    metrics = _load_json(metrics_path)
    r_values, gamma_values = _load_sweep_values(run_dir)

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if len(r_values) * len(gamma_values) == 1:
        output_path = figures_dir / "accuracy_bar.png"
        _plot_accuracy_bar(metrics, output_path)
        if args.previous_run:
            print(f"Saved accuracy bar to {output_path.resolve()}")
        return

    r_values = list(reversed(r_values))
    output_quantum = figures_dir / "accuracy_quantum.png"
    output_classical = figures_dir / "accuracy_classical.png"
    plot_accuracy_heatmap(metrics, r_values, gamma_values, "quantum", output_quantum)
    plot_accuracy_heatmap(
        metrics, r_values, gamma_values, "classical", output_classical
    )

    if args.previous_run:
        print(f"Saved accuracy heatmaps to {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
