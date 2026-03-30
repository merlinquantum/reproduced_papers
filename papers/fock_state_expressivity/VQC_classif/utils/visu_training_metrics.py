from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lib.data import DATASET_ORDER  # noqa: E402
from utils.plotting import plot_training_metrics  # noqa: E402

from runtime_lib import run_from_project  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_dir(previous_run: str | None) -> Path:
    if previous_run:
        return Path(previous_run).expanduser().resolve()
    return run_from_project(PROJECT_DIR)


def _build_results(metrics: dict) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for dataset, data in metrics.items():
        runs = data.get("runs")
        if not runs:
            raise ValueError(f"Missing run histories for dataset '{dataset}'.")
        results[dataset] = {"runs": runs}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot VQC training metrics.")
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
    results = _build_results(metrics)

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "training_metrics.png"
    plot_training_metrics(results, output_path, dataset_order=DATASET_ORDER)
    if args.previous_run:
        print(f"Saved training metrics to {output_path.resolve()}")


if __name__ == "__main__":
    main()
