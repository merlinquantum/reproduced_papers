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

from utils.plotting import plot_training_curves  # noqa: E402

from runtime_lib import run_from_project  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_dir(previous_run: str | None) -> Path:
    if previous_run:
        return Path(previous_run).expanduser().resolve()
    return run_from_project(PROJECT_DIR)


def _load_config_colors(run_dir: Path) -> dict[str, str]:
    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        return {}
    config = _load_json(config_path)
    colors = config.get("plotting", {}).get("colors", [])
    initial_states = config.get("training", {}).get("initial_states", [])
    if not colors or not initial_states:
        return {}
    color_map: dict[str, str] = {}
    for idx, state in enumerate(initial_states):
        label = f"VQC_{state}"
        color_map[label] = colors[idx % len(colors)]
    return color_map


def _build_results(metrics: dict, color_map: dict[str, str]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for label, data in metrics.items():
        runs = data.get("loss_curves", [])
        entry = {"runs": runs}
        if label in color_map:
            entry["color"] = color_map[label]
        results[label] = entry
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot VQC training curves.")
    parser.add_argument(
        "--previous-run",
        dest="previous_run",
        help="Path to a previous run directory to reuse metrics.json.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.previous_run)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json at {metrics_path}")

    metrics = _load_json(metrics_path)
    color_map = _load_config_colors(run_dir)
    results = _build_results(metrics, color_map)

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "training_curves.png"
    plot_training_curves(results, save_path=output_path, show=args.show)
    if args.previous_run:
        print(f"Saved training curves to {output_path.resolve()}")


if __name__ == "__main__":
    main()
