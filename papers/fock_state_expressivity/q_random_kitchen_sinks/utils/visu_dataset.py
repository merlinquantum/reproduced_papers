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

from utils.visualization import plot_dataset_from_payload  # noqa: E402

from runtime_lib import run_from_project  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_dir(previous_run: str | None) -> Path:
    if previous_run:
        return Path(previous_run).expanduser().resolve()
    return run_from_project(PROJECT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the moons dataset.")
    parser.add_argument(
        "--previous-run",
        dest="previous_run",
        help="Path to a previous run directory to reuse dataset.json.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.previous_run)
    dataset_path = run_dir / "visualization_data" / "dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset.json at {dataset_path}")

    payload = _load_json(dataset_path)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "moons_dataset.png"
    plot_dataset_from_payload(payload, output_path)
    if args.previous_run:
        print(f"Saved dataset plot to {output_path.resolve()}")


if __name__ == "__main__":
    main()
