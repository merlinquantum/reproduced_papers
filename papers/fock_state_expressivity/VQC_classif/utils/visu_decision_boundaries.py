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

from utils.plotting import plot_decision_boundary_from_payload  # noqa: E402

from runtime_lib import run_from_project  # noqa: E402


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_dir(previous_run: str | None) -> Path:
    if previous_run:
        return Path(previous_run).expanduser().resolve()
    return run_from_project(PROJECT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot VQC decision boundaries from saved predictions."
    )
    parser.add_argument(
        "--previous-run",
        dest="previous_run",
        help="Path to a previous run directory to reuse boundary_data.json.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.previous_run)
    boundary_path = run_dir / "decision_boundaries" / "boundary_data.json"
    if not boundary_path.exists():
        raise FileNotFoundError(f"Missing boundary_data.json at {boundary_path}")

    payloads = _load_json(boundary_path)
    boundary_dir = run_dir / "figures" / "decision_boundaries"
    boundary_dir.mkdir(parents=True, exist_ok=True)

    for entry in payloads:
        dataset = entry.get("dataset", "dataset")
        model_name = entry.get("requested_model_type", entry.get("model_type", "model"))
        output_path = boundary_dir / f"{dataset}_{model_name}.png"
        plot_decision_boundary_from_payload(entry, output_path)

    if args.previous_run:
        print(f"Saved decision boundaries to {boundary_dir.resolve()}")


if __name__ == "__main__":
    main()
