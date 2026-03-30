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

from utils.visualization import save_kernel_heatmap_from_payload  # noqa: E402

from runtime_lib import run_from_project  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_dir(previous_run: str | None) -> Path:
    if previous_run:
        return Path(previous_run).expanduser().resolve()
    return run_from_project(PROJECT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot kernel heatmaps.")
    parser.add_argument(
        "--previous-run",
        dest="previous_run",
        help="Path to a previous run directory to reuse kernel payloads.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.previous_run)
    visualization_dir = run_dir / "visualization_data"
    quantum_path = visualization_dir / "kernel_quantum.json"
    classical_path = visualization_dir / "kernel_classical.json"
    if not quantum_path.exists() or not classical_path.exists():
        quantum_path = visualization_dir / "kernel_quantum_best.json"
        classical_path = visualization_dir / "kernel_classical_best.json"
    if not quantum_path.exists() or not classical_path.exists():
        raise FileNotFoundError(
            "Missing kernel payloads; run must produce kernel JSON artifacts."
        )

    quantum_payload = _load_json(quantum_path)
    classical_payload = _load_json(classical_path)

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_quantum = figures_dir / "kernel_quantum.png"
    output_classical = figures_dir / "kernel_classical.png"
    save_kernel_heatmap_from_payload(quantum_payload, output_quantum)
    save_kernel_heatmap_from_payload(classical_payload, output_classical)

    if args.previous_run:
        print(f"Saved kernel heatmaps to {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
