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

from utils.visualization import (  # noqa: E402
    plot_combined_decisions_from_payload,
    plot_single_decision_from_payload,
)

from runtime_lib import run_from_project  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_dir(previous_run: str | None) -> Path:
    if previous_run:
        return Path(previous_run).expanduser().resolve()
    return run_from_project(PROJECT_DIR)


def _load_payloads(run_dir: Path) -> tuple[dict, dict]:
    visualization_dir = run_dir / "visualization_data"
    quantum_path = visualization_dir / "combined_decisions_quantum.json"
    classical_path = visualization_dir / "combined_decisions_classical.json"
    if not quantum_path.exists() or not classical_path.exists():
        raise FileNotFoundError(
            "Missing decision payloads; ensure the run completed successfully."
        )
    return _load_json(quantum_path), _load_json(classical_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot decision boundaries.")
    parser.add_argument(
        "--previous-run",
        dest="previous_run",
        help="Path to a previous run directory to reuse decision payloads.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.previous_run)
    quantum_payload, classical_payload = _load_payloads(run_dir)

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    is_single = (
        len(quantum_payload.get("r_values", []))
        * len(quantum_payload.get("gamma_values", []))
        == 1
    )
    if is_single:
        output_quantum = figures_dir / "decision_boundary_quantum.png"
        output_classical = figures_dir / "decision_boundary_classical.png"
        plot_single_decision_from_payload(quantum_payload, output_quantum)
        plot_single_decision_from_payload(classical_payload, output_classical)
        if args.previous_run:
            print(f"Saved decision boundaries to {figures_dir.resolve()}")
        return

    output_quantum = figures_dir / "combined_decisions_quantum.png"
    output_classical = figures_dir / "combined_decisions_classical.png"
    plot_combined_decisions_from_payload(quantum_payload, output_quantum)
    plot_combined_decisions_from_payload(classical_payload, output_classical)

    if args.previous_run:
        print(f"Saved decision boundaries to {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
