from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROJECT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.plotting import plot_accuracy_bars  # noqa: E402

from runtime_lib import run_from_project  # noqa: E402


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_dir(previous_run: str | None) -> Path:
    if previous_run:
        return Path(previous_run).expanduser().resolve()
    return run_from_project(PROJECT_DIR, ["--task", "classify"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot accuracy bar charts.")
    parser.add_argument(
        "--previous-run",
        dest="previous_run",
        help="Path to a previous run directory to reuse accuracy metrics.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.previous_run)
    quantum_path = run_dir / "classify" / "quantum_metrics.json"
    classical_path = run_dir / "classify" / "classical_metrics.json"
    if not quantum_path.exists() or not classical_path.exists():
        warnings.warn(
            "Missing accuracy metrics; the provided previous run has to be a `classify` task run.",
            stacklevel=2,
        )
        raise FileNotFoundError(
            "Missing quantum_metrics.json or classical_metrics.json in run directory. The provided previous run has to be a `classify` task run."
        )

    quantum_metrics = _load_json(quantum_path)
    classical_metrics = _load_json(classical_path)

    figures_dir = run_dir / "classify" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "svm_accuracy.png"
    plot_accuracy_bars(quantum_metrics, classical_metrics, output_path)
    if args.previous_run:
        print(f"Saved accuracy bars to {output_path.resolve()}")


if __name__ == "__main__":
    main()
