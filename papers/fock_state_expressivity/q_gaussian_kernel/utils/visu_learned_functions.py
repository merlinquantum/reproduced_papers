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

from utils.plotting import plot_gaussian_fits_from_payload  # noqa: E402

from runtime_lib import run_from_project  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_dir(previous_run: str | None) -> Path:
    if previous_run:
        return Path(previous_run).expanduser().resolve()
    return run_from_project(PROJECT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot learned quantum kernels from saved predictions."
    )
    parser.add_argument(
        "--previous-run",
        dest="previous_run",
        help="Path to a previous run directory to reuse learned_functions.json.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.previous_run)
    payload_path = run_dir / "sampler" / "visualization_data" / "learned_functions.json"
    if not payload_path.exists():
        warnings.warn(
            "Missing learned_functions.json; the provided previous run has to be a `sampler` task run.",
            stacklevel=2,
        )
        raise FileNotFoundError(
            f"Missing learned_functions.json at {payload_path}. The provided previous run has to be a `sampler` task run."
        )

    payload = _load_json(payload_path)
    figures_dir = run_dir / "sampler" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "learned_vs_target.png"
    plot_gaussian_fits_from_payload(payload, output_path)
    if args.previous_run:
        print(f"Saved learned functions to {output_path.resolve()}")


if __name__ == "__main__":
    main()
