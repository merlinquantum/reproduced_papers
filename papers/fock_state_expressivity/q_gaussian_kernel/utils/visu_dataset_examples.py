from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROJECT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lib.data import prepare_classification_data  # noqa: E402
from utils.plotting import plot_dataset_examples_from_payload  # noqa: E402

from runtime_lib.config import load_config  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_defaults_config() -> dict[str, Any]:
    defaults_path = PROJECT_DIR / "configs" / "defaults.json"
    return load_config(defaults_path)


def _serialize_dataset_payload(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "x_train": splits["x_train"].tolist(),
            "y_train": splits["y_train"].tolist(),
            "x_test": splits["x_test"].tolist(),
            "y_test": splits["y_test"].tolist(),
        }
        for name, splits in datasets.items()
    }


def _resolve_default_output_dir(cfg: dict[str, Any]) -> Path:
    outdir = Path(cfg.get("outdir", "results"))
    if not outdir.is_absolute():
        outdir = PROJECT_DIR / outdir
    return outdir / "figures"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot classification datasets.")
    parser.add_argument(
        "--previous-run",
        dest="previous_run",
        help="Path to a previous run directory to reuse classification_datasets.json.",
    )
    args = parser.parse_args()

    if args.previous_run:
        run_dir = Path(args.previous_run).expanduser().resolve()
        payload_path = (
            run_dir / "classify" / "visualization_data" / "classification_datasets.json"
        )
        if not payload_path.exists():
            warnings.warn(
                "Missing classification_datasets.json; the provided previous run has to be a `classify` task run.",
                stacklevel=2,
            )
            raise FileNotFoundError(
                f"Missing classification_datasets.json at {payload_path}. The provided previous run has to be a `classify` task run."
            )
        payload = _load_json(payload_path)
        figures_dir = run_dir / "classify" / "figures" / "datasets"
    else:
        defaults = _load_defaults_config()
        datasets = prepare_classification_data(
            defaults.get("classification", {}).get("data", {})
        )
        payload = _serialize_dataset_payload(datasets)
        figures_dir = _resolve_default_output_dir(defaults)
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "classification_datasets.png"
    plot_dataset_examples_from_payload(payload, output_path)
    print(f"Saved dataset plots to {output_path.resolve()}")


if __name__ == "__main__":
    main()
