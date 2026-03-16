from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for path in (PROJECT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib.runner import run_train_eval  # noqa: E402

from runtime_lib.config import deep_update, load_config  # noqa: E402
from runtime_lib.dtypes import resolve_config_dtypes  # noqa: E402
from runtime_lib.seed import seed_everything  # noqa: E402
from utils.utils import plot_figure_5  # noqa: E402

DEFAULTS_PATH = PROJECT_DIR / "configs" / "train_eval_circles.json"


def _load_cfg(config_path: Path | None) -> dict:
    cfg = load_config(DEFAULTS_PATH)
    if config_path is not None:
        cfg = deep_update(cfg, load_config(config_path))
    resolve_config_dtypes(cfg)
    return cfg


def _create_run_dir(cfg: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(cfg.get("outdir", "results"))
    run_dir = outdir / f"figure_5_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_results_path(raw: Path, default_name: str) -> Path:
    path = raw
    if path.is_dir():
        path = path / default_name
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 5 from train/eval results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config file to control training before plotting.",
    )
    parser.add_argument(
        "--previous_run",
        type=Path,
        default=None,
        help="Path to train_eval_results.json (or run directory) to plot without re-running.",
    )
    args = parser.parse_args()

    if args.config is not None and args.previous_run is not None:
        parser.error(
            "Use either --config or --previous_run, not both. "
            "Re-run or reuse a prior run so inputs stay consistent."
        )

    if args.previous_run is not None:
        results_path = _resolve_results_path(
            args.previous_run, "train_eval_results.json"
        )
        results = json.loads(results_path.read_text(encoding="utf-8"))
        run_dir = results_path.parent
    else:
        cfg = _load_cfg(args.config)
        seed_everything(cfg.get("seed"))
        run_dir = _create_run_dir(cfg)
        run_train_eval(cfg, run_dir)
        results_path = run_dir / "train_eval_results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))

    dataset_name = results["dataset"]
    train_acc = results["train_accuracies"]
    test_acc = results["test_accuracies"]
    range_layers = results["range_num_layers"]

    plot_figure_5(train_acc, test_acc, range_layers)
    plot_path = run_dir / f"figure_5_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved Figure 5 to: {plot_path}")


if __name__ == "__main__":
    main()
