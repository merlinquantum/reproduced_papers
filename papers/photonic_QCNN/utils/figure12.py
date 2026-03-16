"""Run the Figure 12 simulation pipeline and render figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent.parent
PAPERS_ROOT = REPO_ROOT / "papers"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PAPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPERS_ROOT))

from photonic_QCNN.lib.runner import load_config, run_simulation_pipeline  # noqa: E402


def _stringify(obj):
    if isinstance(obj, dict):
        return {k: _stringify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Figure 12 simulation pipeline."
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Base directory for outputs (defaults to results/figure12).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = PROJECT_ROOT / "configs" / "merlin.json"
    config = load_config(config_path)
    summary = run_simulation_pipeline(args.outdir, config=config)
    base_dir = Path(summary["base_dir"])
    summary_path = base_dir / "summary.json"
    summary_path.write_text(json.dumps(_stringify(summary), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
