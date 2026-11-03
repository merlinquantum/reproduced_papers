"""Run a predeclared sweep plan: ``python utils/run_sweep.py <plan.json> [--sweep-dir DIR]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for path in (REPO_ROOT, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib.sweep import load_plan, run_sweep  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="sweep plan JSON")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=None,
        help="output directory (default: outdir/sweeps/<sweep_id> in the paper folder)",
    )
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    sweep_dir = args.sweep_dir or PROJECT_DIR / "outdir" / "sweeps" / plan["sweep_id"]
    run_sweep(args.plan, sweep_dir, REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
