"""Run several configs in sequence, writing one run directory per config.

This mirrors what the shared runtime does for a single config
(``python implementation.py --paper bosonic_binary_solver --config ...``) and
exists only so that a long queue of experiments can be launched unattended.

    python utils/run_configs.py configs/knapsack_m10_original.json ... [--tag q1]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.runner import train_and_evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--outdir", default=str(ROOT / "outdir"))
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--instances", type=int, default=None, help="override problem.instances"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    for config_path in args.configs:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if args.instances is not None:
            cfg["problem"]["instances"] = args.instances
        name = Path(config_path).stem
        # one stable directory per config, so an interrupted queue resumes into
        # the same place instead of starting a fresh partial run beside it
        existing = sorted(Path(args.outdir).glob(f"run_*_{args.tag}{name}"))
        if existing:
            run_dir = existing[-1]
        else:
            run_dir = (
                Path(args.outdir)
                / f"run_{time.strftime('%Y%m%d-%H%M%S')}_{args.tag}{name}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_snapshot.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8"
        )
        logging.info("=== %s -> %s", config_path, run_dir)
        train_and_evaluate(cfg, run_dir)


if __name__ == "__main__":
    main()
