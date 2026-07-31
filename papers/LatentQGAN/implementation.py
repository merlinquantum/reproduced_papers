"""CLI entry point for the LatentQGAN reproduction.

Usage::

    python implementation.py --config configs/defaults.json --seed 42

Output goes to ``outdir/run_<timestamp>_seed<N>``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lib.runner import train_and_evaluate


def _parse_overrides(overrides: list[str]) -> dict:
    out: dict = {}
    for tok in overrides:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/defaults.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="outdir")
    ap.add_argument("--paper", default="LatentQGAN")
    args, remaining = ap.parse_known_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent / cfg_path
    cfg = json.loads(cfg_path.read_text())
    cfg["seed"] = args.seed
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    cfg.update(_parse_overrides(remaining))

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(__file__).parent
        / args.outdir
        / f"run_{ts}_seed{args.seed}_{cfg.get('model', 'na')}_d{cfg.get('digit', 0)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out = train_and_evaluate(cfg, run_dir)
    print(json.dumps(out["test_metrics"], indent=2))
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
