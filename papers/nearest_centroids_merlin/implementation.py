#!/usr/bin/env python
"""
Local convenience CLI for the nearest_centroids reproduction.

This can be run standalone:
    python implementation.py --config configs/example.json

Or via the parent MerLin runner:
    python ../implementation.py --project nearest_centroids_merlin --config configs/example.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _here() -> Path:
    """Return the project directory (folder containing this file)."""
    return Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nearest Centroids paper reproduction (local runner)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/example.json",
        help=(
            "Path to JSON config, relative to this folder. "
            "Defaults to configs/example.json"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from config",
    )
    return parser


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON config file into a dict."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    # Import here to avoid circular imports when used as a module
    from lib.runner import main as run_experiment

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    project_root = _here()

    # Resolve config path relative to this file if not absolute
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    cfg = load_config(cfg_path)

    # Override seed if provided via CLI
    if args.seed is not None:
        cfg["seed"] = args.seed

    # Delegate to the MerLin-style runner
    run_experiment(cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
