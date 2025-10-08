#!/usr/bin/env python3
"""
Generic CLI entry-point for paper reproduction experiments.
- Loads configuration from JSON via --config
- Allows CLI overrides for common hyperparameters
- Sets up logging, output directory, and config snapshot
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
from pathlib import Path

from lib.config import deep_update, default_config, load_config

# -----------------------------
# Core placeholders
# -----------------------------


def setup_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    # Extend with torch / jax determinism if relevant


def train_and_evaluate(cfg, run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting training")
    logger.debug("Resolved config: %s", json.dumps(cfg, indent=2))
    # TODO: Implement dataset/model/training loop specific to the paper
    # Save a minimal artifact example
    (run_dir / "done.txt").write_text(
        "Training placeholder complete. Replace with actual implementation.\n"
    )
    logger.info("Wrote placeholder artifact: %s", run_dir / "done.txt")


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper reproduction runner")
    p.add_argument("--config", type=str, help="Path to JSON config", default=None)
    p.add_argument("--seed", type=int, help="Random seed", default=None)
    p.add_argument("--outdir", type=str, help="Base output directory", default=None)
    p.add_argument(
        "--device", type=str, help="Device string (cpu, cuda:0, mps)", default=None
    )

    # Common training overrides
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)

    return p


def configure_logging(level: str = "info", log_file: Path | None = None) -> None:
    """Configure root logger with stream handler and optional file handler.

    Example usage:
        configure_logging("debug")
        logger = logging.getLogger(__name__)
        logger.info("Message")
    """
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    log_level = level_map.get(str(level).lower(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)
    # Reset handlers to avoid duplicates on reconfiguration
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def resolve_config(args: argparse.Namespace):
    cfg = default_config()

    # Load from file if provided
    if args.config:
        file_cfg = load_config(Path(args.config))
        cfg = deep_update(cfg, file_cfg)

    # Apply CLI overrides
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.device is not None:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["dataset"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr

    return cfg


def main(argv: list[str] | None = None) -> int:
    # Ensure we operate from the template directory
    configure_logging("info")  # basic console logging before config is resolved
    script_dir = Path(__file__).resolve().parent
    if Path.cwd().resolve() != script_dir:
        logging.info("Switching working directory to %s", script_dir)
        os.chdir(script_dir)

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = resolve_config(args)
    setup_seed(cfg["seed"])

    # Prepare output directory with timestamped run folder
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out = Path(cfg["outdir"])
    run_dir = base_out / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging based on resolved config and add file handler in the run directory
    configure_logging(cfg.get("logging", {}).get("level", "info"), run_dir / "run.log")

    # Save resolved config snapshot
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2))

    # Execute training/eval pipeline
    train_and_evaluate(cfg, run_dir)

    logging.info("Finished. Artifacts in: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
