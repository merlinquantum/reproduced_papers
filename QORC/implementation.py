#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI entry-point for the quantum reservoir article
"""

##########################################################
# Launching commands
# $ micromamba activate qml-cpu
# $ python implementation.py
# $ python implementation.py --epochs 100 --batch-size 100 --learning-rate 0.05 --seed 42 --n-photons 3 --n-modes 12 --b-no-bunching False


##########################################################
# Librairies loading and functions definitions

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path

from lib.config import deep_update, load_config

from lib.lib_qorc_encoding_and_linear_training import qorc_encoding_and_linear_training
from lib.lib_datasets import download_and_save_mnist_with_keras_if_missing_files


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


# Command line arguments parsing


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper reproduction runner")
    p.add_argument("--config", type=str, help="Path to JSON config", default=None)
    p.add_argument("--outdir", type=str, help="Base output directory", default=None)
    p.add_argument(
        "--device", type=str, help="Device string (cpu, cuda:0, mps)", default=None
    )

    # Specific parameters
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--seed", type=int, help="Random seed", default=None)

    p.add_argument("--n-photons", type=int, default=None)
    p.add_argument("--n-modes", type=int, default=None)
    p.add_argument("--b-no-bunching", type=bool, default=None)

    return p


def resolve_config(args: argparse.Namespace):
    f_default_config = "configs/defaults.json"
    cfg = load_config(Path(f_default_config))

    # Load from file if provided
    if args.config:
        file_cfg = load_config(Path(args.config))
        cfg = deep_update(cfg, file_cfg)

    # Apply CLI overrides
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.device is not None:
        cfg["device"] = args.device

    # Specific parameters
    if args.epochs is not None:
        cfg["n_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.seed is not None:
        cfg["seed"] = args.seed

    if args.n_photons is not None:
        cfg["n_photons"] = args.n_photons
    if args.n_modes is not None:
        cfg["n_modes"] = args.n_modes
    if args.b_no_bunching is not None:
        cfg["b_no_bunching"] = args.b_no_bunching

    return cfg


# Call to main training function


def train_and_evaluate(cfg, run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting training")
    logger.debug("Resolved config: %s", json.dumps(cfg, indent=2))

    b_get_keras_mnist = cfg["b_get_keras_mnist"]
    if b_get_keras_mnist:
        # Requires an extra dependancy to Keras
        download_and_save_mnist_with_keras_if_missing_files(
            cfg["f_in_train"], cfg["f_in_test"], logger
        )

    outputs = qorc_encoding_and_linear_training(
        # Main parameters
        n_photons=cfg["n_photons"],
        n_modes=cfg["n_modes"],
        seed=cfg["seed"],
        # Dataset parameters
        f_in_train=cfg["f_in_train"],
        f_in_test=cfg["f_in_test"],
        fold_index=cfg["fold_index"],
        n_fold=cfg["n_fold"],
        n_pixels=cfg["n_pixels"],
        n_outputs=cfg["n_outputs"],
        # Training parameters
        n_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        reduce_lr_patience=cfg["reduce_lr_patience"],
        reduce_lr_factor=cfg["reduce_lr_factor"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        f_out_weights=cfg["f_out_weights"],
        # Other parameters
        b_no_bunching=cfg["b_no_bunching"],
        b_use_tensorboard=cfg["b_use_tensorboard"],
        device_name=cfg["device_name"],
        run_dir=run_dir,
        logger=logger,
    )

    (run_dir / "done.txt").write_text(str(outputs))
    logger.info("Written file: %s", run_dir / "done.txt")


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
