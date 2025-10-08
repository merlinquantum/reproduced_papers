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
# $ python implementation.py --config configs/xp_qorc.json


##########################################################
# Librairies loading and functions definitions

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import List

from lib.config import deep_update, load_config

from lib.lib_qorc_encoding_and_linear_training import qorc_encoding_and_linear_training
from lib.lib_rff_encoding_and_linear_training import rff_encoding_and_linear_training

import pandas as pd


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

    # Specific parameters to qorc
    p.add_argument("--n-photons", type=int, default=None, help="Number of photons")
    p.add_argument("--n-modes", type=int, default=None, help="Number of modes")
    p.add_argument("--seed", type=int, help="Random seed", default=None)
    p.add_argument(
        "--fold-index", type=int, default=None, help="Split train/val fold index"
    )
    p.add_argument(
        "--n-fold", type=int, default=None, help="Split train/val number of folds"
    )
    p.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size")
    p.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    p.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=None,
        help="Reduce learning rate patience",
    )
    p.add_argument(
        "--reduce-lr-factor",
        type=float,
        default=None,
        help="Reduce learning rate factor",
    )
    p.add_argument(
        "--num-workers", type=int, default=None, help="Number of dataloader workers"
    )
    p.add_argument("--pin-memory", type=bool, default=None, help="Enable pin memory")
    p.add_argument(
        "--f-out-weights", type=str, default=None, help="Model checkpoint filepath"
    )
    p.add_argument("--b-no-bunching", type=bool, default=None, help="Disable bunching")
    p.add_argument(
        "--b-use-tensorboard",
        type=bool,
        default=None,
        help="Enable TensorBoard logging",
    )
    p.add_argument(
        "--device", type=str, help="Device string (cpu, cuda:0, mps)", default=None
    )

    # Specific parameters to rff
    p.add_argument(
        "--n-rff-features", type=int, default=None, help="Number of RFF features"
    )
    p.add_argument("--sigma", type=float, default=None, help="RBF kernel bandwidth")
    p.add_argument(
        "--regularization-c",
        type=float,
        default=None,
        help="Regularization strength (C)",
    )
    p.add_argument(
        "--b-optim-via-sgd", type=bool, default=None, help="Use SGD for optimization"
    )
    p.add_argument("--max-iter-sgd", type=int, default=None, help="Max SGD iterations")

    return p


def resolve_config(args: argparse.Namespace):
    cfg = {}

    # Load from file if provided
    if args.config:
        file_cfg = load_config(Path(args.config))
        cfg = deep_update(cfg, file_cfg)

    # Apply CLI overrides
    if args.outdir is not None:
        cfg["outdir"] = args.outdir

    # Specific parameters to qorc
    if args.n_photons is not None:
        cfg["n_photons"] = args.n_photons
    if args.n_modes is not None:
        cfg["n_modes"] = args.n_modes
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.fold_index is not None:
        cfg["fold_index"] = args.fold_index
    if args.n_fold is not None:
        cfg["n_fold"] = args.n_fold
    if args.epochs is not None:
        cfg["n_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.reduce_lr_patience is not None:
        cfg["reduce_lr_patience"] = args.reduce_lr_patience
    if args.reduce_lr_factor is not None:
        cfg["reduce_lr_factor"] = args.reduce_lr_factor
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.pin_memory is not None:
        cfg["pin_memory"] = args.pin_memory
    if args.f_out_weights is not None:
        cfg["f_out_weights"] = args.f_out_weights
    if args.b_no_bunching is not None:
        cfg["b_no_bunching"] = args.b_no_bunching
    if args.b_use_tensorboard is not None:
        cfg["b_use_tensorboard"] = args.b_use_tensorboard
    if args.device is not None:
        cfg["device"] = args.device

    # Specific parameters to rff
    if args.n_rff_features is not None:
        cfg["n_rff_features"] = args.n_rff_features
    if args.sigma is not None:
        cfg["sigma"] = args.sigma
    if args.regularization_c is not None:
        cfg["regularization_c"] = args.regularization_c
    if args.b_optim_via_sgd is not None:
        cfg["b_optim_via_sgd"] = args.b_optim_via_sgd
    if args.max_iter_sgd is not None:
        cfg["max_iter_sgd"] = args.max_iter_sgd

    return cfg


# Call to main training function


def train_and_evaluate(cfg, run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting training")
    logger.debug("Resolved config: %s", json.dumps(cfg, indent=2))

    xp_type = cfg["xp_type"]

    if xp_type == "qorc":
        n_photons = cfg["n_photons"]
        n_modes = cfg["n_modes"]
        seed = cfg["seed"]
        fold_index = cfg["fold_index"]

        # Run with 4 loops over photons, modes, seed, fold
        if (
            isinstance(n_photons, List)
            or isinstance(n_modes, List)
            or isinstance(fold_index, List)
            or isinstance(seed, List)
        ):
            logger.info("Entering loop training over fold/seed/photons/modes:")
            f_out_results_training_csv = cfg["f_out_results_training_csv"]
            assert len(f_out_results_training_csv) > 0, (
                "Error: Empty f_out_results_training_csv"
            )
            f_out_results_training_csv = os.path.join(
                run_dir, f_out_results_training_csv
            )

            if not isinstance(n_photons, List):
                n_photons = [n_photons]

            if not isinstance(n_modes, List):
                n_modes = [n_modes]

            if not isinstance(fold_index, List):
                fold_index = [fold_index]

            if not isinstance(seed, List):
                seed = [seed]

            # Structure to be fed
            df = pd.DataFrame()
            for i, current_n_photons in enumerate(n_photons):
                for j, current_n_modes in enumerate(n_modes):
                    for k, current_fold_index in enumerate(fold_index):
                        for l, current_seed in enumerate(seed):
                            logger.info(
                                "loop index: n_photons {}/{}, n_modes {}/{}, fold_index {}/{}, seed {}/{}".format(
                                    i + 1,
                                    len(n_photons),
                                    j + 1,
                                    len(n_modes),
                                    k + 1,
                                    len(fold_index),
                                    l + 1,
                                    len(seed),
                                )
                            )

                            logger.info(
                                "values: n_photons {}, n_modes {}, fold_index {}, seed {}".format(
                                    current_n_photons,
                                    current_n_modes,
                                    current_fold_index,
                                    current_seed,
                                )
                            )

                            # Sigle run per iteration
                            [
                                train_acc,
                                val_acc,
                                test_acc,
                                qorc_output_size,
                                n_train_epochs,
                                duration_qfeatures,
                                duration_train,
                                best_val_epoch,
                            ] = qorc_encoding_and_linear_training(
                                # Main parameters
                                n_photons=current_n_photons,
                                n_modes=current_n_modes,
                                seed=current_seed,
                                # Dataset parameters
                                fold_index=current_fold_index,
                                n_fold=cfg["n_fold"],
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
                                device_name=cfg["device"],
                                run_dir=run_dir,
                                logger=logger,
                            )

                            # Save outputs in the dataFrame and then save the current dataframe
                            output_fields = {
                                "n_photons": current_n_photons,
                                "n_modes": current_n_modes,
                                "seed": current_seed,
                                "fold_index": current_fold_index,
                                "train_acc": train_acc,
                                "val_acc": val_acc,
                                "test_acc": test_acc,
                                "qorc_output_size": qorc_output_size,
                                "n_train_epochs": n_train_epochs,
                                "duration_qfeatures": duration_qfeatures,
                                "duration_train": duration_train,
                                "best_val_epoch": best_val_epoch,
                            }
                            df_line = pd.DataFrame([output_fields])

                            if df.empty:
                                df = df_line
                            else:
                                df = pd.concat([df, df_line], ignore_index=True)
                            df.to_csv(f_out_results_training_csv, index=False)
                            logger.info("Written file: %s", f_out_results_training_csv)

        else:
            # Single run
            outputs = qorc_encoding_and_linear_training(
                # Main parameters
                n_photons=cfg["n_photons"],
                n_modes=cfg["n_modes"],
                seed=cfg["seed"],
                # Dataset parameters
                fold_index=cfg["fold_index"],
                n_fold=cfg["n_fold"],
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
                device_name=cfg["device"],
                run_dir=run_dir,
                logger=logger,
            )

            (run_dir / "done.txt").write_text(str(outputs))
            logger.info("Written file: %s", run_dir / "done.txt")

    if xp_type == "rff":
        n_rff_features = cfg["n_rff_features"]
        seed = cfg["seed"]

        if isinstance(seed, List) or isinstance(n_rff_features, List):
            logger.info("Entering loop training over seed/n_rff_features:")
            f_out_results_training_csv = cfg["f_out_results_training_csv"]
            assert len(f_out_results_training_csv) > 0, (
                "Error: Empty f_out_results_training_csv"
            )
            f_out_results_training_csv = os.path.join(
                run_dir, f_out_results_training_csv
            )

            if not isinstance(n_rff_features, List):
                n_rff_features = [n_rff_features]

            if not isinstance(seed, List):
                seed = [seed]

            # Structure to be fed
            df = pd.DataFrame()
            for i, current_n_rff_features in enumerate(n_rff_features):
                for j, current_seed in enumerate(seed):
                    logger.info(
                        "loop index: n_rff_features {}/{}, seed {}/{}".format(
                            i + 1,
                            len(n_rff_features),
                            j + 1,
                            len(seed),
                        )
                    )

                    logger.info(
                        "values: n_rff_features {}, seed {}".format(
                            current_n_rff_features,
                            current_seed,
                        )
                    )

                    [
                        train_acc,
                        test_acc,
                        duration_calcul_rff_features,
                        duration_train,
                    ] = rff_encoding_and_linear_training(
                        # Main parameters
                        n_rff_features=current_n_rff_features,
                        sigma=cfg["sigma"],
                        regularization_c=cfg["regularization_c"],
                        seed=current_seed,
                        b_optim_via_sgd=cfg["b_optim_via_sgd"],
                        max_iter_sgd=cfg["max_iter_sgd"],
                        # Dataset parameters
                        run_dir=run_dir,
                        logger=logger,
                    )

                    # Save outputs in the dataFrame and then save the current dataframe
                    output_fields = {
                        "n_rff_features": current_n_rff_features,
                        "seed": current_seed,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "duration_calcul_rff_features": duration_calcul_rff_features,
                        "duration_train": duration_train,
                    }
                    df_line = pd.DataFrame([output_fields])

                    if df.empty:
                        df = df_line
                    else:
                        df = pd.concat([df, df_line], ignore_index=True)
                    df.to_csv(f_out_results_training_csv, index=False)
                    logger.info("Written file: %s", f_out_results_training_csv)

        else:
            outputs = rff_encoding_and_linear_training(
                # Main parameters
                n_rff_features=cfg["n_rff_features"],
                sigma=cfg["sigma"],
                regularization_c=cfg["regularization_c"],
                seed=cfg["seed"],
                b_optim_via_sgd=cfg["b_optim_via_sgd"],
                max_iter_sgd=cfg["max_iter_sgd"],
                # Dataset parameters
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
