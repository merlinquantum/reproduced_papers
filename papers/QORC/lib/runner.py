from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from lib.lib_qorc_encoding_and_linear_training import qorc_encoding_and_linear_training
from lib.lib_rff_encoding_and_linear_training import rff_encoding_and_linear_training


def _as_list(value):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def train_and_evaluate(cfg, run_dir: Path) -> None:
    logger = logging.getLogger(__name__)

    xp_type = cfg["xp_type"].lower()
    if xp_type == "qorc":
        _run_qorc(cfg, run_dir, logger)
    elif xp_type == "rff":
        _run_rff(cfg, run_dir, logger)
    else:
        raise ValueError(f"Unsupported xp_type: {xp_type}")


def _run_qorc(cfg, run_dir: Path, logger: logging.Logger) -> None:
    n_photons = cfg["n_photons"]
    n_modes = cfg["n_modes"]
    seeds = cfg["seed"]
    fold_index = cfg["fold_index"]
    dataset_name = cfg.get("dataset_name", "mnist")
    dataset_truncate = cfg.get("dataset_truncate", 0)
    qpu_device_name = cfg.get("qpu_device_name", cfg.get("qpu_device", "none"))
    qpu_device_nsample = cfg.get("qpu_device_nsample", 10000)

    if any(
        isinstance(val, Sequence) and not isinstance(val, (str, bytes))
        for val in (n_photons, n_modes, seeds, fold_index)
    ):
        logger.info("Entering sweep over photons/modes/folds/seeds")
        out_csv = cfg.get("f_out_results_training_csv")
        if not out_csv:
            raise ValueError("f_out_results_training_csv must be set for sweep runs")
        csv_path = run_dir / out_csv
        df = pd.DataFrame()
        photons_list = _as_list(n_photons)
        modes_list = _as_list(n_modes)
        folds_list = _as_list(fold_index)
        seeds_list = _as_list(seeds)
        for i, photons in enumerate(photons_list):
            for j, modes in enumerate(modes_list):
                for k, fold in enumerate(folds_list):
                    for l, seed in enumerate(seeds_list):
                        logger.info(
                            "Loop indices: n_photons %s/%s, n_modes %s/%s, fold %s/%s, seed %s/%s",
                            i + 1,
                            len(photons_list),
                            j + 1,
                            len(modes_list),
                            k + 1,
                            len(folds_list),
                            l + 1,
                            len(seeds_list),
                        )
                        logger.info(
                            "Values: n_photons %s, n_modes %s, fold_index %s, seed %s",
                            photons,
                            modes,
                            fold,
                            seed,
                        )
                        (
                            train_acc,
                            val_acc,
                            test_acc,
                            qorc_output_size,
                            n_train_epochs,
                            duration_qfeatures,
                            duration_train,
                            best_val_epoch,
                        ) = qorc_encoding_and_linear_training(
                            n_photons=photons,
                            n_modes=modes,
                            seed=seed,
                            dataset_name=dataset_name,
                            fold_index=fold,
                            n_fold=cfg["n_fold"],
                            dataset_truncate=dataset_truncate,
                            n_epochs=cfg["n_epochs"],
                            batch_size=cfg["batch_size"],
                            learning_rate=cfg["learning_rate"],
                            reduce_lr_patience=cfg["reduce_lr_patience"],
                            reduce_lr_factor=cfg["reduce_lr_factor"],
                            num_workers=cfg["num_workers"],
                            pin_memory=cfg["pin_memory"],
                            f_out_weights=cfg["f_out_weights"],
                            b_no_bunching=cfg["b_no_bunching"],
                            b_use_tensorboard=cfg["b_use_tensorboard"],
                            device_name=cfg["device"],
                            qpu_device_name=qpu_device_name,
                            qpu_device_nsample=qpu_device_nsample,
                            run_dir=run_dir,
                            logger=logger,
                        )
                        df_line = pd.DataFrame(
                            [
                                {
                                    "n_photons": photons,
                                    "n_modes": modes,
                                    "seed": seed,
                                    "fold_index": fold,
                                    "train_acc": train_acc,
                                    "val_acc": val_acc,
                                    "test_acc": test_acc,
                                    "qorc_output_size": qorc_output_size,
                                    "n_train_epochs": n_train_epochs,
                                    "duration_qfeatures": duration_qfeatures,
                                    "duration_train": duration_train,
                                    "best_val_epoch": best_val_epoch,
                                }
                            ]
                        )
                        df = pd.concat([df, df_line], ignore_index=True)
                        df.to_csv(csv_path, index=False)
                        logger.info("Written file: %s", csv_path)
        return

    outputs = qorc_encoding_and_linear_training(
        n_photons=n_photons,
        n_modes=n_modes,
        seed=seeds,
        dataset_name=dataset_name,
        fold_index=fold_index,
        n_fold=cfg["n_fold"],
        dataset_truncate=dataset_truncate,
        n_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        reduce_lr_patience=cfg["reduce_lr_patience"],
        reduce_lr_factor=cfg["reduce_lr_factor"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        f_out_weights=cfg["f_out_weights"],
        b_no_bunching=cfg["b_no_bunching"],
        b_use_tensorboard=cfg["b_use_tensorboard"],
        device_name=cfg["device"],
        qpu_device_name=qpu_device_name,
        qpu_device_nsample=qpu_device_nsample,
        run_dir=run_dir,
        logger=logger,
    )
    (run_dir / "done.txt").write_text(str(outputs), encoding="utf-8")
    logger.info("Written file: %s", run_dir / "done.txt")


def _run_rff(cfg, run_dir: Path, logger: logging.Logger) -> None:
    n_rff_features = cfg["n_rff_features"]
    seeds = cfg["seed"]
    dataset_name = cfg.get("dataset_name", "mnist")

    if any(
        isinstance(val, Sequence) and not isinstance(val, (str, bytes))
        for val in (n_rff_features, seeds)
    ):
        logger.info("Entering sweep over n_rff_features/seed")
        out_csv = cfg.get("f_out_results_training_csv")
        if not out_csv:
            raise ValueError("f_out_results_training_csv must be set for sweep runs")
        csv_path = run_dir / out_csv
        df = pd.DataFrame()
        features_list = _as_list(n_rff_features)
        seeds_list = _as_list(seeds)
        for i, features in enumerate(features_list):
            for j, seed in enumerate(seeds_list):
                logger.info(
                    "Loop indices: n_rff_features %s/%s, seed %s/%s",
                    i + 1,
                    len(features_list),
                    j + 1,
                    len(seeds_list),
                )
                logger.info("Values: n_rff_features %s, seed %s", features, seed)
                (
                    train_acc,
                    test_acc,
                    duration_calcul_rff_features,
                    duration_train,
                ) = rff_encoding_and_linear_training(
                    n_rff_features=features,
                    sigma=cfg["sigma"],
                    regularization_c=cfg["regularization_c"],
                    seed=seed,
                    b_optim_via_sgd=cfg["b_optim_via_sgd"],
                    max_iter_sgd=cfg["max_iter_sgd"],
                    dataset_name=dataset_name,
                    run_dir=run_dir,
                    logger=logger,
                )
                df_line = pd.DataFrame(
                    [
                        {
                            "n_rff_features": features,
                            "seed": seed,
                            "train_acc": train_acc,
                            "test_acc": test_acc,
                            "duration_calcul_rff_features": duration_calcul_rff_features,
                            "duration_train": duration_train,
                        }
                    ]
                )
                df = pd.concat([df, df_line], ignore_index=True)
                df.to_csv(csv_path, index=False)
                logger.info("Written file: %s", csv_path)
        return

    outputs = rff_encoding_and_linear_training(
        n_rff_features=n_rff_features,
        sigma=cfg["sigma"],
        regularization_c=cfg["regularization_c"],
        seed=seeds,
        b_optim_via_sgd=cfg["b_optim_via_sgd"],
        max_iter_sgd=cfg["max_iter_sgd"],
        dataset_name=dataset_name,
        run_dir=run_dir,
        logger=logger,
    )
    (run_dir / "done.txt").write_text(str(outputs), encoding="utf-8")
    logger.info("Written file: %s", run_dir / "done.txt")
