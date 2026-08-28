from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

from lib.lib_qorc_encoding_and_linear_training import (
    qorc_encoding_and_linear_training,
)

QORC_M20_COLOR = "#ff7f0e"
QORC_M12_COLOR = "#8c564b"


def run_noisy_qorc_indistinguishability(cfg, run_dir: Path, logger: logging.Logger):
    """Run QORC accuracy sweeps over photon indistinguishability.

    Parameters
    ----------
    cfg : dict
        Resolved experiment configuration.
    run_dir : pathlib.Path
        Directory receiving the sweep data and figure.
    logger : logging.Logger
        Logger receiving experiment progress messages.

    Returns
    -------
    None
        Writes CSV, JSON, and PNG artifacts to ``run_dir``.

    Raises
    ------
    ValueError
        If the two mode sweeps or backend configuration is invalid.
    """
    if cfg.get("dataset_name", "mnist").lower() != "mnist":
        raise ValueError("The indistinguishability experiment requires MNIST.")
    if cfg.get("use_qpu", False) and not cfg.get("qpu_device", "").startswith("qpu:"):
        raise ValueError("use_qpu=True requires qpu_device to start with 'qpu:'.")

    sweep_definitions = [
        (12, cfg["indistinguishability_m12"], QORC_M12_COLOR),
        (20, cfg["indistinguishability_m20"], QORC_M20_COLOR),
    ]
    rows = []
    for n_modes, indistinguishability_values, _ in sweep_definitions:
        for value_index, indistinguishability_percent in enumerate(
            indistinguishability_values
        ):
            logger.info(
                "Running QORC N=%s, M=%s, indistinguishability=%s%%",
                cfg["n_photons"],
                n_modes,
                indistinguishability_percent,
            )
            result = qorc_encoding_and_linear_training(
                n_photons=cfg["n_photons"],
                n_modes=n_modes,
                seed=cfg["seed"] + value_index,
                dataset_name="mnist",
                dataset_sampling="full",
                dataset_sample_count=None,
                dataset_samples_per_class=None,
                noise_enabled=True,
                noise_indistinguishability=indistinguishability_percent / 100.0,
                noise_g2=cfg["noise_g2"],
                noise_g2_distinguishable=cfg["noise_g2_distinguishable"],
                fold_index=0,
                n_fold=0,
                dataset_truncate=0,
                n_epochs=cfg["n_epochs"],
                batch_size=cfg["batch_size"],
                learning_rate=cfg["learning_rate"],
                reduce_lr_patience=cfg["reduce_lr_patience"],
                reduce_lr_factor=cfg["reduce_lr_factor"],
                num_workers=cfg["num_workers"],
                pin_memory=cfg["pin_memory"],
                f_out_weights=f"noisy_qorc_m{n_modes}_{value_index}.pth",
                b_no_bunching=cfg["b_no_bunching"],
                b_use_tensorboard=False,
                device_name=cfg["device"],
                qpu_device_name=cfg.get("qpu_device", "none"),
                qpu_device_nsample=cfg.get("qpu_device_nsample", 10000),
                run_dir=run_dir,
                logger=logger,
            )
            rows.append(
                {
                    "n_photons": cfg["n_photons"],
                    "n_modes": n_modes,
                    "indistinguishability_percent": indistinguishability_percent,
                    "train_accuracy": result[0],
                    "test_accuracy": result[2],
                }
            )

    rows_path = run_dir / "noisy_QORC_indistinguishability.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "rows": rows,
        "mlr_train_accuracy": cfg["mlr_train_accuracy"],
        "mlr_test_accuracy": cfg["mlr_test_accuracy"],
    }
    (run_dir / "noisy_QORC_indistinguishability.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    plot_noisy_qorc_indistinguishability(
        metrics, run_dir / "noisy_QORC_indistinguishability.png"
    )


def plot_noisy_qorc_indistinguishability(metrics: dict, output_path: Path):
    """Plot QORC accuracies and linear regression fits versus indistinguishability.

    Parameters
    ----------
    metrics : dict
        Sweep rows and MLR reference accuracies.
    output_path : pathlib.Path
        PNG output path.

    Returns
    -------
    None
        Saves and closes the generated figure.
    """
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5.5))
    for n_modes, color in ((20, QORC_M20_COLOR), (12, QORC_M12_COLOR)):
        points = [row for row in metrics["rows"] if row["n_modes"] == n_modes]
        points.sort(key=lambda row: row["indistinguishability_percent"])
        x_values = np.asarray(
            [row["indistinguishability_percent"] for row in points], dtype=float
        )
        train_values = np.asarray([row["train_accuracy"] for row in points])
        test_values = np.asarray([row["test_accuracy"] for row in points])
        axis.plot(
            x_values,
            train_values,
            marker="o",
            linestyle="none",
            color=color,
            label=f"QORC train (M={n_modes})",
        )
        axis.plot(
            x_values,
            test_values,
            marker="s",
            linestyle="none",
            color=color,
            alpha=0.9,
            label=f"QORC test (M={n_modes})",
        )
        train_fit = np.polyfit(x_values, train_values, 1)
        test_fit = np.polyfit(x_values, test_values, 1)
        fit_x = np.linspace(x_values.min(), x_values.max(), 100)
        axis.plot(fit_x, np.polyval(train_fit, fit_x), color=color, linestyle="-")
        axis.plot(fit_x, np.polyval(test_fit, fit_x), color=color, linestyle=":")

    axis.axhline(
        metrics["mlr_test_accuracy"], color="black", linestyle="-", label="MLR test"
    )
    axis.axhline(
        metrics["mlr_train_accuracy"],
        color="black",
        linestyle=":",
        label="MLR train",
    )
    axis.set_xlabel("Indistinguishability (%)")
    axis.set_ylabel("Accuracy")
    axis.set_xlim(-2, 102)
    axis.set_ylim(0.92, 1.0)
    axis.grid(alpha=0.3, linestyle="--")
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
