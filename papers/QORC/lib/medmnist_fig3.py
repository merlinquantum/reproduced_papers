from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from lib.comparison import train_linear_baseline
from lib.lib_datasets import get_qorc_dataset
from lib.lib_qorc_encoding_and_linear_training import (
    qorc_encoding_and_linear_training,
)

MLR_COLOR = "#008000"
QORC_COLOR = "#800080"


def run_medmnist_fig3(cfg: dict, run_dir: Path, logger: logging.Logger) -> None:
    """Run the QORC versus MLR macro-F1 comparison for Fig. 3.

    Parameters
    ----------
    cfg : dict
        Resolved Fig. 3 experiment configuration.
    run_dir : pathlib.Path
        Timestamped output directory for experiment artifacts.
    logger : logging.Logger
        Logger receiving experiment progress messages.

    Returns
    -------
    None
        Writes per-run metrics, summary metrics, and a bar plot to ``run_dir``.

    Raises
    ------
    ValueError
        If the configured datasets or number of seeds is invalid.
    """
    datasets = cfg["datasets"]
    seeds = cfg["seeds"]
    if not datasets:
        raise ValueError("datasets must contain at least one MedMNIST dataset.")
    if not seeds:
        raise ValueError("seeds must contain at least one run seed.")

    rows = []
    for dataset_name in datasets:
        normalized_dataset_name = dataset_name.lower()
        for seed in seeds:
            logger.info("Running Fig. 3 on %s with seed %s", dataset_name, seed)
            train_data, train_labels, test_data, test_labels = get_qorc_dataset(
                normalized_dataset_name, seed=seed
            )
            train_data = (
                train_data.reshape(train_data.shape[0], -1).astype(np.float32) / 255.0
            )
            test_data = (
                test_data.reshape(test_data.shape[0], -1).astype(np.float32) / 255.0
            )
            n_classes = int(max(np.max(train_labels), np.max(test_labels))) + 1

            qorc_result = qorc_encoding_and_linear_training(
                n_photons=cfg["n_photons"],
                n_modes=cfg["n_modes"],
                seed=seed,
                dataset_name=normalized_dataset_name,
                dataset_sampling="full",
                dataset_sample_count=None,
                dataset_samples_per_class=None,
                fold_index=0,
                n_fold=0,
                dataset_truncate=cfg.get("dataset_truncate", 0),
                n_epochs=cfg["n_epochs"],
                batch_size=cfg["batch_size"],
                learning_rate=cfg["learning_rate"],
                reduce_lr_patience=cfg["reduce_lr_patience"],
                reduce_lr_factor=cfg["reduce_lr_factor"],
                num_workers=cfg["num_workers"],
                pin_memory=cfg["pin_memory"],
                f_out_weights=f"qorc_{normalized_dataset_name}_{seed}.pth",
                save_weights=cfg["save_weights"],
                b_no_bunching=cfg["b_no_bunching"],
                b_use_tensorboard=cfg["b_use_tensorboard"],
                noise_enabled=cfg.get("noise_enabled", False),
                noise_indistinguishability=cfg.get("noise_indistinguishability", 1.0),
                noise_g2=cfg.get("noise_g2", 0.0),
                noise_g2_distinguishable=cfg.get("noise_g2_distinguishable", True),
                device_name=cfg["device"],
                qpu_device_name=cfg.get("qpu_device", "none"),
                qpu_device_nsample=cfg.get("qpu_device_nsample", 10000),
                run_dir=run_dir,
                logger=logger,
                return_history=True,
            )
            qorc_f1 = f1_score(
                qorc_result["test_targets"],
                qorc_result["test_predictions"],
                average="macro",
            )

            mlr_history = train_linear_baseline(
                train_data,
                train_labels,
                test_data,
                test_labels,
                n_epochs=cfg["n_epochs"],
                batch_size=cfg["batch_size"],
                learning_rate=cfg["linear_learning_rate"],
                seed=seed,
                n_classes=n_classes,
            )
            mlr_f1 = f1_score(
                test_labels,
                mlr_history["test_predictions"],
                average="macro",
            )
            rows.extend(
                [
                    {
                        "dataset": dataset_name,
                        "model": "QORC",
                        "seed": seed,
                        "macro_f1": float(qorc_f1),
                    },
                    {
                        "dataset": dataset_name,
                        "model": "MLR",
                        "seed": seed,
                        "macro_f1": float(mlr_f1),
                    },
                ]
            )

    with (run_dir / "fig3_qorc_mlr_medmnist.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["dataset", "model", "seed", "macro_f1"]
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize_rows(rows, datasets)
    (run_dir / "fig3_qorc_mlr_medmnist.json").write_text(
        json.dumps({"runs": rows, "summary": summary}, indent=2), encoding="utf-8"
    )
    plot_medmnist_fig3(summary, run_dir / "fig3_qorc_mlr_medmnist.png")


def _summarize_rows(rows: list[dict], datasets: list[str]) -> list[dict]:
    summary = []
    for dataset in datasets:
        for model in ("MLR", "QORC"):
            values = np.asarray(
                [
                    row["macro_f1"]
                    for row in rows
                    if row["dataset"] == dataset and row["model"] == model
                ],
                dtype=float,
            )
            summary.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "mean_macro_f1": float(values.mean()),
                    "std_macro_f1": float(values.std()),
                    "n_runs": int(values.size),
                }
            )
    return summary


def plot_medmnist_fig3(summary: list[dict], output_path: Path) -> None:
    """Plot mean macro-F1 scores with standard-deviation error bars.

    Parameters
    ----------
    summary : list[dict]
        Dataset/model summary rows containing means and standard deviations.
    output_path : pathlib.Path
        PNG path for the generated figure.

    Returns
    -------
    None
        Saves and closes the generated bar plot.
    """
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(row["dataset"] for row in summary))
    x_values = np.arange(len(datasets))
    width = 0.34
    figure, axis = plt.subplots(figsize=(6.4, 4.5))
    for offset, model, color in (
        (-width / 2, "MLR", MLR_COLOR),
        (width / 2, "QORC", QORC_COLOR),
    ):
        values = [
            next(
                row
                for row in summary
                if row["dataset"] == dataset and row["model"] == model
            )
            for dataset in datasets
        ]
        means = np.asarray([row["mean_macro_f1"] for row in values])
        errors = np.asarray([row["std_macro_f1"] for row in values])
        bars = axis.bar(
            x_values + offset,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=color,
            label=model,
        )
        axis.bar_label(
            bars, labels=[f"{mean:.2f}" for mean in means], padding=2, fontsize=8
        )

    axis.set_ylabel("Macro F1 Score")
    axis.set_xticks(x_values, datasets)
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3, linestyle="--")
    axis.set_axisbelow(True)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
