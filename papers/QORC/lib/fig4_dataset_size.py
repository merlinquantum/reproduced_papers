from __future__ import annotations

import csv
import gc
import json
import logging
from pathlib import Path

import numpy as np

from lib.comparison import train_linear_baseline
from lib.lib_datasets import get_qorc_dataset
from lib.lib_qorc_encoding_and_linear_training import (
    qorc_encoding_and_linear_training,
)

MLR_COLOR = "#000080"
IDEAL_COLOR = "#008c95"
NOISY_COLOR = "#800080"
QPU_COLOR = "#ff8c00"


def _condition_definitions(cfg: dict) -> list[dict]:
    conditions = [
        {
            "name": "MLR",
            "color": MLR_COLOR,
            "kind": "mlr",
        },
        {
            "name": "QORC (ideal)",
            "color": IDEAL_COLOR,
            "kind": "qorc",
            "noise_enabled": False,
            "qpu_device": "none",
        },
        {
            "name": "QORC (noisy)",
            "color": NOISY_COLOR,
            "kind": "qorc",
            "noise_enabled": True,
            "noise_g2": cfg["noise_g2"],
            "noise_indistinguishability": cfg["noise_indistinguishability"],
            "qpu_device": "none",
        },
    ]
    if cfg["enable_qpu"]:
        conditions.append(
            {
                "name": "QORC (QPU)",
                "color": QPU_COLOR,
                "kind": "qorc",
                "noise_enabled": False,
                "qpu_device": cfg["qpu_device"],
                "qpu": True,
            }
        )
    requested_conditions = cfg.get("conditions")
    if requested_conditions is None:
        return conditions
    available_conditions = {condition["name"]: condition for condition in conditions}
    unknown_conditions = set(requested_conditions) - set(available_conditions)
    if unknown_conditions:
        raise ValueError(f"Unknown Fig. 4 conditions: {sorted(unknown_conditions)}")
    return [available_conditions[name] for name in requested_conditions]


def run_fig4_dataset_size(cfg: dict, run_dir: Path, logger: logging.Logger) -> None:
    """Run the Fig. 4 MNIST training-set-size comparison.

    Parameters
    ----------
    cfg : dict
        Resolved Fig. 4 experiment configuration.
    run_dir : pathlib.Path
        Timestamped output directory for experiment artifacts.
    logger : logging.Logger
        Logger receiving experiment progress messages.

    Returns
    -------
    None
        Writes per-subset metrics, summary metrics, and the figure to ``run_dir``.

    Raises
    ------
    ValueError
        If a training size or subset count is invalid.
    """
    training_sizes = cfg["training_sizes"]
    n_subsets = cfg["n_subsets"]
    if any(size <= 0 or size % 10 for size in training_sizes):
        raise ValueError("training_sizes must be positive multiples of 10.")
    if n_subsets <= 0:
        raise ValueError("n_subsets must be positive.")

    rows = []
    for training_size in training_sizes:
        for subset_index in range(n_subsets):
            subset_seed = cfg["seed"] + subset_index
            train_data, train_labels, test_data, test_labels = get_qorc_dataset(
                "mnist",
                sampling="balanced",
                sample_count=training_size,
                samples_per_class=training_size // 10,
                seed=subset_seed,
            )
            train_data = (
                train_data.reshape(train_data.shape[0], -1).astype(np.float32) / 255.0
            )
            test_data = (
                test_data.reshape(test_data.shape[0], -1).astype(np.float32) / 255.0
            )
            n_classes = int(max(np.max(train_labels), np.max(test_labels))) + 1

            for condition in _condition_definitions(cfg):
                if (
                    condition.get("qpu", False)
                    and training_size > cfg["qpu_max_training_size"]
                ):
                    continue
                logger.info(
                    "Fig. 4: %s, ntr=%s, subset=%s/%s",
                    condition["name"],
                    training_size,
                    subset_index + 1,
                    n_subsets,
                )
                if condition["kind"] == "mlr":
                    history = train_linear_baseline(
                        train_data,
                        train_labels,
                        test_data,
                        test_labels,
                        n_epochs=cfg["n_epochs"],
                        batch_size=cfg["batch_size"],
                        learning_rate=cfg["learning_rate"],
                        seed=subset_seed,
                        n_classes=n_classes,
                    )
                    train_accuracy = history["train_accuracy"][-1]
                    test_accuracy = history["test_accuracy"][-1]
                else:
                    qpu_test_size = 0
                    if condition.get("qpu", False):
                        qpu_test_size = cfg["qpu_total_images"] - training_size
                        if qpu_test_size <= 0:
                            raise ValueError(
                                "qpu_total_images must be greater than every QPU training size."
                            )
                    result = qorc_encoding_and_linear_training(
                        n_photons=cfg["n_photons"],
                        n_modes=cfg["n_modes"],
                        seed=subset_seed,
                        dataset_name="mnist",
                        dataset_sampling="balanced",
                        dataset_sample_count=training_size,
                        dataset_samples_per_class=training_size // 10,
                        fold_index=0,
                        n_fold=0,
                        dataset_truncate=0,
                        test_dataset_truncate=qpu_test_size,
                        feature_batch_size=(
                            cfg["noisy_feature_batch_size"]
                            if condition["name"] == "QORC (noisy)"
                            else cfg["feature_batch_size"]
                        ),
                        n_epochs=cfg["n_epochs"],
                        batch_size=cfg["batch_size"],
                        learning_rate=cfg["learning_rate"],
                        reduce_lr_patience=cfg["reduce_lr_patience"],
                        reduce_lr_factor=cfg["reduce_lr_factor"],
                        num_workers=cfg["num_workers"],
                        pin_memory=cfg["pin_memory"],
                        f_out_weights=f"fig4_{condition['name'].replace(' ', '_')}_{training_size}_{subset_index}.pth",
                        save_weights=cfg["save_weights"],
                        b_no_bunching=cfg["b_no_bunching"],
                        b_use_tensorboard=cfg["b_use_tensorboard"],
                        noise_enabled=condition.get("noise_enabled", False),
                        noise_indistinguishability=condition.get(
                            "noise_indistinguishability", 1.0
                        ),
                        noise_g2=condition.get("noise_g2", 0.0),
                        noise_g2_distinguishable=cfg["noise_g2_distinguishable"],
                        device_name=cfg["device"],
                        qpu_device_name=condition["qpu_device"],
                        qpu_device_nsample=cfg["qpu_device_nsample"],
                        run_dir=run_dir,
                        logger=logger,
                        return_history=False,
                    )
                    train_accuracy, _, test_accuracy = result[:3]
                rows.extend(
                    [
                        {
                            "training_size": training_size,
                            "subset": subset_index,
                            "model": condition["name"],
                            "split": "train",
                            "accuracy": float(train_accuracy),
                        },
                        {
                            "training_size": training_size,
                            "subset": subset_index,
                            "model": condition["name"],
                            "split": "test",
                            "accuracy": float(test_accuracy),
                        },
                    ]
                )
                _write_results_csv(rows, run_dir / "fig4_dataset_size_comparison.csv")
                gc.collect()

    summary = _summarize_rows(rows)
    output_csv = run_dir / "fig4_dataset_size_comparison.csv"
    _write_results_csv(rows, output_csv)
    output_json = run_dir / "fig4_dataset_size_comparison.json"
    output_json.write_text(
        json.dumps(
            {"runs": rows, "summary": summary, "hill_fits": cfg["hill_fits"]}, indent=2
        ),
        encoding="utf-8",
    )
    plot_fig4_dataset_size(summary, cfg, run_dir / "fig4_dataset_size_comparison.png")


def _write_results_csv(rows: list[dict], output_path: Path) -> None:
    """Atomically write the completed Fig. 4 experiments to a CSV file.

    Parameters
    ----------
    rows : list[dict]
        Completed training and test accuracy rows.
    output_path : pathlib.Path
        Destination CSV path.

    Returns
    -------
    None
        Writes the current results snapshot to ``output_path``.
    """
    temporary_path = output_path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["training_size", "subset", "model", "split", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(output_path)


def _summarize_rows(rows: list[dict]) -> list[dict]:
    summary = []
    keys = sorted({(row["training_size"], row["model"], row["split"]) for row in rows})
    for training_size, model, split in keys:
        values = np.asarray(
            [
                row["accuracy"]
                for row in rows
                if row["training_size"] == training_size
                and row["model"] == model
                and row["split"] == split
            ],
            dtype=float,
        )
        summary.append(
            {
                "training_size": training_size,
                "model": model,
                "split": split,
                "mean_accuracy": float(values.mean()),
                "std_accuracy": float(values.std()),
                "n_subsets": int(values.size),
            }
        )
    return summary


def _hill_function(x_values: np.ndarray, parameters: list[float]) -> np.ndarray:
    limit, exponent, midpoint = parameters
    return limit / (1.0 + (midpoint / x_values) ** exponent)


def plot_fig4_dataset_size(summary: list[dict], cfg: dict, output_path: Path) -> None:
    """Plot Fig. 4 accuracy curves, Hill fits, and the uncertainty inset.

    Parameters
    ----------
    summary : list[dict]
        Mean and standard-deviation accuracy rows.
    cfg : dict
        Resolved Fig. 4 configuration containing reference lines and fits.
    output_path : pathlib.Path
        PNG path for the generated figure.

    Returns
    -------
    None
        Saves and closes the generated figure.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    conditions = _condition_definitions(cfg)
    x_values = np.asarray(cfg["training_sizes"], dtype=float)
    figure, axis = plt.subplots(figsize=(11, 4.6))
    handles = []
    for condition in conditions:
        available_x_values = np.asarray(
            [
                x
                for x in x_values
                if any(
                    row["training_size"] == x
                    and row["model"] == condition["name"]
                    and row["split"] == "test"
                    for row in summary
                )
            ]
        )
        if available_x_values.size == 0:
            continue
        condition_x_values = available_x_values
        if condition.get("qpu", False):
            condition_x_values = condition_x_values[
                condition_x_values <= cfg["qpu_max_training_size"]
            ]
        means = np.asarray(
            [
                _summary_value(summary, x, condition["name"], "test", "mean_accuracy")
                for x in condition_x_values
            ]
        )
        stds = np.asarray(
            [
                _summary_value(summary, x, condition["name"], "test", "std_accuracy")
                for x in condition_x_values
            ]
        )
        marker = {
            "MLR": "s",
            "QORC (ideal)": "v",
            "QORC (noisy)": "^",
            "QORC (QPU)": "o",
        }[condition["name"]]
        plot_line = axis.errorbar(
            condition_x_values,
            means,
            yerr=stds,
            color=condition["color"],
            marker=marker,
            linestyle="--",
            linewidth=1.5,
            capsize=2,
            label=condition["name"],
        )
        handles.append(plot_line)
        if condition["name"] in cfg["hill_fits"]:
            fit_x = np.geomspace(
                max(1, condition_x_values.min()), condition_x_values.max(), 300
            )
            axis.plot(
                fit_x,
                _hill_function(fit_x, cfg["hill_fits"][condition["name"]]),
                color=condition["color"],
                linewidth=1,
            )

    axis.axhline(
        cfg["mlr_best_train_accuracy"], color="grey", linestyle="-", linewidth=1.5
    )
    axis.axhline(
        cfg["mlr_best_test_accuracy"], color="grey", linestyle="--", linewidth=1.5
    )
    axis.set_xscale("log")
    axis.set_xlabel("# Training Images")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(*cfg["y_limits"])
    axis.grid(alpha=0.3, linestyle="--")
    axis.legend(handles=handles, loc="upper left")

    inset = inset_axes(axis, width="33%", height="38%", loc="lower right", borderpad=2)
    inset_x = x_values[x_values <= cfg["inset_max_training_size"]]
    for condition in conditions:
        condition_inset_x = inset_x
        condition_inset_x = np.asarray(
            [
                x
                for x in condition_inset_x
                if any(
                    row["training_size"] == x
                    and row["model"] == condition["name"]
                    and row["split"] == "test"
                    for row in summary
                )
            ]
        )
        if condition_inset_x.size == 0:
            continue
        if condition.get("qpu", False):
            condition_inset_x = condition_inset_x[
                condition_inset_x <= cfg["qpu_max_training_size"]
            ]
        means = np.asarray(
            [
                _summary_value(summary, x, condition["name"], "test", "mean_accuracy")
                for x in condition_inset_x
            ]
        )
        stds = np.asarray(
            [
                _summary_value(summary, x, condition["name"], "test", "std_accuracy")
                for x in condition_inset_x
            ]
        )
        inset.fill_between(
            condition_inset_x,
            means - stds,
            means + stds,
            color=condition["color"],
            alpha=0.18,
        )
        inset.plot(
            condition_inset_x,
            means,
            color=condition["color"],
            marker={
                "MLR": "s",
                "QORC (ideal)": "v",
                "QORC (noisy)": "^",
                "QORC (QPU)": "o",
            }[condition["name"]],
            linestyle="--",
        )
    inset.set_xlim(cfg["inset_limits"][0], cfg["inset_limits"][1])
    inset.set_ylim(cfg["inset_limits"][2], cfg["inset_limits"][3])
    inset.grid(alpha=0.25, linestyle="--")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _summary_value(summary, training_size, model, split, field):
    row = next(
        row
        for row in summary
        if row["training_size"] == training_size
        and row["model"] == model
        and row["split"] == split
    )
    return row[field]
