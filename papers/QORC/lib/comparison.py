from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lib.lib_datasets import get_qorc_dataset, split_fold_numpy
from lib.lib_qorc_encoding_and_linear_training import (
    qorc_encoding_and_linear_training,
)

QORC_ORANGE = "#f28e2b"
LINEAR_DARK_BLUE = "#163b65"


def run_qorc_lsvc_comparison(cfg, run_dir: Path, logger: logging.Logger) -> None:
    """Run the MNIST QORC versus raw-pixel L-SVC comparison.

    Parameters
    ----------
    cfg : dict
        Resolved comparison configuration.
    run_dir : pathlib.Path
        Timestamped output directory for the experiment artifacts.
    logger : logging.Logger
        Logger receiving experiment progress messages.

    Returns
    -------
    None
        Writes comparison metrics and a PNG plot to ``run_dir``.

    Raises
    ------
    ValueError
        If the comparison is configured with a dataset other than MNIST.
    """
    dataset_name = cfg.get("dataset_name", "mnist").lower()
    if dataset_name != "mnist":
        raise ValueError("The QORC/L-SVC comparison currently supports MNIST only.")

    seed = cfg["seed"]
    dataset_sampling = cfg.get("dataset_sampling", "full")
    dataset_sample_count = cfg.get("dataset_sample_count")
    dataset_samples_per_class = cfg.get("dataset_samples_per_class")
    train_data, train_labels, test_data, test_labels = get_qorc_dataset(
        dataset_name,
        sampling=dataset_sampling,
        sample_count=dataset_sample_count,
        samples_per_class=dataset_samples_per_class,
        seed=seed,
    )
    train_data = train_data.reshape(train_data.shape[0], -1).astype(np.float32) / 255.0
    test_data = test_data.reshape(test_data.shape[0], -1).astype(np.float32) / 255.0
    _val_labels, _val_data, train_labels, train_data = split_fold_numpy(
        train_labels,
        train_data,
        cfg["n_fold"],
        cfg["fold_index"],
        split_seed=seed,
    )

    logger.info("Training QORC for %s epochs", cfg["n_epochs"])
    qorc_result = qorc_encoding_and_linear_training(
        n_photons=cfg["n_photons"],
        n_modes=cfg["n_modes"],
        seed=seed,
        dataset_name=dataset_name,
        dataset_sampling=dataset_sampling,
        dataset_sample_count=dataset_sample_count,
        dataset_samples_per_class=dataset_samples_per_class,
        noise_enabled=cfg.get("noise_enabled", False),
        noise_indistinguishability=cfg.get("noise_indistinguishability", 1.0),
        noise_g2=cfg.get("noise_g2", 0.0),
        noise_g2_distinguishable=cfg.get("noise_g2_distinguishable", True),
        fold_index=cfg["fold_index"],
        n_fold=cfg["n_fold"],
        dataset_truncate=cfg.get("dataset_truncate", 0),
        n_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        reduce_lr_patience=cfg["reduce_lr_patience"],
        reduce_lr_factor=cfg["reduce_lr_factor"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        f_out_weights="qorc_weights.pth",
        save_weights=cfg["save_weights"],
        b_no_bunching=cfg["b_no_bunching"],
        b_use_tensorboard=cfg["b_use_tensorboard"],
        device_name=cfg["device"],
        qpu_device_name=cfg.get("qpu_device", "none"),
        qpu_device_nsample=cfg.get("qpu_device_nsample", 10000),
        run_dir=run_dir,
        logger=logger,
        return_history=True,
    )

    logger.info("Training raw-pixel MLR baseline")
    linear_history = train_linear_baseline(
        train_data,
        train_labels,
        test_data,
        test_labels,
        n_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg.get("linear_learning_rate", cfg["learning_rate"]),
        seed=seed,
    )

    epochs = len(qorc_result["train_accuracy"])
    metrics = {
        "epochs": list(range(1, epochs + 1)),
        "qorc_train_accuracy": qorc_result["train_accuracy"],
        "qorc_test_accuracy": qorc_result["test_accuracy"],
        "qorc_train_loss": qorc_result["train_loss"],
        "qorc_test_loss": qorc_result["test_loss"],
        "linear_train_accuracy": linear_history["train_accuracy"],
        "linear_test_accuracy": linear_history["test_accuracy"],
        "linear_train_loss": linear_history["train_loss"],
        "linear_test_loss": linear_history["test_loss"],
    }
    (run_dir / "comparison_QORC_LSVC_mnist.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    plot_qorc_lsvc_comparison(metrics, run_dir / "comparison_QORC_LSVC_mnist.png")


def train_linear_baseline(
    train_data,
    train_labels,
    test_data,
    test_labels,
    n_epochs,
    batch_size,
    learning_rate,
    seed,
    n_classes=10,
):
    """Train a raw-pixel MLR classifier.

    Parameters
    ----------
    train_data : numpy.ndarray
        Flattened, normalized training images.
    train_labels : numpy.ndarray
        Training class labels.
    test_data : numpy.ndarray
        Flattened, normalized test images.
    test_labels : numpy.ndarray
        Test class labels.
    n_epochs : int
        Number of optimization epochs.
    batch_size : int
        Training minibatch size.
    learning_rate : float
        Adagrad learning rate.
    seed : int
        Random seed for model initialization and minibatch ordering.
    n_classes : int
        Number of output classes. Default value is 10.

    Returns
    -------
    dict[str, list[float]]
        Epoch-wise train/test accuracy and cross-entropy loss.
    """
    torch.manual_seed(seed)
    model = nn.Linear(train_data.shape[1], n_classes)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_inputs = torch.from_numpy(train_data)
    train_targets = torch.from_numpy(train_labels).long()
    test_inputs = torch.from_numpy(test_data)
    test_targets = torch.from_numpy(test_labels).long()
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    history = {
        "train_accuracy": [],
        "test_accuracy": [],
        "train_loss": [],
        "test_loss": [],
        "test_predictions": [],
    }
    for _ in range(n_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(train_inputs)
            test_logits = model(test_inputs)
            history["train_loss"].append(float(criterion(train_logits, train_targets)))
            history["test_loss"].append(float(criterion(test_logits, test_targets)))
            history["train_accuracy"].append(
                float((train_logits.argmax(dim=1) == train_targets).float().mean())
            )
            history["test_accuracy"].append(
                float((test_logits.argmax(dim=1) == test_targets).float().mean())
            )
            history["test_predictions"] = test_logits.argmax(dim=1).tolist()
    return history


def plot_qorc_lsvc_comparison(metrics: dict, output_path: Path) -> None:
    """Plot QORC and MLR training/test accuracy and loss curves.

    Parameters
    ----------
    metrics : dict
        Metrics containing epoch-wise QORC values and scalar MLR values.
    output_path : pathlib.Path
        PNG path for the generated figure.

    Returns
    -------
    None
        Saves and closes the figure.
    """
    import matplotlib.pyplot as plt

    epochs = np.asarray(metrics["epochs"])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    accuracy_axis, loss_axis = axes
    accuracy_axis.plot(
        epochs,
        metrics["qorc_train_accuracy"],
        color=QORC_ORANGE,
        label="QORC train",
        linewidth=2,
    )
    accuracy_axis.plot(
        epochs,
        metrics["qorc_test_accuracy"],
        color=QORC_ORANGE,
        linestyle="--",
        label="QORC test",
        linewidth=2,
    )
    accuracy_axis.plot(
        epochs,
        metrics["linear_train_accuracy"],
        color=LINEAR_DARK_BLUE,
        label="MLR train",
        linewidth=2,
    )
    accuracy_axis.plot(
        epochs,
        metrics["linear_test_accuracy"],
        color=LINEAR_DARK_BLUE,
        linestyle="--",
        label="MLR test",
        linewidth=2,
    )
    accuracy_axis.set_xlabel("QORC readout epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0.0, 1.0)
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend()
    loss_axis.plot(
        epochs,
        metrics["qorc_train_loss"],
        color=QORC_ORANGE,
        label="QORC train",
        linewidth=2,
    )
    loss_axis.plot(
        epochs,
        metrics["qorc_test_loss"],
        color=QORC_ORANGE,
        linestyle="--",
        label="QORC test",
        linewidth=2,
    )
    loss_axis.plot(
        epochs,
        metrics["linear_train_loss"],
        color=LINEAR_DARK_BLUE,
        label="MLR train",
        linewidth=2,
    )
    loss_axis.plot(
        epochs,
        metrics["linear_test_loss"],
        color=LINEAR_DARK_BLUE,
        linestyle="--",
        label="MLR test",
        linewidth=2,
    )
    loss_axis.set_xlabel("QORC readout epoch")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.set_title("Loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()
    figure.suptitle("QORC versus MLR on MNIST")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
