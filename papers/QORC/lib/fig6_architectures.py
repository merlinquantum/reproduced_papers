from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lib.lib_datasets import get_qorc_dataset
from lib.lib_qorc_encoding_and_linear_training import (
    create_qorc_reservoir_classifier,
    initialize_reservoir_normalization_in_batches,
    make_reservoir_dataset_in_batches,
)

ARCHITECTURES = {
    "Linear": {"layers": 0},
    "ShallowF": {"layers": 1, "activation": "relu", "units": 16, "dropout": [0.0]},
    "Shallow": {"layers": 1, "activation": "relu", "units": 96, "dropout": [0.3]},
    "Deep": {"layers": 3, "units": [256, 256, 32], "dropout": [0.3, 0.4, 0.1]},
}


class Fig6Model(nn.Module):
    """Configurable Fig. 6 readout architecture."""

    def __init__(self, input_size, architecture):
        super().__init__()
        if architecture["layers"] == 0:
            self.network = nn.Linear(input_size, 10)
            return
        activation = (
            nn.ReLU() if architecture.get("activation", "relu") == "relu" else nn.ELU()
        )
        units = (
            architecture["units"]
            if isinstance(architecture["units"], list)
            else [architecture["units"]]
        )
        dropout = architecture["dropout"]
        layers = []
        previous_size = input_size
        for index, hidden_size in enumerate(units):
            layers.extend(
                [
                    nn.Linear(previous_size, hidden_size),
                    activation,
                    nn.Dropout(dropout[index]),
                ]
            )
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


def _train_model(
    train_inputs, train_targets, test_inputs, test_targets, model_config, seed, device
):
    torch.manual_seed(seed)
    model = Fig6Model(train_inputs.shape[1], model_config["architecture"]).to(device)
    optimizer_name = model_config["optimizer"].lower()
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adagrad": torch.optim.Adagrad,
        "rmsprop": torch.optim.RMSprop,
    }[optimizer_name]
    optimizer = optimizer_class(model.parameters(), lr=model_config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=model_config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(model_config["epochs"]):
        model.train()
        for inputs, targets in loader:
            optimizer.zero_grad()
            criterion(model(inputs.to(device)), targets.to(device)).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        train_predictions = model(train_inputs.to(device)).argmax(dim=1).cpu()
        test_predictions = model(test_inputs.to(device)).argmax(dim=1).cpu()
    return {
        "train_accuracy": float((train_predictions == train_targets).float().mean()),
        "test_accuracy": float((test_predictions == test_targets).float().mean()),
    }


def run_fig6_architectures(cfg: dict, run_dir: Path, logger: logging.Logger) -> None:
    """Run the Fig. 6 accelerated and non-accelerated architecture benchmark.

    Parameters
    ----------
    cfg : dict
        Resolved Fig. 6 experiment configuration.
    run_dir : pathlib.Path
        Timestamped output directory for experiment artifacts.
    logger : logging.Logger
        Logger receiving experiment progress messages.

    Returns
    -------
    None
        Writes run metrics, aggregate metrics, and the comparison figure.
    """
    train_data, train_labels, test_data, test_labels = get_qorc_dataset("mnist")
    train_data = train_data.reshape(len(train_data), -1).astype(np.float32) / 255.0
    test_data = test_data.reshape(len(test_data), -1).astype(np.float32) / 255.0
    train_targets = torch.as_tensor(train_labels, dtype=torch.long)
    test_targets = torch.as_tensor(test_labels, dtype=torch.long)
    device = torch.device(cfg["device"])
    rows = []

    for run_index in range(cfg["n_runs"]):
        run_seed = cfg["seed"] + run_index
        reservoir = create_qorc_reservoir_classifier(
            n_photons=cfg["n_photons"],
            n_components=cfg["n_modes"],
            seed=run_seed,
            device_name=cfg["device"],
            b_no_bunching=cfg["b_no_bunching"],
            input_features=train_data.shape[1],
            n_classes=10,
            cache=False,
        )
        reservoir.fit_reservoir(train_data)
        initialize_reservoir_normalization_in_batches(
            reservoir, train_data, cfg["feature_batch_size"]
        )
        qorc_train_dataset = make_reservoir_dataset_in_batches(
            reservoir, train_data, train_labels, cfg["feature_batch_size"]
        )
        qorc_test_dataset = make_reservoir_dataset_in_batches(
            reservoir, test_data, test_labels, cfg["feature_batch_size"]
        )
        qorc_train_inputs = qorc_train_dataset.tensors[0]
        qorc_test_inputs = qorc_test_dataset.tensors[0]
        logger.info(
            "Fig. 6 run %s/%s: reservoir features ready", run_index + 1, cfg["n_runs"]
        )

        for model_name, architecture in ARCHITECTURES.items():
            for accelerated in (False, True):
                variant = "accelerated" if accelerated else "baseline"
                model_config = cfg["models"][model_name][variant]
                train_inputs = (
                    qorc_train_inputs if accelerated else torch.as_tensor(train_data)
                )
                test_inputs = (
                    qorc_test_inputs if accelerated else torch.as_tensor(test_data)
                )
                result = _train_model(
                    train_inputs,
                    train_targets,
                    test_inputs,
                    test_targets,
                    {
                        "architecture": architecture,
                        "epochs": cfg["epochs"],
                        **model_config,
                    },
                    run_seed,
                    device,
                )
                rows.extend(
                    [
                        {
                            "run": run_index,
                            "seed": run_seed,
                            "model": model_name,
                            "accelerated": accelerated,
                            "split": "train",
                            "accuracy": result["train_accuracy"],
                        },
                        {
                            "run": run_index,
                            "seed": run_seed,
                            "model": model_name,
                            "accelerated": accelerated,
                            "split": "test",
                            "accuracy": result["test_accuracy"],
                        },
                    ]
                )
        del reservoir, qorc_train_dataset, qorc_test_dataset

    summary = _summarize(rows)
    (run_dir / "fig6_mnist_different_architectures.json").write_text(
        json.dumps(
            {
                "runs": rows,
                "summary": summary,
                "target_test_accuracies_percent": cfg["target_test_accuracies_percent"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (run_dir / "fig6_mnist_different_architectures.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["run", "seed", "model", "accelerated", "split", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)
    plot_fig6(summary, run_dir / "fig6_mnist_different_architectures.png")


def _summarize(rows):
    summary = []
    keys = sorted({(row["model"], row["accelerated"], row["split"]) for row in rows})
    for model, accelerated, split in keys:
        values = np.asarray(
            [
                row["accuracy"]
                for row in rows
                if (row["model"], row["accelerated"], row["split"])
                == (model, accelerated, split)
            ]
        )
        summary.append(
            {
                "model": model,
                "accelerated": accelerated,
                "split": split,
                "mean_accuracy": float(values.mean()),
                "std_accuracy": float(values.std()),
                "n_runs": int(values.size),
            }
        )
    return summary


def plot_fig6(summary, output_path: Path) -> None:
    """Plot Fig. 6 mean test accuracy with run-to-run standard deviations.

    Parameters
    ----------
    summary : list[dict]
        Aggregate accuracy rows.
    output_path : pathlib.Path
        PNG path for the generated figure.

    Returns
    -------
    None
        Saves and closes the generated figure.
    """
    import matplotlib.pyplot as plt

    models = list(ARCHITECTURES)
    x_values = np.arange(len(models))
    width = 0.35
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for offset, accelerated, color, label in (
        (-width / 2, False, "#163b65", "Classical"),
        (width / 2, True, "#f28e2b", "Quantum"),
    ):
        values = [
            next(
                row
                for row in summary
                if row["model"] == model
                and row["accelerated"] == accelerated
                and row["split"] == "test"
            )
            for model in models
        ]
        means = [row["mean_accuracy"] for row in values]
        bars = axis.bar(
            x_values + offset,
            means,
            width,
            yerr=[row["std_accuracy"] for row in values],
            capsize=3,
            color=color,
            label=label,
        )
        axis.bar_label(
            bars, labels=[f"{mean:.3f}" for mean in means], padding=3, fontsize=8
        )
    axis.set_xticks(x_values, models)
    axis.set_ylabel("Test Accuracy")
    axis.set_ylim(0.85, 1.0)
    axis.grid(axis="y", alpha=0.3, linestyle="--")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
