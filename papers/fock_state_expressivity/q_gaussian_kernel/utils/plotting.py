from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

NOTEBOOK_COLORS = {
    2: "#1f77b4",  # blue
    4: "#ff7f0e",  # orange
    6: "#2ca02c",  # green
    8: "#d62728",  # red
    10: "#9467bd",  # purple
}


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_gaussian_fits(grid, best_models: list[dict], save_path: Path) -> None:
    _ensure_dir(save_path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    unique_photons = sorted({entry["photons"] for entry in best_models})
    extra_needed = max(0, len(unique_photons) - len(NOTEBOOK_COLORS))
    extra_colors = (
        plt.cm.viridis(np.linspace(0, 1, extra_needed)) if extra_needed > 0 else []
    )
    color_map: dict[int, str] = {}
    extra_idx = 0
    for photons in unique_photons:
        if photons in NOTEBOOK_COLORS:
            color_map[photons] = NOTEBOOK_COLORS[photons]
        else:
            color_map[photons] = extra_colors[extra_idx]
            extra_idx += 1

    for idx, (sigma_label, sigma_value, target) in enumerate(
        zip(grid.sigma_labels, grid.sigma_values, grid.targets)
    ):
        axis = axes[idx // 2][idx % 2]
        axis.scatter(grid.x_on_pi, target, s=10, color="black", label="Target")
        axis.set_title(f"σ = {sigma_value:.2f}")
        axis.set_xlabel("x / π")
        axis.set_ylabel("Kernel value")

        sigma_models = [
            entry for entry in best_models if entry["sigma_label"] == sigma_label
        ]
        sorted_models = sorted(sigma_models, key=lambda e: e["photons"])
        for entry in sorted_models:
            color = color_map.get(entry["photons"], "#333333")
            axis.plot(
                grid.x_on_pi,
                entry["prediction"],
                color=color,
                linewidth=1.5,
                label=f"n={entry['photons']}",
            )

        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()

    fig.suptitle("Learned quantum kernels vs target Gaussians", fontsize=14)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_gaussian_fits_from_payload(payload: dict, save_path: Path) -> None:
    _ensure_dir(save_path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    sigma_labels = payload["sigma_labels"]
    sigma_values = payload["sigma_values"]
    targets = payload["targets"]
    x_on_pi = np.array(payload["x_on_pi"])

    predictions = payload.get("predictions", [])
    photon_values = sorted({entry["photons"] for entry in predictions})
    extra_needed = max(0, len(photon_values) - len(NOTEBOOK_COLORS))
    extra_colors = (
        plt.cm.viridis(np.linspace(0, 1, extra_needed)) if extra_needed > 0 else []
    )
    color_map: dict[int, str] = {}
    extra_idx = 0
    for photons in photon_values:
        if photons in NOTEBOOK_COLORS:
            color_map[photons] = NOTEBOOK_COLORS[photons]
        else:
            color_map[photons] = extra_colors[extra_idx]
            extra_idx += 1

    for idx, (sigma_label, sigma_value, target) in enumerate(
        zip(sigma_labels, sigma_values, targets)
    ):
        axis = axes[idx // 2][idx % 2]
        axis.scatter(x_on_pi, target, s=10, color="black", label="Target")
        axis.set_title(f"σ = {sigma_value:.2f}")
        axis.set_xlabel("x / π")
        axis.set_ylabel("Kernel value")

        sigma_models = [
            entry for entry in predictions if entry["sigma_label"] == sigma_label
        ]
        sorted_models = sorted(sigma_models, key=lambda e: e["photons"])
        for entry in sorted_models:
            if entry.get("prediction") is None:
                continue
            color = color_map.get(entry["photons"], "#333333")
            axis.plot(
                x_on_pi,
                entry["prediction"],
                color=color,
                linewidth=1.5,
                label=f"n={entry['photons']}",
            )

        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()

    fig.suptitle("Learned quantum kernels vs target Gaussians", fontsize=14)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_dataset_examples(
    datasets: dict[str, dict[str, torch.Tensor]], save_path: Path
) -> None:
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = ["Circular", "Moon", "Blob"]
    for ax, name, title in zip(axes, ["circular", "moon", "blob"], titles):
        train = datasets[name]["x_train"].numpy()
        train_y = datasets[name]["y_train"].numpy()
        test = datasets[name]["x_test"].numpy()
        test_y = datasets[name]["y_test"].numpy()

        ax.scatter(train[:, 0], train[:, 1], c=train_y, cmap="bwr", marker="o")
        ax.scatter(test[:, 0], test[:, 1], c=test_y, cmap="bwr", marker="x")
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        legend_elements = [
            Line2D([0], [0], marker="o", color="gray", linestyle="None", label="Train"),
            Line2D([0], [0], marker="x", color="gray", linestyle="None", label="Test"),
        ]
        ax.legend(handles=legend_elements)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_dataset_examples_from_payload(payload: dict, save_path: Path) -> None:
    datasets = {
        name: {
            "x_train": torch.tensor(data["x_train"]),
            "y_train": torch.tensor(data["y_train"]),
            "x_test": torch.tensor(data["x_test"]),
            "y_test": torch.tensor(data["y_test"]),
        }
        for name, data in payload.items()
    }
    plot_dataset_examples(datasets, save_path)


def plot_accuracy_bars(
    quantum_entries: list[dict],
    classical_entries: list[dict],
    save_path: Path,
) -> None:
    _ensure_dir(save_path)
    sigma_labels = sorted({entry["sigma_label"] for entry in quantum_entries})
    cols = 2
    rows = ceil(len(sigma_labels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    datasets = ["circular", "moon", "blob"]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]

    quantum_by_sigma = {}
    for entry in quantum_entries:
        quantum_by_sigma.setdefault(entry["sigma_label"], []).append(entry)
    classical_by_sigma = {entry["sigma_label"]: entry for entry in classical_entries}

    for idx, sigma_label in enumerate(sigma_labels):
        ax = axes[idx // cols][idx % cols]
        ax.set_title(sigma_label.replace("sigma=", "σ = "))
        sigma_quantum = sorted(
            quantum_by_sigma.get(sigma_label, []), key=lambda e: e["n_photons"]
        )
        if not sigma_quantum:
            continue
        photon_values = [entry["n_photons"] for entry in sigma_quantum]
        include_classical = sigma_label in classical_by_sigma
        positions = np.arange(len(photon_values) + (1 if include_classical else 0))
        bar_width = 0.2

        quantum_x: list[float] = []
        classical_x: list[float] = []

        for idx, dataset in enumerate(datasets):
            heights = [entry[f"{dataset}_acc"] for entry in sigma_quantum]
            if include_classical:
                heights.append(classical_by_sigma[sigma_label][f"{dataset}_acc"])
            x = positions + idx * bar_width
            ax.bar(x, heights, width=bar_width, color=colors[idx])
            for x_pos, h in zip(x, heights):
                ax.text(
                    x_pos,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                )
            quantum_x.extend(x[: len(photon_values)])
            if include_classical:
                classical_x.append(x[-1])

        labels = [f"{p} photons" for p in photon_values]
        if include_classical:
            labels.append("Classical")
        ax.set_xticks(positions + bar_width)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if include_classical and quantum_x and classical_x:
            boundary = (max(quantum_x) + min(classical_x)) / 2
            ax.axvline(
                boundary, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8
            )

    for idx in range(len(sigma_labels), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    handles = [
        Patch(facecolor=colors[i], label=datasets[i].capitalize())
        for i in range(len(datasets))
    ]
    fig.legend(
        handles=handles, loc="center left", bbox_to_anchor=(0.9, 0.5), title="Dataset"
    )
    fig.suptitle(
        "SVM accuracy using quantum vs classical Gaussian kernels", fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
