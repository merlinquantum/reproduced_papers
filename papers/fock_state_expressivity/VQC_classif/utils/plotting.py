from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_dataset_samples(
    x: torch.Tensor, y: torch.Tensor, title: str, save_path: Path
) -> None:
    _ensure_dir(save_path)
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    plt.figure(figsize=(5, 4))
    plt.scatter(x_np[:, 0], x_np[:, 1], c=y_np, cmap="bwr", edgecolor="k")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title, fontsize=14)
    legend_elements = [
        Patch(facecolor="blue", edgecolor="k", label="Label = 0"),
        Patch(facecolor="red", edgecolor="k", label="Label = 1"),
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_decision_boundary(entry: dict, save_path: Path, resolution: int = 100) -> None:
    _ensure_dir(save_path)
    model = entry["model"]
    model_type = entry["model_type"]
    activation = entry.get("activation", "none")
    dataset_name = entry["dataset"].title()
    best_acc = entry.get("best_acc", 0.0)
    initial_state = entry.get("initial_state")

    x_train = entry["x_train"].cpu().numpy()
    y_train = entry["y_train"].cpu().numpy()
    x_test = entry["x_test"].cpu().numpy()
    y_test = entry["y_test"].cpu().numpy()

    combined = np.vstack([x_train, x_test])
    x_min, x_max = combined[:, 0].min() - 1, combined[:, 0].max() + 1
    y_min, y_max = combined[:, 1].min() - 1, combined[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if model_type.startswith("svm"):
        preds = model.predict(grid_points).astype(float)
    else:
        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(grid_points, dtype=torch.float32))
        if activation == "softmax":
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        else:
            preds = torch.round(outputs).squeeze().cpu().numpy()

    preds = (preds > 0.5).astype(int)
    class_map = preds.reshape(xx.shape).astype(float)

    plt.figure(figsize=(6, 5))
    region_cmap = ListedColormap(["#a6c8ff", "#ffb3b3"])
    plt.contourf(
        xx, yy, class_map, levels=[-0.5, 0.5, 1.5], cmap=region_cmap, alpha=0.9
    )
    plt.contour(xx, yy, class_map, levels=[0.5], colors="k", linewidths=0.8)

    point_cmap = ListedColormap(["#1f77b4", "#d62728"])
    plt.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train,
        cmap=point_cmap,
        vmin=0,
        vmax=1,
        marker="o",
        edgecolor="k",
        label="Train",
    )
    plt.scatter(
        x_test[:, 0],
        x_test[:, 1],
        c=y_test,
        cmap=point_cmap,
        vmin=0,
        vmax=1,
        marker="x",
        label="Test",
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    title_model = model_type.upper()
    if model_type == "vqc" and initial_state is not None:
        title_model = f"{title_model} {initial_state}"
    plt.title(f"{title_model} Decision Boundary on {dataset_name}\nAcc: {best_acc:.3f}")
    legend_elements = [
        Patch(facecolor="blue", label="Label 0"),
        Patch(facecolor="red", label="Label 1"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            label="Train",
            markerfacecolor="gray",
            markersize=6,
            linestyle="",
        ),
        Line2D(
            [0], [0], marker="x", color="gray", label="Test", markersize=6, linestyle=""
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_decision_boundary_from_payload(
    payload: dict, save_path: Path, resolution: int = 100
) -> None:
    _ensure_dir(save_path)
    x_train = np.array(payload["x_train"])
    y_train = np.array(payload["y_train"])
    x_test = np.array(payload["x_test"])
    y_test = np.array(payload["y_test"])
    xx = np.array(payload["grid_x"])
    yy = np.array(payload["grid_y"])
    class_map = np.array(payload["class_map"])

    model_type = payload.get("model_type", "vqc")
    dataset_name = payload.get("dataset", "").title()
    best_acc = payload.get("best_acc", 0.0)
    initial_state = payload.get("initial_state")

    plt.figure(figsize=(6, 5))
    region_cmap = ListedColormap(["#a6c8ff", "#ffb3b3"])
    plt.contourf(
        xx, yy, class_map, levels=[-0.5, 0.5, 1.5], cmap=region_cmap, alpha=0.9
    )
    plt.contour(xx, yy, class_map, levels=[0.5], colors="k", linewidths=0.8)

    point_cmap = ListedColormap(["#1f77b4", "#d62728"])
    plt.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train,
        cmap=point_cmap,
        vmin=0,
        vmax=1,
        marker="o",
        edgecolor="k",
        label="Train",
    )
    plt.scatter(
        x_test[:, 0],
        x_test[:, 1],
        c=y_test,
        cmap=point_cmap,
        vmin=0,
        vmax=1,
        marker="x",
        label="Test",
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    title_model = model_type.upper()
    if model_type == "vqc" and initial_state is not None:
        title_model = f"{title_model} {initial_state}"
    plt.title(f"{title_model} Decision Boundary on {dataset_name}\nAcc: {best_acc:.3f}")
    legend_elements = [
        Patch(facecolor="blue", label="Label 0"),
        Patch(facecolor="red", label="Label 1"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            label="Train",
            markerfacecolor="gray",
            markersize=6,
            linestyle="",
        ),
        Line2D(
            [0], [0], marker="x", color="gray", label="Test", markersize=6, linestyle=""
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_training_metrics(
    results: dict[str, dict],
    save_path: Path,
    dataset_order: Iterable[str] | None = None,
) -> None:
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if dataset_order is None:
        datasets = list(results.keys())
    else:
        datasets = [name for name in dataset_order if name in results]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    metrics = [
        ("losses", "Training Loss"),
        ("train_accuracies", "Training Accuracy"),
        ("test_accuracies", "Test Accuracy"),
    ]

    for idx, dataset in enumerate(datasets):
        runs = results[dataset]["runs"]
        color = colors[idx % len(colors)]
        epochs = len(runs[0]["losses"])
        for metric_idx, (key, title) in enumerate(metrics):
            metric_runs = [run[key] for run in runs]
            mean_vals = [
                np.mean([run[i] for run in metric_runs]) for i in range(epochs)
            ]
            min_vals = [np.min([run[i] for run in metric_runs]) for i in range(epochs)]
            max_vals = [np.max([run[i] for run in metric_runs]) for i in range(epochs)]

            ax = axes[metric_idx]
            ax.plot(mean_vals, label=dataset.title(), color=color, linewidth=2)
            ax.fill_between(range(epochs), min_vals, max_vals, color=color, alpha=0.2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle="--", alpha=0.6)

    for ax in axes:
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
