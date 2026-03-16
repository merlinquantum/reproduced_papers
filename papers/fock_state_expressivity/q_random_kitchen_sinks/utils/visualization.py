from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lib.approx_kernel import classical_features, transform_inputs
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.svm import SVC

CMAP_BACKGROUND = ListedColormap(["#AAAAFF", "#FFAAAA"])


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_dataset(x_train, x_test, y_train, y_test, save_path: Path) -> None:
    _ensure_dir(save_path)
    plt.figure(figsize=(6, 6))
    plt.scatter(
        x_train[y_train == 0][:, 0],
        x_train[y_train == 0][:, 1],
        color="red",
        marker="o",
        label="Class 0 - Train",
        s=40,
    )
    plt.scatter(
        x_test[y_test == 0][:, 0],
        x_test[y_test == 0][:, 1],
        color="red",
        marker="x",
        label="Class 0 - Test",
        s=40,
    )
    plt.scatter(
        x_train[y_train == 1][:, 0],
        x_train[y_train == 1][:, 1],
        color="blue",
        marker="o",
        label="Class 1 - Train",
        s=40,
    )
    plt.scatter(
        x_test[y_test == 1][:, 0],
        x_test[y_test == 1][:, 1],
        color="blue",
        marker="x",
        label="Class 1 - Test",
        s=40,
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_dataset_from_payload(payload: dict, save_path: Path) -> None:
    plot_dataset(
        np.array(payload["x_train"]),
        np.array(payload["x_test"]),
        np.array(payload["y_train"]),
        np.array(payload["y_test"]),
        save_path,
    )


def plot_accuracy_heatmap(
    results: list[dict],
    r_values: Iterable[int],
    gamma_values: Iterable[int],
    method: str,
    save_path: Path,
) -> None:
    _ensure_dir(save_path)
    heatmap = np.zeros((len(r_values), len(gamma_values)))
    for i, r in enumerate(r_values):
        for j, gamma in enumerate(gamma_values):
            subset = [
                entry
                for entry in results
                if entry["method"] == method
                and entry["r"] == r
                and entry["gamma"] == gamma
            ]
            if subset:
                heatmap[i, j] = np.mean([entry["accuracy"] for entry in subset])

    plt.figure(figsize=(10, 4))
    plt.imshow(heatmap, cmap="viridis", aspect="auto", origin="lower")
    plt.colorbar(label="Accuracy")
    plt.xticks(range(len(gamma_values)), gamma_values)
    plt.yticks(range(len(r_values)), r_values)
    plt.xlabel("Gamma")
    plt.ylabel("R")
    plt.title(f"{method.capitalize()} random kitchen sinks accuracy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_kernel_heatmap(
    kernel: np.ndarray, y_train: np.ndarray, title: str, save_path: Path
) -> None:
    _ensure_dir(save_path)
    sorted_idx = np.argsort(y_train)
    k_sorted = kernel[sorted_idx][:, sorted_idx]
    split = np.sum(y_train[sorted_idx] == y_train[sorted_idx][0])
    plt.figure(figsize=(6, 5))
    ax = plt.imshow(k_sorted, cmap="viridis")
    plt.axhline(split, color="white")
    plt.axvline(split, color="white")
    plt.title(title)
    plt.colorbar(ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_kernel_heatmap_from_payload(payload: dict, save_path: Path) -> None:
    save_kernel_heatmap(
        np.array(payload["kernel"]),
        np.array(payload["y_train"]),
        payload.get("title", "Kernel heatmap"),
        save_path,
    )


def _build_mesh(
    dataset, resolution: float = 0.02
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test = dataset[0], dataset[1]
    x_all = np.vstack((x_train, x_test))
    x_min, x_max = x_all[:, 0].min() - 0.2, x_all[:, 0].max() + 0.2
    y_min, y_max = x_all[:, 1].min() - 0.2, x_all[:, 1].max() + 0.2
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def _decision_map(entry: dict, dataset, xx: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x_train, _, y_train, _ = dataset
    w = entry["decision_data"]["w"]
    b = entry["decision_data"]["b"]
    r = entry["decision_data"].get("r", 1)
    gamma = entry["decision_data"]["gamma"]

    if entry["method"] == "classical":
        grid_proj = transform_inputs(grid, w, b, r, gamma)
        train_proj = transform_inputs(x_train, w, b, r, gamma)
        z_grid = classical_features(grid_proj)
        z_train = classical_features(train_proj)
    else:
        model = entry["decision_data"]["model"]
        model.eval()
        scale = entry["decision_data"]["pre_scale"]
        z_scale = entry["decision_data"]["z_scale"]
        grid_proj = transform_inputs(grid, w, b, r, gamma)
        train_proj = transform_inputs(x_train, w, b, r, gamma)
        with torch.no_grad():
            grid_input = (
                torch.tensor(grid_proj, dtype=torch.float32).view(-1, 1) * scale
            )
            train_input = (
                torch.tensor(train_proj, dtype=torch.float32).view(-1, 1) * scale
            )
            z_grid = model(grid_input).view(len(grid_proj), -1).cpu().numpy() * z_scale
            z_train = (
                model(train_input).view(len(train_proj), -1).cpu().numpy() * z_scale
            )

    k_grid = z_grid @ z_train.T
    svm = SVC(kernel="precomputed", C=entry.get("C", 5.0))
    svm.fit(entry["kernel_train"], y_train)
    preds = svm.decision_function(k_grid).reshape(xx.shape)
    return (preds > 0).astype(int)


def _render_panel(
    ax,
    class_map: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    dataset,
    train_kwargs: dict | None = None,
    test_kwargs: dict | None = None,
) -> None:
    x_train, x_test, y_train, y_test = dataset
    base_train = {"marker": "o", "s": 15, "edgecolor": "k", "label": "Train"}
    base_test = {"marker": "x", "s": 20, "label": "Test"}
    if train_kwargs:
        base_train.update(train_kwargs)
    if test_kwargs:
        base_test.update(test_kwargs)

    ax.contourf(xx, yy, class_map, cmap=CMAP_BACKGROUND, alpha=0.6)
    ax.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train,
        cmap="bwr",
        vmin=0,
        vmax=1,
        **base_train,
    )
    ax.scatter(
        x_test[:, 0],
        x_test[:, 1],
        c=y_test,
        cmap="bwr",
        vmin=0,
        vmax=1,
        **base_test,
    )


def plot_decision_boundary(entry: dict, dataset, save_path: Path) -> None:
    _ensure_dir(save_path)
    xx, yy, grid = _build_mesh(dataset)
    class_map = _decision_map(entry, dataset, xx, grid)

    fig, ax = plt.subplots(figsize=(7, 5))
    _render_panel(ax, class_map, xx, yy, dataset)
    r = entry["decision_data"].get("r", 1)
    gamma = entry["decision_data"]["gamma"]
    ax.set_title(f"{entry['method'].capitalize()} R={r}, γ={gamma}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    accuracy = entry.get("accuracy")
    if accuracy is not None:
        ax.text(
            0.02,
            0.95,
            f"Acc: {accuracy:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="#222222",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

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
    ax.legend(handles=legend_elements, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def build_decision_payload(
    entries: list[dict],
    dataset,
    r_values: Iterable[int],
    gamma_values: Iterable[int],
    method: str,
    *,
    resolution: float = 0.02,
) -> dict:
    xx, yy, grid = _build_mesh(dataset, resolution)
    lookup: dict[tuple[int, int], dict] = {}
    for entry in entries:
        if entry["method"] != method:
            continue
        key = (entry["r"], entry["gamma"])
        if key not in lookup or entry["accuracy"] > lookup[key]["accuracy"]:
            lookup[key] = entry

    payload_entries: list[dict] = []
    for r in r_values:
        for gamma in gamma_values:
            entry = lookup.get((r, gamma))
            if entry is None:
                continue
            class_map = _decision_map(entry, dataset, xx, grid)
            payload_entries.append(
                {
                    "r": r,
                    "gamma": gamma,
                    "accuracy": entry.get("accuracy"),
                    "class_map": class_map.tolist(),
                }
            )

    x_train, x_test, y_train, y_test = dataset
    return {
        "method": method,
        "r_values": list(r_values),
        "gamma_values": list(gamma_values),
        "grid_x": xx.tolist(),
        "grid_y": yy.tolist(),
        "entries": payload_entries,
        "dataset": {
            "x_train": x_train.tolist(),
            "x_test": x_test.tolist(),
            "y_train": y_train.tolist(),
            "y_test": y_test.tolist(),
        },
    }


def plot_combined_decisions(
    entries: list[dict],
    dataset,
    r_values: Iterable[int],
    gamma_values: Iterable[int],
    method: str,
    save_path: Path,
) -> None:
    _ensure_dir(save_path)
    xx, yy, grid = _build_mesh(dataset)
    lookup: dict[tuple[int, int], dict] = {}
    for entry in entries:
        if entry["method"] != method:
            continue
        key = (entry["r"], entry["gamma"])
        if key not in lookup or entry["accuracy"] > lookup[key]["accuracy"]:
            lookup[key] = entry

    rows = len(r_values)
    cols = len(gamma_values)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 1.2, rows * 1.2), squeeze=False
    )
    for i, r in enumerate(r_values):
        for j, gamma in enumerate(gamma_values):
            ax = axes[i][j]
            entry = lookup.get((r, gamma))
            if entry is None:
                ax.axis("off")
                continue
            class_map = _decision_map(entry, dataset, xx, grid)
            _render_panel(
                ax,
                class_map,
                xx,
                yy,
                dataset,
                train_kwargs={"marker": "o", "s": 8, "edgecolor": "none"},
                test_kwargs={"marker": "x", "s": 9, "linewidths": 0.8},
            )
            if entry.get("accuracy") is not None:
                ax.text(
                    0.02,
                    0.9,
                    f"Acc {entry['accuracy']:.3f}",
                    transform=ax.transAxes,
                    fontsize=6.5,
                    color="#111",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"γ={gamma}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"R={r}", fontsize=8)

    title = (
        "Quantum decision boundaries and test accuracies"
        if method == "quantum"
        else "Classical decision boundaries and test accuracies"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_combined_decisions_from_payload(payload: dict, save_path: Path) -> None:
    _ensure_dir(save_path)
    dataset_payload = payload["dataset"]
    dataset = (
        np.array(dataset_payload["x_train"]),
        np.array(dataset_payload["x_test"]),
        np.array(dataset_payload["y_train"]),
        np.array(dataset_payload["y_test"]),
    )

    r_values = payload["r_values"]
    gamma_values = payload["gamma_values"]
    xx = np.array(payload["grid_x"])
    yy = np.array(payload["grid_y"])
    lookup = {(entry["r"], entry["gamma"]): entry for entry in payload["entries"]}

    rows = len(r_values)
    cols = len(gamma_values)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 1.2, rows * 1.2), squeeze=False
    )
    for i, r in enumerate(r_values):
        for j, gamma in enumerate(gamma_values):
            ax = axes[i][j]
            entry = lookup.get((r, gamma))
            if entry is None:
                ax.axis("off")
                continue
            class_map = np.array(entry["class_map"])
            _render_panel(
                ax,
                class_map,
                xx,
                yy,
                dataset,
                train_kwargs={"marker": "o", "s": 8, "edgecolor": "none"},
                test_kwargs={"marker": "x", "s": 9, "linewidths": 0.8},
            )
            accuracy = entry.get("accuracy")
            if accuracy is not None:
                ax.text(
                    0.02,
                    0.9,
                    f"Acc {accuracy:.3f}",
                    transform=ax.transAxes,
                    fontsize=6.5,
                    color="#111",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"γ={gamma}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"R={r}", fontsize=8)

    title = (
        "Quantum decision boundaries and test accuracies"
        if payload.get("method") == "quantum"
        else "Classical decision boundaries and test accuracies"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_single_decision_from_payload(payload: dict, save_path: Path) -> None:
    _ensure_dir(save_path)
    dataset_payload = payload["dataset"]
    dataset = (
        np.array(dataset_payload["x_train"]),
        np.array(dataset_payload["x_test"]),
        np.array(dataset_payload["y_train"]),
        np.array(dataset_payload["y_test"]),
    )
    xx = np.array(payload["grid_x"])
    yy = np.array(payload["grid_y"])
    entries = payload.get("entries", [])
    if not entries:
        raise ValueError("No decision entries available in payload.")
    entry = entries[0]
    class_map = np.array(entry["class_map"])

    fig, ax = plt.subplots(figsize=(7, 5))
    _render_panel(ax, class_map, xx, yy, dataset)
    r = entry.get("r", 1)
    gamma = entry.get("gamma", 1)
    method = payload.get("method", "quantum")
    ax.set_title(f"{method.capitalize()} R={r}, γ={gamma}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    accuracy = entry.get("accuracy")
    if accuracy is not None:
        ax.text(
            0.02,
            0.95,
            f"Acc: {accuracy:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="#222222",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

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
    ax.legend(handles=legend_elements, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
