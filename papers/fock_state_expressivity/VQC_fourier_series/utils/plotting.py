from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _ensure_parent(path: Path) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)


def plot_training_curves(
    results: dict[str, dict], save_path: Path | None = None, show: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in results.items():
        color = data.get("color", None)
        losses_runs: list[list[float]] = [run["losses"] for run in data["runs"]]
        epochs = len(losses_runs[0])
        mean_losses = [
            sum(run[i] for run in losses_runs) / len(losses_runs) for i in range(epochs)
        ]
        min_losses = [min(run[i] for run in losses_runs) for i in range(epochs)]
        max_losses = [max(run[i] for run in losses_runs) for i in range(epochs)]

        ax.plot(mean_losses, label=label, color=color, linewidth=2)
        ax.fill_between(range(epochs), min_losses, max_losses, color=color, alpha=0.2)

    ax.set_title("Training Loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        _ensure_parent(Path(save_path))
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_learned_functions(
    best_models: list[dict],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x_numpy = inputs.squeeze().cpu().numpy()
    target_numpy = targets.cpu().numpy()

    ax.scatter(x_numpy, target_numpy, label="Target g(x)", s=15, alpha=0.6)

    for entry in best_models:
        model = entry["model"]
        color = entry.get("color")
        label = entry["label"]
        model.eval()
        with torch.no_grad():
            preds = model(inputs).view(-1).cpu().numpy()
        ax.plot(x_numpy, preds, label=label, linewidth=2, color=color)

    ax.set_title("Learned Functions vs Target Fourier Series")
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        _ensure_parent(Path(save_path))
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_learned_functions_from_predictions(
    payload: dict, save_path: Path | None = None, show: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x_numpy = payload["x"]
    target_numpy = payload["target"]

    ax.scatter(x_numpy, target_numpy, label="Target g(x)", s=15, alpha=0.6)

    for entry in payload.get("entries", []):
        label = entry.get("label", "VQC")
        color = entry.get("color")
        preds = entry.get("predictions", [])
        ax.plot(x_numpy, preds, label=label, linewidth=2, color=color)

    ax.set_title("Learned Functions vs Target Fourier Series")
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        _ensure_parent(Path(save_path))
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
