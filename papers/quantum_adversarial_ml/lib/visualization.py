"""
Visualization Utilities
=======================

Plotting functions for adversarial learning results,
matching figures from the Lu et al. (2020) paper.
"""

import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def plot_training_curves(
    history: dict[str, list[float]],
    title: str = "Training Progress",
    save_path: Optional[str] = None,
):
    """Plot training loss and accuracy curves.

    Reproduces Figure 4/5 from the paper.

    Args:
        history: Dictionary with train_loss, train_acc, test_loss, test_acc
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    if "test_loss" in history:
        ax1.plot(epochs, history["test_loss"], "r-", label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "b-", label="Train")
    if "test_acc" in history:
        ax2.plot(epochs, history["test_acc"], "r-", label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_adversarial_examples(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    clean_labels: torch.Tensor,
    adv_predictions: torch.Tensor,
    clean_predictions: torch.Tensor,
    n_examples: int = 4,
    image_size: Optional[tuple[int, int]] = None,
    class_names: Optional[list[str]] = None,
    title: str = "Adversarial Examples",
    save_path: Optional[str] = None,
):
    """Plot clean images and their adversarial counterparts.

    Reproduces Figure 6/9 from the paper.

    Args:
        clean_images: Clean images tensor
        adv_images: Adversarial images tensor
        clean_labels: True labels
        adv_predictions: Predictions on adversarial images
        clean_predictions: Predictions on clean images
        n_examples: Number of examples to show
        image_size: Image dimensions for reshaping (inferred if None)
        class_names: Optional class name mapping
        title: Plot title
        save_path: Path to save figure
    """
    # Infer image size from data if not provided
    if image_size is None:
        total_size = clean_images[0].numel()
        side = int(np.sqrt(total_size))
        if side * side == total_size:
            image_size = (side, side)
        else:
            # Not a square image, try to display as 1D or find factors
            image_size = (1, total_size)

    fig, axes = plt.subplots(2, n_examples, figsize=(3 * n_examples, 6))

    for i in range(n_examples):
        if i >= len(clean_images):
            break

        # Clean image
        clean_data = clean_images[i].cpu().numpy().flatten()
        try:
            clean_img = clean_data.reshape(image_size)
        except ValueError:
            # If reshape fails, just display as 1D
            clean_img = clean_data.reshape(1, -1)

        axes[0, i].imshow(clean_img, cmap="gray", aspect="auto")
        clean_pred = clean_predictions[i].item()
        true_label = clean_labels[i].item()

        if class_names:
            pred_name = (
                class_names[clean_pred]
                if clean_pred < len(class_names)
                else str(clean_pred)
            )
            true_name = (
                class_names[true_label]
                if true_label < len(class_names)
                else str(true_label)
            )
            axes[0, i].set_title(f"Clean\nTrue: {true_name}\nPred: {pred_name}")
        else:
            axes[0, i].set_title(f"Clean\nTrue: {true_label}\nPred: {clean_pred}")
        axes[0, i].axis("off")

        # Adversarial image
        adv_data = adv_images[i].cpu().numpy().flatten()
        try:
            adv_img = adv_data.reshape(image_size)
        except ValueError:
            adv_img = adv_data.reshape(1, -1)

        axes[1, i].imshow(adv_img, cmap="gray", aspect="auto")
        adv_pred = adv_predictions[i].item()

        if class_names:
            adv_pred_name = (
                class_names[adv_pred] if adv_pred < len(class_names) else str(adv_pred)
            )
            axes[1, i].set_title(f"Adversarial\nPred: {adv_pred_name}")
        else:
            axes[1, i].set_title(f"Adversarial\nPred: {adv_pred}")
        axes[1, i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_attack_progress(
    accuracies: list[float],
    fidelities: Optional[list[float]] = None,
    attack_name: str = "BIM",
    xlabel: str = "Iteration",
    title: str = "Attack Progress",
    save_path: Optional[str] = None,
):
    """Plot accuracy decay during iterative attack.

    Reproduces Figure 7 from the paper.

    Args:
        accuracies: Accuracy at each iteration
        fidelities: Optional fidelity at each iteration
        attack_name: Name of attack method
        xlabel: X-axis label
        title: Plot title
        save_path: Path to save figure
    """
    if fidelities is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    iterations = range(len(accuracies))

    # Accuracy plot
    ax1.plot(iterations, accuracies, "b-o", label=attack_name)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{title} - Accuracy vs {xlabel}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Fidelity plot
    if fidelities is not None:
        ax2.plot(fidelities, accuracies, "r-o", label=attack_name)
        ax2.set_xlabel("Average Fidelity")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"{title} - Accuracy vs Fidelity")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_robustness_comparison(
    results: dict[str, dict[float, float]],
    title: str = "Robustness to Different Attacks",
    save_path: Optional[str] = None,
):
    """Plot model robustness across different attacks and epsilon values.

    Args:
        results: Dictionary mapping attack -> epsilon -> accuracy
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (attack_name, eps_dict), color in zip(results.items(), colors):
        epsilons = sorted(eps_dict.keys())
        accuracies = [eps_dict[e] for e in epsilons]
        ax.plot(epsilons, accuracies, "-o", label=attack_name.upper(), color=color)

    ax.set_xlabel("Epsilon (ε)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_adversarial_training_progress(
    history: dict[str, list[float]],
    title: str = "Adversarial Training Progress",
    save_path: Optional[str] = None,
):
    """Plot adversarial training progress.

    Reproduces Figure 16 from the paper.

    Args:
        history: Training history with clean and adversarial accuracies
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(history["train_acc"]) + 1)

    # Plot training accuracy on clean samples
    if "train_acc" in history:
        ax.plot(epochs, history["train_acc"], "b-", label="Clean (train)", alpha=0.7)

    # Plot validation accuracy on clean samples
    if "test_acc" in history:
        ax.plot(epochs, history["test_acc"], "b--", label="Clean (val)", alpha=0.7)

    # Plot accuracy on adversarial samples
    if "test_adv_acc" in history:
        ax.plot(
            epochs,
            history["test_adv_acc"],
            "r-",
            label="Adversarial (test)",
            linewidth=2,
        )

    if "train_adv_acc" in history:
        ax.plot(
            epochs,
            history["train_adv_acc"],
            "r--",
            label="Adversarial (train)",
            alpha=0.7,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_fidelity_distribution(
    fidelities: torch.Tensor,
    title: str = "Fidelity Distribution",
    save_path: Optional[str] = None,
):
    """Plot distribution of fidelities between clean and adversarial samples.

    Args:
        fidelities: Fidelity values
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    fid_np = fidelities.cpu().numpy()

    ax.hist(fid_np, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(
        np.mean(fid_np), color="r", linestyle="--", label=f"Mean: {np.mean(fid_np):.3f}"
    )

    ax.set_xlabel("Fidelity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_noise_vs_adversarial(
    results: dict[str, Any] = None,
    noise_results: dict[float, float] = None,
    adv_results: dict[float, float] = None,
    title: str = "Random Noise vs Adversarial Perturbations",
    save_path: Optional[str] = None,
):
    """Compare effect of random noise vs adversarial perturbations.

    Reproduces Figure 11 from the paper. Shows that adversarial
    perturbations are much more effective at fooling the classifier
    than random noise of the same magnitude.

    For photonic systems, we also include photon loss noise.

    Args:
        results: Complete results dict from compare_noise_vs_adversarial
                 (preferred format with all noise types)
        noise_results: Legacy format - noise_strength -> accuracy
        adv_results: Legacy format - epsilon -> accuracy
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # New format with comprehensive results
    if results is not None and "epsilon_values" in results:
        epsilon_values = results["epsilon_values"]
        clean_acc = results.get("clean_accuracy", 1.0)

        # Plot clean accuracy baseline
        ax.axhline(
            y=clean_acc,
            color="gray",
            linestyle="--",
            linewidth=2,
            label=f"Clean ({clean_acc:.1%})",
            alpha=0.7,
        )

        # Adversarial
        if "adversarial" in results:
            ax.plot(
                epsilon_values,
                results["adversarial"],
                "ro-",
                linewidth=2,
                markersize=8,
                label="Adversarial (BIM)",
            )

        # Random uniform noise
        if "random_uniform" in results:
            ax.plot(
                epsilon_values,
                results["random_uniform"],
                "bs--",
                linewidth=2,
                markersize=8,
                label="Random Uniform",
                alpha=0.8,
            )

        # Random Gaussian noise
        if "random_gaussian" in results:
            ax.plot(
                epsilon_values,
                results["random_gaussian"],
                "g^--",
                linewidth=2,
                markersize=8,
                label="Random Gaussian",
                alpha=0.8,
            )

        # Photon loss (photonic-specific)
        if "photon_loss" in results:
            ax.plot(
                epsilon_values,
                results["photon_loss"],
                "md-",
                linewidth=2,
                markersize=8,
                label="Photon Loss",
                alpha=0.8,
            )

        ax.set_xlabel("Perturbation Magnitude (ε) / Loss Rate", fontsize=12)

    # Legacy format support
    elif noise_results is not None and adv_results is not None:
        # Random noise
        noise_strengths = sorted(noise_results.keys())
        noise_accs = [noise_results[n] for n in noise_strengths]
        ax.plot(noise_strengths, noise_accs, "b-o", label="Random Noise")

        # Adversarial
        adv_epsilons = sorted(adv_results.keys())
        adv_accs = [adv_results[e] for e in adv_epsilons]
        ax.plot(adv_epsilons, adv_accs, "r-s", label="Adversarial (BIM)")

        ax.set_xlabel("Perturbation Strength")
    else:
        raise ValueError(
            "Provide either 'results' dict or both 'noise_results' and 'adv_results'"
        )

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")
        plt.close()
    else:
        plt.show()
