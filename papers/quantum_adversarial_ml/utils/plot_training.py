"""
Training Visualization Utilities
================================

Generate training curve figures from saved results.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_training_from_results(
    run_dir: Path,
    results: dict[str, Any],
    config: dict[str, Any] | None = None,
    force: bool = False,
) -> Path | None:
    """Generate training curves from results.

    Args:
        run_dir: Path to run directory
        results: Results dictionary containing history
        config: Optional config for title customization
        force: Overwrite existing figure

    Returns:
        Path to generated figure or None
    """
    output_path = run_dir / "training_curves.png"

    if output_path.exists() and not force:
        logger.info(f"Skipping existing: {output_path}")
        return output_path

    # Extract history from results
    history = results.get("history", results)

    # Check for required keys
    if "train_loss" not in history and "train_acc" not in history:
        logger.warning("No training history found in results")
        return None

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Determine title
    if config:
        exp_type = config.get("experiment", "Training")
        dataset = config.get("dataset", {})
        if "digits" in dataset:
            digits = dataset["digits"]
            title = f"MNIST {len(digits)}-class ({', '.join(map(str, digits))})"
        else:
            title = exp_type.replace("_", " ").title()
    else:
        title = "Training Progress"

    # Plot loss
    if "train_loss" in history:
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
        if "test_loss" in history:
            axes[0].plot(epochs, history["test_loss"], "r-", label="Test", linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title(f"{title} - Loss", fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    if "train_acc" in history:
        epochs = range(1, len(history["train_acc"]) + 1)
        axes[1].plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
        if "test_acc" in history:
            axes[1].plot(epochs, history["test_acc"], "r-", label="Test", linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title(f"{title} - Accuracy", fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Generated: {output_path}")
    return output_path


def plot_adversarial_training_from_results(
    run_dir: Path,
    results: dict[str, Any],
    config: dict[str, Any] | None = None,
    force: bool = False,
) -> Path | None:
    """Generate adversarial training progress figure.

    Args:
        run_dir: Path to run directory
        results: Results dictionary containing history
        config: Optional config for customization
        force: Overwrite existing figure

    Returns:
        Path to generated figure or None
    """
    output_path = run_dir / "adversarial_training.png"

    if output_path.exists() and not force:
        logger.info(f"Skipping existing: {output_path}")
        return output_path

    history = results.get("history", results)

    if "test_acc" not in history:
        logger.warning("No adversarial training history found")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(history["test_acc"]) + 1)

    # Clean accuracy
    ax.plot(epochs, history["test_acc"], "b-", label="Clean (test)", linewidth=2)
    if "train_acc" in history:
        ax.plot(epochs, history["train_acc"], "b--", label="Clean (train)", alpha=0.7)

    # Adversarial accuracy
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

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Adversarial Training Progress", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Generated: {output_path}")
    return output_path
