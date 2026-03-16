"""
Attack Visualization Utilities
==============================

Generate attack-related figures from saved results.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_attack_from_results(
    run_dir: Path,
    results: dict[str, Any],
    config: dict[str, Any] | None = None,
    force: bool = False,
) -> Path | None:
    """Generate attack summary figure from results.

    Args:
        run_dir: Path to run directory
        results: Results dictionary
        config: Optional config for customization
        force: Overwrite existing figure

    Returns:
        Path to generated figure or None
    """
    output_path = run_dir / "attack_summary.png"

    if output_path.exists() and not force:
        logger.info(f"Skipping existing: {output_path}")
        return output_path

    # Extract attack results
    attack_method = results.get("attack_method", "Unknown")
    epsilon = results.get("epsilon", 0.0)
    fooling_rate = results.get("fooling_rate", 0.0)
    adv_acc = results.get("adversarial_accuracy", 0.0)
    clean_acc = results.get("clean_accuracy", 1.0)
    avg_fidelity = results.get("average_fidelity", 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of accuracies
    categories = ["Clean", "Adversarial"]
    accuracies = [clean_acc, adv_acc]
    colors = ["#2ecc71", "#e74c3c"]

    axes[0].bar(categories, accuracies, color=colors, edgecolor="black", linewidth=1.5)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title(
        f"{attack_method.upper()} Attack (ε={epsilon})\nFooling Rate: {fooling_rate:.1%}",
        fontsize=14,
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (_cat, acc) in enumerate(zip(categories, accuracies)):
        axes[0].text(
            i, acc + 0.02, f"{acc:.1%}", ha="center", fontsize=12, fontweight="bold"
        )

    # Metrics summary
    metrics = {
        "Fooling Rate": fooling_rate,
        "Clean Accuracy": clean_acc,
        "Adversarial Accuracy": adv_acc,
        "Average Fidelity": avg_fidelity,
    }

    y_pos = np.arange(len(metrics))
    values = list(metrics.values())
    labels = list(metrics.keys())

    colors2 = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    bars = axes[1].barh(y_pos, values, color=colors2, edgecolor="black", linewidth=1.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=11)
    axes[1].set_xlabel("Value", fontsize=12)
    axes[1].set_title("Attack Metrics", fontsize=14)
    axes[1].set_xlim(0, 1.05)
    axes[1].grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, values):
        axes[1].text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Generated: {output_path}")
    return output_path


def plot_robustness_from_results(
    run_dir: Path,
    results: dict[str, Any],
    config: dict[str, Any] | None = None,
    force: bool = False,
) -> Path | None:
    """Generate robustness comparison figure.

    Args:
        run_dir: Path to run directory
        results: Results dictionary containing robustness data
        config: Optional config for customization
        force: Overwrite existing figure

    Returns:
        Path to generated figure or None
    """
    output_path = run_dir / "robustness.png"

    if output_path.exists() and not force:
        logger.info(f"Skipping existing: {output_path}")
        return output_path

    robustness = results.get("robustness", {})
    if not robustness:
        logger.warning("No robustness data found in results")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(robustness)))

    for (attack_name, eps_dict), color in zip(robustness.items(), colors):
        if isinstance(eps_dict, dict):
            epsilons = sorted([float(e) for e in eps_dict.keys()])
            accuracies = [
                eps_dict[str(e)] if str(e) in eps_dict else eps_dict[e]
                for e in epsilons
            ]
        else:
            continue

        ax.plot(
            epsilons,
            accuracies,
            "-o",
            label=attack_name.upper(),
            color=color,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Robustness to Different Attacks", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Generated: {output_path}")
    return output_path


def plot_attack_progress_from_results(
    run_dir: Path,
    results: dict[str, Any],
    config: dict[str, Any] | None = None,
    force: bool = False,
) -> Path | None:
    """Generate attack progress figure showing accuracy vs iterations.

    Args:
        run_dir: Path to run directory
        results: Results dictionary containing iteration data
        config: Optional config for customization
        force: Overwrite existing figure

    Returns:
        Path to generated figure or None
    """
    output_path = run_dir / "attack_progress.png"

    if output_path.exists() and not force:
        logger.info(f"Skipping existing: {output_path}")
        return output_path

    accuracies = results.get("accuracies_per_iteration", [])
    fidelities = results.get("fidelities_per_iteration", None)

    if not accuracies:
        logger.warning("No iteration data found in results")
        return None

    attack_name = results.get("attack_method", "Attack")

    if fidelities is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    iterations = range(len(accuracies))

    # Accuracy vs iteration
    ax1.plot(
        iterations,
        accuracies,
        "b-o",
        label=attack_name.upper(),
        linewidth=2,
        markersize=6,
    )
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(f"{attack_name.upper()} - Accuracy vs Iteration", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Accuracy vs fidelity
    if fidelities is not None:
        ax2.plot(
            fidelities,
            accuracies,
            "r-o",
            label=attack_name.upper(),
            linewidth=2,
            markersize=6,
        )
        ax2.set_xlabel("Average Fidelity", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title(f"{attack_name.upper()} - Accuracy vs Fidelity", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Generated: {output_path}")
    return output_path
