"""
Comparison Visualization Utilities
==================================

Generate comparison figures from saved results.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_noise_comparison_from_results(
    run_dir: Path,
    results: dict[str, Any],
    config: dict[str, Any] | None = None,
    force: bool = False,
) -> Path | None:
    """Generate noise vs adversarial comparison figure.

    Reproduces Figure 11 from Lu et al. (2020).

    Args:
        run_dir: Path to run directory
        results: Results dictionary
        config: Optional config for customization
        force: Overwrite existing figure

    Returns:
        Path to generated figure or None
    """
    output_path = run_dir / "noise_vs_adversarial.png"

    if output_path.exists() and not force:
        logger.info(f"Skipping existing: {output_path}")
        return output_path

    if "epsilon_values" not in results:
        logger.warning("No noise comparison data found in results")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

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
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Adversarial vs Random Noise vs Photon Loss", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Generated: {output_path}")
    return output_path


def plot_transfer_from_results(
    run_dir: Path,
    results: dict[str, Any],
    config: dict[str, Any] | None = None,
    force: bool = False,
) -> Path | None:
    """Generate transfer attack results table/figure.

    Reproduces Table III from Lu et al. (2020).

    Args:
        run_dir: Path to run directory
        results: Results dictionary
        config: Optional config for customization
        force: Overwrite existing figure

    Returns:
        Path to generated figure or None
    """
    output_path = run_dir / "transfer_attack.png"

    if output_path.exists() and not force:
        logger.info(f"Skipping existing: {output_path}")
        return output_path

    transfer_results = results.get("transfer_results", results)

    if not transfer_results or "surrogate_types" not in results:
        logger.warning("No transfer attack data found in results")
        return None

    # Create a heatmap-style figure
    fig, ax = plt.subplots(figsize=(10, 6))

    surrogate_types = results.get("surrogate_types", [])
    attack_methods = results.get("attack_methods", [])

    # Build data matrix
    data = []
    row_labels = []
    for surrogate in surrogate_types:
        for attack in attack_methods:
            key = f"{surrogate}_{attack}"
            if key in transfer_results:
                fooling_rate = transfer_results[key].get("fooling_rate", 0)
                data.append([fooling_rate])
                row_labels.append(f"{surrogate.upper()} + {attack.upper()}")

    if not data:
        logger.warning("Could not parse transfer results")
        return None

    data = np.array(data)

    # Create horizontal bar chart
    y_pos = np.arange(len(row_labels))
    colors = plt.cm.RdYlGn_r(data.flatten())

    bars = ax.barh(y_pos, data.flatten(), color=colors, edgecolor="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Fooling Rate", fontsize=12)
    ax.set_title("Black-box Transfer Attack Results", fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, data.flatten()):
        ax.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Generated: {output_path}")
    return output_path


def plot_model_comparison_from_results(
    run_dir: Path,
    results: dict[str, Any],
    config: dict[str, Any] | None = None,
    force: bool = False,
) -> Path | None:
    """Generate model comparison figure (photonic vs gate-based).

    Args:
        run_dir: Path to run directory
        results: Results dictionary
        config: Optional config for customization
        force: Overwrite existing figure

    Returns:
        Path to generated figure or None
    """
    output_path = run_dir / "model_comparison.png"

    if output_path.exists() and not force:
        logger.info(f"Skipping existing: {output_path}")
        return output_path

    # Extract comparison data
    photonic = results.get("photonic", {})
    gate_based = results.get("gate_based", {})

    if not photonic and not gate_based:
        logger.warning("No comparison data found in results")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Clean accuracy comparison
    models = []
    clean_accs = []
    adv_accs = []

    if photonic:
        models.append("Photonic\n(MerLin)")
        clean_accs.append(photonic.get("clean_accuracy", 0))
        adv_accs.append(photonic.get("adversarial_accuracy", 0))

    if gate_based:
        models.append("Gate-based\n(PennyLane)")
        clean_accs.append(gate_based.get("clean_accuracy", 0))
        adv_accs.append(gate_based.get("adversarial_accuracy", 0))

    x = np.arange(len(models))
    width = 0.35

    bars1 = axes[0].bar(
        x - width / 2,
        clean_accs,
        width,
        label="Clean",
        color="#2ecc71",
        edgecolor="black",
    )
    bars2 = axes[0].bar(
        x + width / 2,
        adv_accs,
        width,
        label="Adversarial",
        color="#e74c3c",
        edgecolor="black",
    )

    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Model Comparison: Clean vs Adversarial", fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(
                f"{height:.1%}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Robustness comparison (if available)
    if "robustness" in photonic or "robustness" in gate_based:
        # Plot robustness curves
        if "robustness" in photonic:
            for attack, eps_dict in photonic.get("robustness", {}).items():
                if isinstance(eps_dict, dict):
                    epsilons = sorted([float(e) for e in eps_dict.keys()])
                    accs = [
                        eps_dict[str(e)] if str(e) in eps_dict else eps_dict[e]
                        for e in epsilons
                    ]
                    axes[1].plot(
                        epsilons, accs, "-o", label=f"Photonic ({attack})", linewidth=2
                    )

        if "robustness" in gate_based:
            for attack, eps_dict in gate_based.get("robustness", {}).items():
                if isinstance(eps_dict, dict):
                    epsilons = sorted([float(e) for e in eps_dict.keys()])
                    accs = [
                        eps_dict[str(e)] if str(e) in eps_dict else eps_dict[e]
                        for e in epsilons
                    ]
                    axes[1].plot(
                        epsilons,
                        accs,
                        "--s",
                        label=f"Gate-based ({attack})",
                        linewidth=2,
                    )

        axes[1].set_xlabel("Epsilon (ε)", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title("Robustness Comparison", fontsize=14)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)
    else:
        # Show metrics table instead
        axes[1].axis("off")
        metrics_text = "Comparison Metrics:\n\n"
        if photonic:
            metrics_text += "Photonic (MerLin):\n"
            metrics_text += f"  Clean: {photonic.get('clean_accuracy', 0):.1%}\n"
            metrics_text += (
                f"  Adversarial: {photonic.get('adversarial_accuracy', 0):.1%}\n"
            )
            metrics_text += f"  Fooling Rate: {photonic.get('fooling_rate', 0):.1%}\n\n"
        if gate_based:
            metrics_text += "Gate-based (PennyLane):\n"
            metrics_text += f"  Clean: {gate_based.get('clean_accuracy', 0):.1%}\n"
            metrics_text += (
                f"  Adversarial: {gate_based.get('adversarial_accuracy', 0):.1%}\n"
            )
            metrics_text += f"  Fooling Rate: {gate_based.get('fooling_rate', 0):.1%}\n"

        axes[1].text(
            0.1,
            0.5,
            metrics_text,
            transform=axes[1].transAxes,
            fontsize=12,
            verticalalignment="center",
            fontfamily="monospace",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Generated: {output_path}")
    return output_path
