"""
Visualization module for Quantum Nearest Centroid paper reproduction.

Generates all figures from the paper:
- Fig 8: Synthetic data (distance ratios + classification error bars)
- Fig 9: IRIS data (distance ratios + classification error bars)
- Fig 10: IRIS 2D PCA visualization
- Fig 11: MNIST data (distance ratios + classification error bars)
- Fig 12: Confusion matrices
- Fig 13: c_exp vs c_sim scatter plot
- Tables I, II: Label comparisons
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def set_paper_style():
    """Set matplotlib style to match paper aesthetics."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.figsize": (6, 4),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_classification_error_bars(results, title, save_path=None):
    """
    Generate classification error bar chart (like Fig 8b, 9b, 11b).

    Args:
        results: List of dicts with keys:
            - 'label': x-axis label (e.g., "Nq=4\nNc=2\nNs=100")
            - 'c_acc_mean', 'c_acc_std': Classical accuracy
            - 'acc_mean', 'acc_std': Cirq (quantum) accuracy
            - 'ml_acc_mean', 'ml_acc_std': Merlin accuracy
        title: Plot title
        save_path: Path to save figure
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    width = 0.25

    # Convert accuracy to error percentage
    classical_err = [(1 - r["c_acc_mean"]) * 100 for r in results]
    classical_std = [r["c_acc_std"] * 100 for r in results]

    cirq_err = [(1 - r["acc_mean"]) * 100 for r in results]
    cirq_std = [r["acc_std"] * 100 for r in results]

    merlin_err = [(1 - r["ml_acc_mean"]) * 100 for r in results]
    merlin_std = [r["ml_acc_std"] * 100 for r in results]

    # Plot bars
    ax.bar(
        x - width,
        classical_err,
        width,
        yerr=classical_std,
        label="Classical",
        color="#2ecc71",
        capsize=4,
        alpha=0.8,
    )
    ax.bar(
        x,
        cirq_err,
        width,
        yerr=cirq_std,
        label="Cirq (IonQ)",
        color="#3498db",
        capsize=4,
        alpha=0.8,
    )
    ax.bar(
        x + width,
        merlin_err,
        width,
        yerr=merlin_std,
        label="Merlin",
        color="#e74c3c",
        capsize=4,
        alpha=0.8,
    )

    ax.set_xlabel("Experiment Configuration")
    ax.set_ylabel("Classification Error %")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig, ax


def plot_distance_ratios(results, title, save_path=None):
    """
    Generate distance ratio plot (like Fig 8a, 9a, 11a).

    Args:
        results: List of dicts with keys:
            - 'label': x-axis label
            - 'dist_ratios_cirq': List of (l_exp / l_sim) ratios for Cirq
            - 'dist_ratios_merlin': List of ratios for Merlin
        title: Plot title
        save_path: Path to save figure
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = [r["label"] for r in results]
    x = np.arange(len(labels))

    # Calculate means and stds
    cirq_means = [np.mean(r.get("dist_ratios_cirq", [1.0])) for r in results]
    cirq_stds = [np.std(r.get("dist_ratios_cirq", [0.0])) for r in results]

    merlin_means = [np.mean(r.get("dist_ratios_merlin", [1.0])) for r in results]
    merlin_stds = [np.std(r.get("dist_ratios_merlin", [0.0])) for r in results]

    # Plot with error bars
    ax.errorbar(
        x - 0.1,
        cirq_means,
        yerr=cirq_stds,
        fmt="s",
        color="#3498db",
        label="Cirq",
        capsize=4,
        markersize=8,
    )
    ax.errorbar(
        x + 0.1,
        merlin_means,
        yerr=merlin_stds,
        fmt="o",
        color="#e74c3c",
        label="Merlin",
        capsize=4,
        markersize=8,
    )

    # Reference line at 1.0
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Experiment Configuration")
    ax.set_ylabel(r"$l_{exp} / l_{sim}$")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig, ax


def plot_combined_figure(results, title, save_path=None):
    """Generate combined figure with distance ratios (top) and classification error (bottom)."""
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    width = 0.25

    # Top plot: Distance ratios
    cirq_means = [np.mean(r.get("dist_ratios_cirq", [1.0])) for r in results]
    cirq_stds = [np.std(r.get("dist_ratios_cirq", [0.0])) for r in results]
    merlin_means = [np.mean(r.get("dist_ratios_merlin", [1.0])) for r in results]
    merlin_stds = [np.std(r.get("dist_ratios_merlin", [0.0])) for r in results]

    ax1.errorbar(
        x - 0.1,
        cirq_means,
        yerr=cirq_stds,
        fmt="s",
        color="#3498db",
        label="Cirq",
        capsize=4,
        markersize=8,
    )
    ax1.errorbar(
        x + 0.1,
        merlin_means,
        yerr=merlin_stds,
        fmt="o",
        color="#e74c3c",
        label="Merlin",
        capsize=4,
        markersize=8,
    )
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel(r"$l_{exp} / l_{sim}$")
    ax1.set_xticks(x)
    ax1.set_xticklabels([])
    ax1.legend()
    ax1.text(-0.1, 1.05, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold")

    # Bottom plot: Classification error
    classical_err = [(1 - r["c_acc_mean"]) * 100 for r in results]
    classical_std = [r["c_acc_std"] * 100 for r in results]
    cirq_err = [(1 - r["acc_mean"]) * 100 for r in results]
    cirq_std = [r["acc_std"] * 100 for r in results]
    merlin_err = [(1 - r["ml_acc_mean"]) * 100 for r in results]
    merlin_std = [r["ml_acc_std"] * 100 for r in results]

    ax2.bar(
        x - width,
        classical_err,
        width,
        yerr=classical_std,
        label="Classical",
        color="#2ecc71",
        capsize=4,
        alpha=0.8,
    )
    ax2.bar(
        x,
        cirq_err,
        width,
        yerr=cirq_std,
        label="Cirq",
        color="#3498db",
        capsize=4,
        alpha=0.8,
    )
    ax2.bar(
        x + width,
        merlin_err,
        width,
        yerr=merlin_std,
        label="Merlin",
        color="#e74c3c",
        capsize=4,
        alpha=0.8,
    )

    ax2.set_xlabel("Experiment Configuration")
    ax2.set_ylabel("Classification Error %")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.set_ylim(bottom=0)
    ax2.text(-0.1, 1.05, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig, (ax1, ax2)


def plot_confusion_matrix(
    y_true, y_pred, classes, title, accuracy=None, save_path=None
):
    """Generate a single confusion matrix plot."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    # annotation with percentage and counts
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_normalized[i, j] * 100
            count = cm[i, j]
            annot[i, j] = f"{pct:.1f}%\n{count}"

    sns.heatmap(
        cm_normalized,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        xticklabels=classes,
        yticklabels=classes,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar=True,
        annot_kws={"size": 8},
    )

    ax.set_xlabel("Output Class")
    ax.set_ylabel("Target Class")

    if accuracy is not None:
        title = f"{title}\nAccuracy: {accuracy:.2f}%"
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig, ax


def plot_confusion_matrices_comparison(
    y_true, y_pred_classical, y_pred_quantum, classes, save_path=None
):
    """Side-by-side confusion matrices comparing classical and quantum."""
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, y_pred, title in [
        (ax1, y_pred_classical, "Classical"),
        (ax2, y_pred_quantum, "Quantum"),
    ]:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        accuracy = np.trace(cm) / np.sum(cm) * 100

        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                pct = cm_normalized[i, j] * 100
                count = cm[i, j]
                annot[i, j] = f"{pct:.1f}%\n{count}"

        sns.heatmap(
            cm_normalized,
            annot=annot,
            fmt="",
            cmap="RdYlGn",
            xticklabels=classes,
            yticklabels=classes,
            vmin=0,
            vmax=1,
            ax=ax,
            cbar=True,
            annot_kws={"size": 7},
        )

        ax.set_xlabel("Output Class")
        ax.set_ylabel("Target Class")
        ax.set_title(f"{title}\nAccuracy: {accuracy:.2f}%")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig, (ax1, ax2)


def create_label_comparison_table(
    sampling_labels,
    classical_labels,
    quantum_no_mit_labels,
    quantum_mit_labels,
    save_path=None,
):
    """Generate label comparison table (like Tables I, II)."""
    n = len(sampling_labels)

    header = "| " + " | ".join(["Method"] + [str(i) for i in range(n)]) + " |"
    separator = "|" + "|".join(["-" * 20] + ["-" * 3 for _ in range(n)]) + "|"

    rows = []
    rows.append(
        "| Sampling of points | " + " | ".join(map(str, sampling_labels)) + " |"
    )
    rows.append("| Classical NC | " + " | ".join(map(str, classical_labels)) + " |")
    rows.append(
        "| QNC (no mitigation) | " + " | ".join(map(str, quantum_no_mit_labels)) + " |"
    )
    rows.append(
        "| QNC (with mitigation) | " + " | ".join(map(str, quantum_mit_labels)) + " |"
    )

    table = "\n".join([header, separator] + rows)

    quantum_no_mit_acc = (
        np.mean(np.array(quantum_no_mit_labels) == np.array(classical_labels)) * 100
    )
    quantum_mit_acc = (
        np.mean(np.array(quantum_mit_labels) == np.array(classical_labels)) * 100
    )

    summary = "\nAccuracy vs Classical:\n"
    summary += f"  QNC (no mitigation): {quantum_no_mit_acc:.1f}%\n"
    summary += f"  QNC (with mitigation): {quantum_mit_acc:.1f}%\n"

    result = table + summary

    if save_path:
        with open(save_path, "w") as f:
            f.write(result)
        print(f"Saved: {save_path}")

    return result


def generate_all_figures(results_dir, output_dir):
    """Generate figures from a saved summary_results.json."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary_results.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results = json.load(f)
    else:
        print(f"Warning: {summary_path} not found")
        results = []

    if results:
        formatted_results = []
        for r in results:
            classes_str = ",".join(map(str, r["classes"]))
            formatted_results.append(
                {
                    "label": f"Classes: {classes_str}\nn={r['n_samples']}",
                    "c_acc_mean": r["c_acc_mean"],
                    "c_acc_std": r["c_acc_std"],
                    "acc_mean": r["acc_mean"],
                    "acc_std": r["acc_std"],
                    "ml_acc_mean": r["ml_acc_mean"],
                    "ml_acc_std": r["ml_acc_std"],
                    "dist_ratios_cirq": r.get("dist_ratios_cirq", [1.0]),
                    "dist_ratios_merlin": r.get("dist_ratios_merlin", [1.0]),
                }
            )

        plot_classification_error_bars(
            formatted_results,
            "Classification Results",
            save_path=output_dir / "classification_error.png",
        )

        plot_combined_figure(
            formatted_results,
            "Experiment Results",
            save_path=output_dir / "combined_figure.png",
        )

    print(f"\nAll figures saved to: {output_dir}")


# ===========================================================================
# ADDITIONAL VISUALIZATION FUNCTIONS FOR PAPER REPRODUCTION
# Add these functions to lib/visualization.py
# ===========================================================================


def plot_iris_pca_scatter(
    X_pca: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    centroids_pca: np.ndarray = None,
    title: str = "IRIS Classification",
    save_path=None,
    before_mitigation: bool = False,
):
    """
    Generate IRIS 2D PCA scatter plot (Figure 10 from paper).

    Shows data points with:
    - Boundary color = true label (human-assigned)
    - Fill color = predicted label (from classifier)

    Parameters
    ----------
    X_pca : np.ndarray
        2D PCA-transformed data points, shape (n_samples, 2)
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    centroids_pca : np.ndarray, optional
        2D PCA-transformed centroids
    title : str
        Plot title
    save_path : str or Path, optional
        Path to save figure
    before_mitigation : bool
        If True, title indicates "before error mitigation"
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    # Color map for classes
    colors = ["#e74c3c", "#3498db", "#2ecc71"]  # Red, Blue, Green
    class_names = ["Setosa", "Versicolor", "Virginica"]

    unique_classes = np.unique(y_true)

    for i, point in enumerate(X_pca):
        true_class = int(y_true[i])
        pred_class = int(y_pred[i])

        # Boundary color = true label, fill color = predicted label
        ax.scatter(
            point[0],
            point[1],
            c=colors[pred_class],
            edgecolors=colors[true_class],
            s=80,
            linewidths=2,
            alpha=0.7,
        )

    # Plot centroids if provided
    if centroids_pca is not None:
        for i, centroid in enumerate(centroids_pca):
            ax.scatter(
                centroid[0],
                centroid[1],
                c=colors[i],
                marker="X",
                s=200,
                edgecolors="black",
                linewidths=2,
                label=f"Centroid {class_names[i]}"
                if i < len(class_names)
                else f"Centroid {i}",
            )

    # Create legend
    from matplotlib.lines import Line2D

    legend_elements = []
    for _i, (color, name) in enumerate(
        zip(colors[: len(unique_classes)], class_names[: len(unique_classes)])
    ):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=10,
                label=name,
            )
        )
    ax.legend(handles=legend_elements, loc="best")

    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred) * 100

    mit_str = (
        "(before error mitigation)" if before_mitigation else "(after error mitigation)"
    )
    ax.set_title(f"{title} {mit_str}\nAccuracy: {accuracy:.1f}%")
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig, ax


def plot_iris_comparison(
    X_pca: np.ndarray,
    y_true: np.ndarray,
    y_pred_before: np.ndarray,
    y_pred_after: np.ndarray,
    save_path=None,
):
    """
    Generate side-by-side IRIS scatter plots (Figure 10 from paper).

    Left: Before error mitigation
    Right: After error mitigation
    """
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for ax, y_pred, title_suffix in [
        (ax1, y_pred_before, "(a) Before Error Mitigation"),
        (ax2, y_pred_after, "(b) After Error Mitigation"),
    ]:
        for i, point in enumerate(X_pca):
            true_class = int(y_true[i])
            pred_class = int(y_pred[i])

            ax.scatter(
                point[0],
                point[1],
                c=colors[pred_class],
                edgecolors=colors[true_class],
                s=80,
                linewidths=2,
                alpha=0.7,
            )

        accuracy = np.mean(y_true == y_pred) * 100
        ax.set_title(f"{title_suffix}\nAccuracy: {accuracy:.1f}%")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    # Set same axis limits for both
    all_x = X_pca[:, 0]
    all_y = X_pca[:, 1]
    margin = 0.5
    for ax in [ax1, ax2]:
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig, (ax1, ax2)


def plot_c_exp_vs_c_sim(
    c_sim_values: np.ndarray,
    c_exp_no_mitigation: np.ndarray,
    c_exp_with_mitigation: np.ndarray,
    save_path=None,
):
    """
    Generate c_exp vs c_sim scatter plot with linear fits (Figure 13 from paper).

    This validates the noise model by showing the linear relationship between
    simulated and experimental inner product values.

    Parameters
    ----------
    c_sim_values : np.ndarray
        Simulated (ideal) inner product squared values
    c_exp_no_mitigation : np.ndarray
        Experimental values without error mitigation
    c_exp_with_mitigation : np.ndarray
        Experimental values with error mitigation
    save_path : optional
        Path to save figure
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter plots
    ax.scatter(
        c_sim_values,
        c_exp_no_mitigation,
        c="#3498db",
        marker="o",
        s=50,
        alpha=0.6,
        label="No mitigation",
    )
    ax.scatter(
        c_sim_values,
        c_exp_with_mitigation,
        c="#e74c3c",
        marker="s",
        s=50,
        alpha=0.6,
        label="After mitigation",
    )

    # Linear fits
    # No mitigation fit
    mask_no_mit = ~np.isnan(c_exp_no_mitigation) & ~np.isnan(c_sim_values)
    if mask_no_mit.sum() > 1:
        coeffs_no_mit = np.polyfit(
            c_sim_values[mask_no_mit], c_exp_no_mitigation[mask_no_mit], 1
        )
        x_fit = np.linspace(0, 1, 100)
        y_fit_no_mit = coeffs_no_mit[0] * x_fit + coeffs_no_mit[1]
        ax.plot(
            x_fit,
            y_fit_no_mit,
            "#3498db",
            linestyle="-",
            linewidth=2,
            label=f"No mitigation: y = {coeffs_no_mit[0]:.2f}*x + {coeffs_no_mit[1]:.3f}",
        )

    # With mitigation fit
    mask_mit = ~np.isnan(c_exp_with_mitigation) & ~np.isnan(c_sim_values)
    if mask_mit.sum() > 1:
        coeffs_mit = np.polyfit(
            c_sim_values[mask_mit], c_exp_with_mitigation[mask_mit], 1
        )
        y_fit_mit = coeffs_mit[0] * x_fit + coeffs_mit[1]
        ax.plot(
            x_fit,
            y_fit_mit,
            "#e74c3c",
            linestyle="-",
            linewidth=2,
            label=f"After mitigation: y = {coeffs_mit[0]:.2f}*x + {coeffs_mit[1]:.3f}",
        )

    # Reference line y = x (ideal)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Ideal (y = x)")

    ax.set_xlabel(r"$c_{sim}$")
    ax.set_ylabel(r"$c_{exp}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.8, np.nanmax(c_exp_with_mitigation) * 1.1))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Experimental vs Simulated Inner Product")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig, ax


def plot_synthetic_results(
    results_4q: dict,
    results_8q: dict,
    save_path=None,
):
    """
    Generate synthetic data results plot (Figure 8 from paper).

    Parameters
    ----------
    results_4q : dict
        Results for 4-qubit experiments with keys like 'Nc2', 'Nc4'
    results_8q : dict
        Results for 8-qubit experiments
    """
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Prepare data
    configs = [
        ("Nq=4\nNc=2\nNs=100", results_4q.get("Nc2", {})),
        ("Nq=4\nNc=4\nNs=500", results_4q.get("Nc4", {})),
        ("Nq=8\nNc=2\nNs=1000", results_8q.get("Nc2", {})),
        ("Nq=8\nNc=4\nNs=1000", results_8q.get("Nc4", {})),
    ]

    labels = [c[0] for c in configs]
    x = np.arange(len(labels))

    # (a) Distance ratios
    dist_ratios = [c[1].get("dist_ratio_mean", 1.0) for c in configs]
    dist_ratio_stds = [c[1].get("dist_ratio_std", 0.0) for c in configs]

    ax1.errorbar(
        x,
        dist_ratios,
        yerr=dist_ratio_stds,
        fmt="o",
        color="#3498db",
        capsize=5,
        markersize=8,
    )
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel(r"$l_{exp} / l_{sim}$")
    ax1.set_xticks(x)
    ax1.set_xticklabels([])
    ax1.set_ylim(0, 3)
    ax1.text(-0.1, 1.05, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold")

    # (b) Classification error
    width = 0.35
    before_err = [(1 - c[1].get("acc_no_mit", 1.0)) * 100 for c in configs]
    after_err = [(1 - c[1].get("acc_mit", 1.0)) * 100 for c in configs]

    ax2.bar(
        x - width / 2,
        before_err,
        width,
        label="Before mitigation",
        color="#3498db",
        alpha=0.8,
    )
    ax2.bar(
        x + width / 2,
        after_err,
        width,
        label="After mitigation",
        color="#e74c3c",
        alpha=0.8,
    )

    ax2.set_ylabel("Classification Error %")
    ax2.set_xlabel("Experiment Configuration")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.set_ylim(0, 35)
    ax2.text(-0.1, 1.05, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig, (ax1, ax2)


# ===========================================================================
# ADDITIONAL VISUALIZATION FUNCTIONS FOR PAPER REPRODUCTION
# Add these functions to lib/visualization.py
# ===========================================================================


# ===========================================================================
# ADDITIONAL VISUALIZATION FUNCTIONS FOR PAPER REPRODUCTION
# Add these functions to lib/visualization.py
# ===========================================================================
