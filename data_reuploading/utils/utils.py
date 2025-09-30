"""
utils.py - Utility functions for quantum data re-uploading experiments
====================================================================

This module contains utility functions for visualization and analysis
of quantum machine learning experiments.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update(
    {
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "figure.figsize": (10, 6),
    }
)
# Attempt to use LaTeX for mathematical expressions
try:
    plt.rcParams.update(
        {"text.usetex": False, "text.latex.preamble": r"\usepackage{amsfonts,amsmath}"}
    )
except Exception:
    warnings.warn(
        "LaTeX rendering not available, falling back to default fonts", stacklevel=2
    )
    plt.rcParams["text.usetex"] = False

# Color palette for consistent plotting
COLORS = ["#DDA0DD", "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#98FB98"]


def plot_training_loss(model):
    """
    Plot the training loss from a fitted model's training history.

    Parameters
    ----------
    model : MerlinReuploadingClassifier or PercevalReuploadingClassifier
        A fitted model with training_history_ attribute
    title : str
        Title for the plot
    """
    if not hasattr(model, "training_history_") or not model.training_history_["loss"]:
        print("No training history found. Train the model first.")
        return

    losses = model.training_history_["loss"]
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, color=COLORS[0], linewidth=2, marker="+", markersize=4)
    plt.xlabel("$epoch$")
    plt.ylabel(r"$\mathcal{L}_{LDA}$")
    plt.grid(True, alpha=0.8)

    # Add statistics
    final_loss = losses[-1]
    min_loss = min(losses)
    min_epoch = epochs[losses.index(min_loss)]

    plt.text(
        0.7,
        0.9,
        f"Final $\\mathcal{{L}}_{{LDA}} = {final_loss:.4f}$\n $\\min \\mathcal{{L}}_{{LDA}} = {min_loss:.4f} \\; (epoch = {min_epoch})$",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.show()


def plot_classification_map(
    model,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    x_lim=None,
    y_lim=None,
    resolution=200,
    n_shots=None,
    figsize=(10, 8),
):
    """
    Model visualization combining probabilities, data, and uncertainty.

    Parameters
    ----------
    model : fitted classifier
        Trained model with predict_proba() method
    X_train, y_train : np.ndarray
        Training data and labels
    X_test, y_test : np.ndarray, optional
        Test data and labels (shown with different markers)
    x_lim, y_lim : tuple, optional
        Axis limits. If None, auto-determined from data
    resolution : int, default=150
        Grid resolution for probability heatmap
    n_shots : int, default=2000
        Number of shots used to compute the map
    figsize : tuple, default=(10, 8)
        Figure size
    """

    # Auto-determine limits if not provided
    all_data = X_train
    if X_test is not None:
        all_data = np.vstack([X_train, X_test])

    if x_lim is None:
        margin = 0.15 * (all_data[:, 0].max() - all_data[:, 0].min())
        x_lim = (all_data[:, 0].min() - margin, all_data[:, 0].max() + margin)

    if y_lim is None:
        margin = 0.15 * (all_data[:, 1].max() - all_data[:, 1].min())
        y_lim = (all_data[:, 1].min() - margin, all_data[:, 1].max() + margin)

    # Create grid
    x = np.linspace(x_lim[0], x_lim[1], resolution)
    y = np.linspace(y_lim[0], y_lim[1], resolution)
    X_grid, Y_grid = np.meshgrid(x, y)
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

    # Get predictions and probabilities
    if n_shots is not None:
        grid_probabilities = model.predict_proba(grid_points, n_shots)
    else:
        grid_probabilities = model.predict_proba(grid_points)
    prob_class_0 = grid_probabilities[:, 0].reshape(X_grid.shape)
    prob_class_1 = grid_probabilities[:, 1].reshape(X_grid.shape)

    # Calculate uncertainty (entropy)
    epsilon = 1e-10
    -(
        prob_class_0 * np.log2(prob_class_0 + epsilon)
        + prob_class_1 * np.log2(prob_class_1 + epsilon)
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Create a custom colormap: red -> white -> blue
    # This represents P(class 0) in red, P(class 1) in blue, uncertain in white
    colors_list = ["#d62728", "#ffffff", "#1f77b4"]  # Red -> White -> Blue
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("red_white_blue", colors_list, N=n_bins)

    # Convert probabilities to a single value for coloring
    # -1 = definitely class 0, 0 = uncertain, +1 = definitely class 1
    prob_difference = prob_class_1 - prob_class_0  # Range: [-1, 1]

    # Create the probability heatmap
    im = ax.contourf(
        X_grid,
        Y_grid,
        prob_difference,
        levels=np.linspace(-1, 1, 21),
        cmap=cmap,
        alpha=0.7,
        extend="both",
    )

    # Add decision boundary (where P(class 0) = P(class 1) = 0.5)
    ax.contour(
        X_grid,
        Y_grid,
        prob_difference,
        levels=[0],
        colors=["black"],
        linewidths=2.5,
        linestyles="-",
        alpha=0.8,
    )

    handles, _labels = [], []

    # Plot training data
    ax.scatter(
        X_train[y_train == 0, 0],
        X_train[y_train == 0, 1],
        c="darkred",
        s=80,
        alpha=0.9,
        edgecolors="white",
        linewidth=1.5,
        label="Class 0 (Train)",
        marker="o",
        zorder=5,
    )
    ax.scatter(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        c="darkblue",
        s=80,
        alpha=0.9,
        edgecolors="white",
        linewidth=1.5,
        label="Class 1 (Train)",
        marker="o",
        zorder=5,
    )

    # Plot test data if provided
    if X_test is not None and y_test is not None:
        ax.scatter(
            X_test[y_test == 0, 0],
            X_test[y_test == 0, 1],
            c="darkred",
            s=100,
            alpha=0.9,
            edgecolors="white",
            linewidth=2,
            label="Class 0 (Test)",
            marker="s",
            zorder=5,
        )
        ax.scatter(
            X_test[y_test == 1, 0],
            X_test[y_test == 1, 1],
            c="darkblue",
            s=100,
            alpha=0.9,
            edgecolors="white",
            linewidth=2,
            label="Class 1 (Test)",
            marker="s",
            zorder=5,
        )

    # Formatting
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Create custom legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="darkred",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Class 0 (Train)",
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="darkblue",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Class 1 (Train)",
            linestyle="None",
        ),
    ]

    if X_test is not None and y_test is not None:
        legend_elements.extend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="darkred",
                    markersize=10,
                    markeredgecolor="white",
                    markeredgewidth=2,
                    label="Class 0 (Test)",
                    linestyle="None",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="darkblue",
                    markersize=10,
                    markeredgecolor="white",
                    markeredgewidth=2,
                    label="Class 1 (Test)",
                    linestyle="None",
                ),
            ]
        )

    # Add uncertainty to legend if present
    if handles:
        legend_elements.extend(handles)

    ax.legend(
        handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=11, ncol=3
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(
        [
            "class 0\n(Certain)",
            "class 0\n(Likely)",
            "Uncertain",
            "class 1\n(Likely)",
            "class 1\n(Certain)",
        ]
    )

    plt.tight_layout()
    plt.show()


def plot_probability_axis(model, X_train, y_train, X_test=None, y_test=None, offset=0):
    """Simple horizontal axis plot of p_01 quantum features."""

    train_p_01 = model.get_quantum_features(X_train)[:, 0]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axhline(y=0, color="black", linewidth=2)

    # Training data
    ax.scatter(
        train_p_01[y_train == 0],
        np.ones(sum(y_train == 0)) * offset,
        c="red",
        s=60,
        marker="o",
        alpha=0.7,
        label="Class 0 (Train)",
    )
    ax.scatter(
        train_p_01[y_train == 1],
        np.ones(sum(y_train == 1)) * (-offset),
        c="blue",
        s=60,
        marker="o",
        alpha=0.7,
        label="Class 1 (Train)",
    )

    # Test data
    if X_test is not None:
        test_p_01 = model.get_quantum_features(X_test)[:, 0]
        ax.scatter(
            test_p_01[y_test == 0],
            np.zeros(sum(y_test == 0)) * offset,
            c="red",
            s=80,
            marker="s",
            alpha=0.7,
            label="Class 0 (Test)",
        )
        ax.scatter(
            test_p_01[y_test == 1],
            np.zeros(sum(y_test == 1)) * (-offset),
            c="blue",
            s=80,
            marker="s",
            alpha=0.7,
            label="Class 1 (Test)",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel(r"$p_{01}$")
    ax.set_yticks([])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()


def plot_figure_5(
    all_train_accuracies, all_test_accuracies, range_num_layers, COLORS=None
):
    """
    Plot the training and test accuracy as a function of the number of layers using box plots.

    Parameters:
    - all_train_accuracies: list of lists, each sublist contains multiple train accuracies for a layer
    - all_test_accuracies: list of lists, each sublist contains multiple test accuracies for a layer
    - range_num_layers: list of layer counts (e.g., [1, 2, 3, 4, 5])
    - COLORS: optional list of two colors for 'Train' and 'Test'
    """
    # Prepare the dataframe
    data = []
    for i, num_layers in enumerate(range_num_layers):
        for acc in all_train_accuracies[i]:
            data.append({"Layers": num_layers, "Accuracy": acc, "Type": "Train"})
        for acc in all_test_accuracies[i]:
            data.append({"Layers": num_layers, "Accuracy": acc, "Type": "Test"})
    df = pd.DataFrame(data)

    # Default colors if none provided
    if COLORS is None:
        COLORS = ["#1f77b4", "#ff7f0e"]  # Blue for train, orange for test

    # Plot using boxplot
    plt.figure(figsize=(10, 6))
    # df["Layers"] = df["Layers"].astype(int)  # or float if needed
    # df["Accuracy"] = df["Accuracy"].astype(float)
    df["Layers"] = pd.to_numeric(df["Layers"], errors="coerce")
    df["Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")
    sns.boxplot(
        data=df,
        x="Layers",
        y="Accuracy",
        hue="Type",
        palette=COLORS,
        orientation="vertical",
    )
    plt.xlabel(r"Number of layers")
    plt.ylabel(r"Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
