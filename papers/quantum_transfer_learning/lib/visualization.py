"""
Visualization Utilities
=======================

Plotting functions for results visualization, matching figures from the paper.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def plot_spiral_classification(
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        accuracy: float,
        title: str = "Spiral Classification",
        save_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        ax: Optional[plt.Axes] = None
):
    """Plot 2D spiral classification results.

    Recreates Figure 2 from the paper showing decision boundaries
    and data points.

    Args:
        model: Trained classifier
        X_train, y_train: Training data
        X_test, y_test: Test data
        accuracy: Test accuracy
        title: Plot title
        save_path: Path to save figure
        device: Torch device
        ax: Optional matplotlib axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        own_figure = True
    else:
        own_figure = False

    # Create meshgrid for decision boundary
    x_min, x_max = -1.2, 1.2
    y_min, y_max = -1.2, 1.2
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Predict on grid
    model.eval()
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    with torch.no_grad():
        grid_tensor = torch.tensor(grid_points).to(device)
        outputs = model(grid_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()

    Z = preds.reshape(xx.shape)

    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)

    # Plot training points (pale)
    ax.scatter(
        X_train[y_train == 0, 0], X_train[y_train == 0, 1],
        c='red', alpha=0.3, s=20, label='Class 0 (train)'
    )
    ax.scatter(
        X_train[y_train == 1, 0], X_train[y_train == 1, 1],
        c='blue', alpha=0.3, s=20, label='Class 1 (train)'
    )

    # Plot test points (sharp)
    ax.scatter(
        X_test[y_test == 0, 0], X_test[y_test == 0, 1],
        c='red', alpha=1.0, s=40, edgecolors='black', linewidths=0.5
    )
    ax.scatter(
        X_test[y_test == 1, 0], X_test[y_test == 1, 1],
        c='blue', alpha=1.0, s=40, edgecolors='black', linewidths=0.5
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)

    # Add accuracy annotation
    ax.text(
        0.95, 0.05, f'.{int(accuracy * 100):02d}',
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='bottom',
        horizontalalignment='right'
    )

    if own_figure:
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_training_curves(
        history: Dict[str, List[float]],
        title: str = "Training Progress",
        save_path: Optional[str] = None
):
    """Plot training loss and accuracy curves.

    Args:
        history: Dictionary with train_loss, train_acc, test_loss, test_acc
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], 'b-', label='Train')
    ax1.plot(epochs, history["test_loss"], 'r-', label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], 'b-', label='Train')
    ax2.plot(epochs, history["test_acc"], 'r-', label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_image_predictions(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        class_names: List[str],
        n_images: int = 4,
        title: str = "Image Classification",
        save_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        axes: Optional[List[plt.Axes]] = None
):
    """Plot sample images with predictions.

    Recreates Figure 3 from the paper showing classified images.

    Args:
        model: Trained classifier
        dataloader: Data loader with images
        class_names: List of class names
        n_images: Number of images to show
        title: Plot title
        save_path: Path to save figure
        device: Torch device
        axes: Optional list of matplotlib axes to plot on (length must match n_images)
    """
    model.eval()

    # Get batch of images
    images, labels = next(iter(dataloader))
    images = images[:n_images]
    labels = labels[:n_images]

    # Make predictions
    with torch.no_grad():
        images_device = images.to(device)
        outputs = model(images_device)
        preds = outputs.argmax(dim=1).cpu()

    # Create figure if axes not provided
    if axes is None:
        fig, axes = plt.subplots(1, n_images, figsize=(3 * n_images, 3))
        own_figure = True
    else:
        own_figure = False

    if n_images == 1:
        axes = [axes]

    # Denormalize for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, ax in enumerate(axes):
        if i >= len(images):
            ax.axis('off')
            continue

        img = images[i].numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        pred_label = class_names[preds[i]]
        ax.set_title(f'[{pred_label}]')
        ax.axis('off')

    if own_figure:
        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_comparison(
        results_quantum: Dict[str, Any],
        results_classical: Dict[str, Any],
        title: str = "Quantum vs Classical",
        save_path: Optional[str] = None
):
    """Plot comparison between quantum and classical models.

    Args:
        results_quantum: Quantum model results
        results_classical: Classical model results
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Extract histories
    q_hist = results_quantum["history"]
    c_hist = results_classical["history"]

    epochs_q = range(1, len(q_hist["test_acc"]) + 1)
    epochs_c = range(1, len(c_hist["test_acc"]) + 1)

    # Test accuracy comparison
    ax1.plot(epochs_q, q_hist["test_acc"], 'b-', label='Quantum', linewidth=2)
    ax1.plot(epochs_c, c_hist["test_acc"], 'r--', label='Classical', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Final accuracy bar chart
    final_accs = [
        results_classical["best_accuracy"],
        results_quantum["best_accuracy"]
    ]
    labels = ['Classical', 'Quantum']
    colors = ['red', 'blue']

    bars = ax2.bar(labels, final_accs, color=colors, alpha=0.7)
    ax2.set_ylabel('Best Test Accuracy')
    ax2.set_title('Final Comparison')
    ax2.set_ylim(0, 1.1)

    # Add value labels
    for bar, acc in zip(bars, final_accs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{acc:.3f}', ha='center', fontsize=12
        )

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
