from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_fake_progress(csv_path: str | Path) -> np.ndarray:
    """Load fake_progress.csv as a 2D array (samples x pixels)."""
    path = Path(csv_path)
    return np.loadtxt(path, delimiter=",")


def show_sample(csv_path: str | Path, index: int = -1, image_size: int = 8) -> None:
    """Display a single generated sample from fake_progress.csv."""
    data = load_fake_progress(csv_path)
    sample = data[index].reshape(image_size, image_size)
    plt.imshow(sample, cmap="gray")
    plt.axis("off")
    plt.show()


def show_grid(
    csv_path: str | Path,
    count: int = 16,
    image_size: int = 8,
    cols: int = 4,
) -> None:
    """Display a grid of generated samples from fake_progress.csv."""
    data = load_fake_progress(csv_path)
    count = min(count, len(data))
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_2d(axes)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")
        if idx < count:
            sample = data[idx].reshape(image_size, image_size)
            ax.imshow(sample, cmap="gray")
    plt.tight_layout()
    plt.show()


csv_path = Path(
    "/Users/cassandrenotton/Documents/projects/QML_project/fork_reproduced_papers/reproduced_papers/papers/photonic_QGAN/outdir/run_20260120-150550/digits/config_1/run_1/fake_progress.csv"
)
show_sample(csv_path, index=-1, image_size=8)
show_grid(csv_path, count=16, image_size=8, cols=4)
