"""Figures written by runs: kernel heatmaps in the style of Figs. 3-4."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def save_kernel_heatmaps(
    kernels: dict[str, np.ndarray], path: Path, title: str
) -> None:
    """Save side-by-side viridis heatmaps with 1-based outcome ticks (paper style)."""
    count = len(kernels)
    figure, axes = plt.subplots(1, count, figsize=(4.2 * count, 4), squeeze=False)
    for axis, (name, kernel) in zip(axes[0], kernels.items()):
        image = axis.imshow(kernel, cmap="viridis", vmin=0.0, vmax=1.0)
        size = kernel.shape[0]
        axis.set_xticks(range(size), [str(i + 1) for i in range(size)], fontsize=7)
        axis.set_yticks(range(size), [str(i + 1) for i in range(size)], fontsize=7)
        axis.set_title(name, fontsize=9)
    figure.colorbar(image, ax=axes[0].tolist(), shrink=0.8)
    figure.suptitle(title, fontsize=10)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
