from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize samples from fake_progress.csv."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to fake_progress.csv.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Index of the sample to display (default: -1).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=8,
        help="Image side length (default: 8).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=16,
        help="Number of samples to show in grid (default: 16).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in grid (default: 4).",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Only show the single sample, skip the grid.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Only show the grid, skip the single sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_sample:
        show_sample(args.csv_path, index=args.index, image_size=args.image_size)
    if not args.no_grid:
        show_grid(
            args.csv_path,
            count=args.count,
            image_size=args.image_size,
            cols=args.cols,
        )


if __name__ == "__main__":
    main()
