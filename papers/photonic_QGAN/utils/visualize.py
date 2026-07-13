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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize generated samples from fake_progress.csv."
    )
    parser.add_argument("csv_path", type=Path, help="Path to fake_progress.csv")
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Sample index for show_sample (default: -1)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=16,
        help="Number of samples for show_grid (default: 16)",
    )
    parser.add_argument(
        "--image-size", type=int, default=8, help="Image size in pixels (default: 8)"
    )
    parser.add_argument(
        "--cols", type=int, default=4, help="Columns in the grid (default: 4)"
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "grid", "both"],
        default="both",
        help="What to display (default: both)",
    )
    args = parser.parse_args()

    if args.mode in ("sample", "both"):
        show_sample(args.csv_path, index=args.index, image_size=args.image_size)
    if args.mode in ("grid", "both"):
        show_grid(
            args.csv_path, count=args.count, image_size=args.image_size, cols=args.cols
        )


if __name__ == "__main__":
    main()
