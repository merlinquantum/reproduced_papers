"""Digitise the 15x15 kernel heatmaps of Figs. 3e and 4 into `substitute-data`.

The authors' kernel matrices are unpublished. This reads the arXiv HTML figure
PNGs cached under ``data/adaptive_boson_sampling/sources/figures/``, maps each
cell's centre colour to a value through the panel's own colorbar (nearest RGB on
the bar, linear in the tick positions), and writes:

- ``data/adaptive_boson_sampling/digitized/figure_kernels.npz`` (used by runs),
- ``papers/adaptive_boson_sampling/results/digitized_kernels/<key>.csv`` and
  ``manifest.json`` (curated, small, committed).

Pixel boxes and tick rows below were measured on the cached PNGs (Figure3.png
2015x1062, Figure4.png 2035x1324); the script fails if the image sizes differ.
Expected error is about one colorbar step plus JPEG-free PNG antialiasing,
reported per kernel as diagonal deviation from 1 and asymmetry.

Usage (from the repository root)::

    python papers/adaptive_boson_sampling/utils/digitize_figures.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_lib.data_paths import paper_data_dir  # noqa: E402

logger = logging.getLogger(__name__)

# (figure file, expected size, heatmap boxes, colorbar) per panel row.
# Heatmap box: (key, x_left, x_right, y_top, y_bottom) of the cell area.
# Colorbar: x range of its interior, y extent, and (tick_y, tick_value) pairs.
PANELS = [
    {
        "figure": "Figure4.png",
        "size": (2035, 1324),
        "heatmaps": [
            ("fig4a_B2_simulated", 1135, 1486, 54, 404),
            ("fig4a_B2_experimental", 1558, 1909, 54, 404),
        ],
        "colorbar": {
            "x": (1960, 1972),
            "y": (51, 406),
            "ticks": [(52.5, 1.0), (129.5, 0.8), (206.5, 0.6), (283.5, 0.4), (360.5, 0.2)],
        },
    },
    {
        "figure": "Figure4.png",
        "size": (2035, 1324),
        "heatmaps": [
            ("fig4b_B3_simulated", 1135, 1486, 497, 847),
            ("fig4b_B3_experimental", 1558, 1909, 497, 847),
        ],
        "colorbar": {
            "x": (1960, 1972),
            "y": (497, 847),
            "ticks": [(495.5, 1.0), (571.5, 0.8), (647.5, 0.6), (723.5, 0.4), (800.5, 0.2)],
        },
    },
    {
        "figure": "Figure3.png",
        "size": (2015, 1062),
        "heatmaps": [
            ("fig3e_B1_theoretical", 1093, 1470, 663, 1038),
            ("fig3e_B1_experimental", 1540, 1917, 663, 1038),
        ],
        "colorbar": {
            "x": (1943, 1959),
            "y": (663, 1038),
            "ticks": [
                (661.5, 1.0), (709.0, 0.9), (757.0, 0.8), (804.5, 0.7),
                (852.0, 0.6), (900.0, 0.5), (947.5, 0.4), (995.5, 0.3),
            ],
        },
    },
]  # fmt: skip

N_CELLS = 15


def colorbar_lookup(image: np.ndarray, colorbar: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (colours, values) sampled along the colorbar interior."""
    x_left, x_right = colorbar["x"]
    y_top, y_bottom = colorbar["y"]
    tick_rows = np.array([row for row, _ in colorbar["ticks"]])
    tick_values = np.array([value for _, value in colorbar["ticks"]])
    slope, intercept = np.polyfit(tick_rows, tick_values, 1)
    rows = np.arange(y_top + 2, y_bottom - 1)
    colours = image[rows, x_left:x_right].mean(axis=1)
    return colours, slope * rows + intercept


def digitize_heatmap(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    colours: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Return the 15x15 values of one heatmap from cell-centre colours."""
    x_left, x_right, y_top, y_bottom = box
    cell_width = (x_right - x_left) / N_CELLS
    cell_height = (y_bottom - y_top) / N_CELLS
    kernel = np.empty((N_CELLS, N_CELLS))
    for row in range(N_CELLS):
        for column in range(N_CELLS):
            centre_x = int(x_left + (column + 0.5) * cell_width)
            centre_y = int(y_top + (row + 0.5) * cell_height)
            patch = image[centre_y - 3 : centre_y + 4, centre_x - 3 : centre_x + 4]
            colour = np.median(patch.reshape(-1, 3), axis=0)
            distance = np.sum((colours - colour) ** 2, axis=1)
            kernel[row, column] = values[int(np.argmin(distance))]
    return kernel


def main(argv: list[str] | None = None) -> int:
    """Digitise every configured heatmap and write the archive and CSVs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=None, help="Override the data root")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    paper_dir = paper_data_dir("adaptive_boson_sampling", data_root=args.data_root)
    figures_dir = paper_dir / "sources" / "figures"
    archive_path = paper_dir / "digitized" / "figure_kernels.npz"
    results_dir = PROJECT_DIR / "results" / "digitized_kernels"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    kernels: dict[str, np.ndarray] = {}
    manifest: dict[str, dict] = {}
    for panel in PANELS:
        source = figures_dir / panel["figure"]
        if not source.exists():
            raise FileNotFoundError(f"missing cached figure {source}")
        image_file = Image.open(source)
        if image_file.size != panel["size"]:
            raise ValueError(
                f"{source} is {image_file.size}, calibration expects {panel['size']}"
            )
        image = np.asarray(image_file.convert("RGB")).astype(float)
        colours, values = colorbar_lookup(image, panel["colorbar"])
        for key, *box in panel["heatmaps"]:
            kernel = digitize_heatmap(image, tuple(box), colours, values)
            kernels[key] = kernel
            diagonal_error = float(np.max(np.abs(np.diag(kernel) - 1.0)))
            asymmetry = float(np.max(np.abs(kernel - kernel.T)))
            manifest[key] = {
                "source": f"data/adaptive_boson_sampling/sources/figures/{panel['figure']}",
                "max_abs_diagonal_minus_one": diagonal_error,
                "max_abs_asymmetry": asymmetry,
                "colorbar_value_range": [float(values.min()), float(values.max())],
            }
            np.savetxt(results_dir / f"{key}.csv", kernel, delimiter=",", fmt="%.4f")
            logger.info(
                "digitised %s | diag_err=%.3f | asymmetry=%.3f",
                key,
                diagonal_error,
                asymmetry,
            )

    np.savez(archive_path, **kernels)
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("wrote %s and %s", archive_path, results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
