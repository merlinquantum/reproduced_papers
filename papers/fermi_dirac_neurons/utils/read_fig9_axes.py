"""Read final loss values off the Fig. 9 crop by pixel position on the log axis.

Usage::

    python3 papers/fermi_dirac_neurons/utils/read_fig9_axes.py \
        data/fermi_dirac_neurons/sources/screenshots/figures/fig09_linear_vs_tanh.png [--trace]

``--trace`` samples every curve at fixed epochs instead of the final value
(x axis from the 0 and 10 000 ticks; a curve hidden by the legend or another
curve at an epoch gives None).

The four panels are cropped at fixed fractions of the image. In each panel
the y-axis spine is the darkest column in the left third; major ticks are the
dark rows just left of it. The tick labels were read from the crop by eye
(``TICK_LABELS``, top to bottom); log10(loss) is interpolated linearly between
ticks. Curve rows are taken at the right end of each curve (epoch 10 000) by
colour; the red dashed baseline in panel (d) is hidden by the legend at the
right end and is read further left (it is flat).
"""

from __future__ import annotations

import json
import sys

import numpy as np
from PIL import Image

# Exponents of the labelled major ticks, top to bottom, read from the crop.
TICK_LABELS = {
    "a_tfim_2q": [1, -3, -7, -11, -15, -19, -23, -27],
    "b_tfim_3q": [0, -3, -6, -9, -12, -15, -18, -21],
    "c_heis_2q": [0, -1, -2, -3, -4],
    "d_heis_3q": [-1, -2, -3, -4],
}
# Panel boxes (left, top, right, bottom) as fractions of the image size.
PANELS = {
    "a_tfim_2q": (0.0, 40 / 1630, 0.5, 560 / 1630),
    "b_tfim_3q": (0.5, 40 / 1630, 1.0, 560 / 1630),
    "c_heis_2q": (0.0, 720 / 1630, 0.5, 1230 / 1630),
    "d_heis_3q": (0.5, 720 / 1630, 1.0, 1230 / 1630),
}
COLOURS = {
    "tanh": lambda r, g, b: (b > 180) & (r < 80) & (g < 80),
    "linear": lambda r, g, b: (g > 90) & (r < 60) & (b < 60),
    "classical": lambda r, g, b: (r > 200) & (g < 100) & (b < 100),
}


def read_panel(pixels: np.ndarray, exponents: list[int]) -> dict:
    """Return log10 of each curve's final value in one panel."""
    brightness = pixels.sum(2)
    spine = int(np.argmax((brightness[:, : pixels.shape[1] // 3] < 400).sum(0)))
    tick_rows = np.flatnonzero(
        (brightness[:, spine - 4] < 400) & (brightness[:, spine - 2] < 400)
    )
    groups = np.split(tick_rows, np.flatnonzero(np.diff(tick_rows) > 1) + 1)
    ticks = np.array([np.mean(group) for group in groups])
    if len(ticks) != len(exponents):
        raise ValueError(f"Found {len(ticks)} ticks, expected {len(exponents)}")
    right = pixels.shape[1] // 2 + int(
        np.argmax((brightness[:, pixels.shape[1] // 2 :] < 400).sum(0))
    )
    red, green, blue = (pixels[..., channel] for channel in range(3))
    readings = {}
    for name, rule in COLOURS.items():
        mask = rule(red, green, blue)
        rows = np.flatnonzero(mask[:, right - 42 : right - 12].any(1))
        if not len(rows):
            rows = np.flatnonzero(mask[:, spine + 50 : spine + 220].any(1))
        # Ticks are evenly spaced in log10, so a linear fit also extrapolates
        # below the lowest labelled tick (np.interp would clamp there).
        slope, intercept = np.polyfit(ticks, exponents, 1)
        readings[name] = float(slope * np.mean(rows) + intercept)
    return readings


def _x_axis(brightness: np.ndarray, spine: int) -> tuple[float, float]:
    """Return the pixel columns of the epoch-0 and epoch-10 000 ticks."""
    bottom = int(
        np.max(np.flatnonzero((brightness[:, spine + 5 : spine + 400] < 400).all(1)))
    )
    tick_columns = np.flatnonzero((brightness[bottom + 2 : bottom + 5] < 400).all(0))
    groups = np.split(tick_columns, np.flatnonzero(np.diff(tick_columns) > 1) + 1)
    centres = [float(np.mean(group)) for group in groups]
    # Six labelled ticks: 0, 2000, ..., 10 000.
    if len(centres) != 6:
        raise ValueError(f"Found {len(centres)} x ticks, expected 6")
    return centres[0], centres[-1]


def trace_panel(pixels: np.ndarray, exponents: list[int], epochs: list[int]) -> dict:
    """Return log10 loss of each curve at the requested epochs (None where hidden)."""
    brightness = pixels.sum(2)
    spine = int(np.argmax((brightness[:, : pixels.shape[1] // 3] < 400).sum(0)))
    tick_rows = np.flatnonzero(
        (brightness[:, spine - 4] < 400) & (brightness[:, spine - 2] < 400)
    )
    groups = np.split(tick_rows, np.flatnonzero(np.diff(tick_rows) > 1) + 1)
    slope, intercept = np.polyfit([np.mean(group) for group in groups], exponents, 1)
    x0, x10k = _x_axis(brightness, spine)
    red, green, blue = (pixels[..., channel] for channel in range(3))
    traces = {}
    for name, rule in COLOURS.items():
        mask = rule(red, green, blue)
        values = []
        for epoch in epochs:
            column = int(round(x0 + (x10k - x0) * epoch / 10_000))
            rows = np.flatnonzero(mask[:, column - 1 : column + 2].any(1))
            values.append(
                float(slope * np.mean(rows) + intercept) if len(rows) else None
            )
        traces[name] = values
    return traces


def main() -> None:
    """Print log10 final-loss readings for the four Fig. 9 panels as JSON."""
    image = Image.open(sys.argv[1]).convert("RGB")
    width, height = image.size
    result = {}
    for panel, (left, top, right, bottom) in PANELS.items():
        box = (
            int(left * width),
            int(top * height),
            int(right * width),
            int(bottom * height),
        )
        result[panel] = read_panel(
            np.asarray(image.crop(box)).astype(int), TICK_LABELS[panel]
        )
    if len(sys.argv) > 2 and sys.argv[2] == "--trace":
        epochs = [50, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 9900]
        result = {"epochs": epochs}
        for panel, (left, top, right, bottom) in PANELS.items():
            box = (
                int(left * width),
                int(top * height),
                int(right * width),
                int(bottom * height),
            )
            result[panel] = trace_panel(
                np.asarray(image.crop(box)).astype(int), TICK_LABELS[panel], epochs
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
