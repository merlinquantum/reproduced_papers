#!/usr/bin/env python3
"""Plot representative examples from a processed RF-RQKS dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PANEL_TITLES = {
    "normal": "(a) Normal Spectrogram",
    "chirp": "(b) Chirp Anomaly",
    "barrage": "(c) Barrage Jamming",
    "frequency_hopping": "(d) Frequency Hopping Jamming",
}


def parse_arguments() -> argparse.Namespace:
    """Parse processed-dataset plotting arguments.

    Returns
    -------
    argparse.Namespace
        Dataset root, split name, and output path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Processed RF-RQKS dataset root",
    )
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG. Default value is <dataset-root>/<split>_examples.png.",
    )
    return parser.parse_args()


def read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    """Read processed sample metadata.

    Parameters
    ----------
    metadata_path : pathlib.Path
        Split metadata CSV.

    Returns
    -------
    list[dict[str, str]]
        Metadata rows in file order.
    """
    with metadata_path.open(encoding="utf-8", newline="") as metadata_file:
        return list(csv.DictReader(metadata_file))


def select_example_rows(
    metadata_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Select one normal row and one row for every anomaly family.

    The normal example is paired with the selected chirp example so panels (a)
    and (b) share the same measured LTE background.

    Parameters
    ----------
    metadata_rows : list[dict[str, str]]
        Rows from one processed split's metadata file.

    Returns
    -------
    dict[str, dict[str, str]]
        Rows keyed by ``normal``, ``chirp``, ``barrage``, and
        ``frequency_hopping``.

    Raises
    ------
    ValueError
        If an anomaly family or the chirp's paired normal row is absent.
    """
    selected_rows: dict[str, dict[str, str]] = {}
    for anomaly_type in ("chirp", "barrage", "frequency_hopping"):
        selected_row = next(
            (
                row
                for row in metadata_rows
                if row["label"] == "1" and row["anomaly_type"] == anomaly_type
            ),
            None,
        )
        if selected_row is None:
            raise ValueError(f"No {anomaly_type} example exists in the selected split")
        selected_rows[anomaly_type] = selected_row

    chirp_pair_index = selected_rows["chirp"]["pair_index"]
    normal_row = next(
        (
            row
            for row in metadata_rows
            if row["label"] == "0" and row["pair_index"] == chirp_pair_index
        ),
        None,
    )
    if normal_row is None:
        raise ValueError("The selected chirp example has no paired normal row")
    return {"normal": normal_row, **selected_rows}


def plot_examples(
    spectrograms: np.ndarray,
    selected_rows: dict[str, dict[str, str]],
    dataset_config: dict[str, Any],
    split_name: str,
    output_path: Path,
) -> None:
    """Render selected processed spectrograms as a four-panel figure.

    Parameters
    ----------
    spectrograms : numpy.ndarray
        Memory-mapped frequency-by-time spectrogram array.
    selected_rows : dict[str, dict[str, str]]
        Metadata rows selected for each panel.
    dataset_config : dict
        Dataset configuration stored in the processed manifest.
    split_name : str
        Split displayed in the figure title.
    output_path : pathlib.Path
        PNG destination.

    Raises
    ------
    IndexError
        If a selected sample index is outside the spectrogram array.
    """
    figure, axes = plt.subplots(1, 4, figsize=(20, 5.2), constrained_layout=True)
    frequency_extent_mhz = float(dataset_config["bandwidth_hz"]) / 2.0 / 1e6
    duration_ms = (
        float(dataset_config["iq_points_per_sample"])
        / float(dataset_config["sampling_rate_hz"])
        * 1e3
    )
    for axis, panel_name in zip(axes, PANEL_TITLES):
        metadata_row = selected_rows[panel_name]
        sample_index = int(metadata_row["sample_index"])
        if not 0 <= sample_index < spectrograms.shape[0]:
            raise IndexError(f"Sample index {sample_index} is outside the array")
        spectrogram = np.asarray(spectrograms[sample_index])
        lower_limit, upper_limit = np.percentile(spectrogram, [1.0, 99.8])
        image = axis.imshow(
            spectrogram.T,
            origin="upper",
            aspect="auto",
            extent=[-frequency_extent_mhz, frequency_extent_mhz, duration_ms, 0.0],
            cmap="viridis",
            vmin=lower_limit,
            vmax=upper_limit,
        )
        title = PANEL_TITLES[panel_name]
        if metadata_row["jsr_db"]:
            title += f"\nJSR {float(metadata_row['jsr_db']):g} dB"
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("Frequency [MHz]")
        axis.set_ylabel("Time [ms]")
        colorbar = figure.colorbar(image, ax=axis, pad=0.02, fraction=0.046)
        colorbar.set_label("Magnitude [dB]")

    figure.suptitle(f"Processed RF-RQKS examples - {split_name} split", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Load a processed split and save its example figure."""
    arguments = parse_arguments()
    dataset_root = arguments.dataset_root.resolve()
    output_path = arguments.output or dataset_root / f"{arguments.split}_examples.png"
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    split_root = dataset_root / arguments.split
    spectrograms = np.load(split_root / "spectrograms.npy", mmap_mode="r")
    metadata_rows = read_metadata(split_root / "metadata.csv")
    selected_rows = select_example_rows(metadata_rows)
    plot_examples(
        spectrograms,
        selected_rows,
        manifest["configuration"],
        arguments.split,
        output_path,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
