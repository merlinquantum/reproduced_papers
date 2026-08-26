"""Tests for plotting examples from a processed RF-RQKS dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from utils.plot_data_examples import plot_examples, select_example_rows


def build_metadata_row(
    sample_index: int,
    pair_index: int,
    label: int,
    anomaly_type: str = "",
    jsr_db: str = "",
) -> dict[str, str]:
    """Create the metadata fields consumed by the plotting utility.

    Parameters
    ----------
    sample_index : int
        Spectrogram array index.
    pair_index : int
        Shared normal/anomalous source-pair index.
    label : int
        Binary sample label.
    anomaly_type : str
        Anomaly family. Default value is an empty string.
    jsr_db : str
        Jamming-to-signal ratio. Default value is an empty string.

    Returns
    -------
    dict[str, str]
        String-valued metadata row.
    """
    return {
        "sample_index": str(sample_index),
        "pair_index": str(pair_index),
        "label": str(label),
        "anomaly_type": anomaly_type,
        "jsr_db": jsr_db,
    }


def test_select_examples_pairs_normal_with_chirp() -> None:
    metadata_rows = [
        build_metadata_row(0, 0, 0),
        build_metadata_row(1, 0, 1, "barrage", "-2"),
        build_metadata_row(2, 1, 0),
        build_metadata_row(3, 1, 1, "chirp", "5"),
        build_metadata_row(4, 2, 0),
        build_metadata_row(5, 2, 1, "frequency_hopping", "0"),
    ]

    selected_rows = select_example_rows(metadata_rows)

    assert selected_rows["normal"]["sample_index"] == "2"
    assert selected_rows["chirp"]["sample_index"] == "3"
    assert selected_rows["barrage"]["sample_index"] == "1"
    assert selected_rows["frequency_hopping"]["sample_index"] == "5"


def test_plot_examples_writes_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    spectrograms = rng.standard_normal((6, 12, 10)).astype(np.float32)
    selected_rows = {
        "normal": build_metadata_row(0, 0, 0),
        "chirp": build_metadata_row(1, 0, 1, "chirp", "5"),
        "barrage": build_metadata_row(3, 1, 1, "barrage", "-2"),
        "frequency_hopping": build_metadata_row(5, 2, 1, "frequency_hopping", "0"),
    }
    dataset_config = {
        "bandwidth_hz": 48_000_000.0,
        "iq_points_per_sample": 1_300_000,
        "sampling_rate_hz": 61_440_000.0,
    }
    output_path = tmp_path / "examples.png"

    plot_examples(
        spectrograms,
        selected_rows,
        dataset_config,
        "train",
        output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
