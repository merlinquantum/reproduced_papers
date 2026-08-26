"""Expand paired RF-RQKS spectrogram datasets for ablation experiments."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

_AUGMENTATION_FIELDS = (
    "source_pair_index",
    "augmentation_index",
    "frequency_shift",
    "time_shift",
)


def _translate_spectrogram(
    spectrogram: np.ndarray, frequency_shift: int, time_shift: int
) -> np.ndarray:
    """Translate one spectrogram while preserving its dimensions.

    Parameters
    ----------
    spectrogram : numpy.ndarray
        Two-dimensional frequency-by-time log-magnitude spectrogram.
    frequency_shift : int
        Integer translation along the frequency axis.
    time_shift : int
        Integer translation along the time axis.

    Returns
    -------
    numpy.ndarray
        Translated spectrogram with the same shape and dtype.
    """
    translated = ndimage.shift(
        spectrogram,
        shift=(frequency_shift, time_shift),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return translated.astype(spectrogram.dtype, copy=False)


def _read_split_metadata(metadata_path: Path) -> list[dict[str, str]]:
    """Read metadata rows for one processed dataset split."""
    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    if not rows:
        raise ValueError(f"Metadata is empty: {metadata_path}")
    return rows


def _validate_split(
    split_root: Path, spectrograms: np.ndarray, labels: np.ndarray, rows: list[dict[str, str]]
) -> dict[int, tuple[int, int]]:
    """Validate pair layout and return pair IDs mapped to row indices."""
    if spectrograms.ndim != 3:
        raise ValueError(f"Spectrograms must be three-dimensional: {split_root}")
    if labels.shape != (spectrograms.shape[0],) or len(rows) != spectrograms.shape[0]:
        raise ValueError(f"Spectrogram, label, and metadata lengths disagree: {split_root}")

    pair_rows: dict[int, list[int]] = {}
    for row_index, row in enumerate(rows):
        pair_rows.setdefault(int(row["pair_index"]), []).append(row_index)
    validated_pairs: dict[int, tuple[int, int]] = {}
    for pair_index, row_indices in pair_rows.items():
        if len(row_indices) != 2:
            raise ValueError(f"Pair {pair_index} does not contain exactly two rows")
        first, second = row_indices
        if labels[first] != 0 or labels[second] != 1:
            raise ValueError(f"Pair {pair_index} must contain labels [0, 1]")
        validated_pairs[pair_index] = (first, second)
    return validated_pairs


def _augment_split(
    input_split_root: Path,
    output_split_root: Path,
    target_pair_count: int,
    seed: int,
    maximum_frequency_shift_fraction: float,
    maximum_time_shift_fraction: float,
) -> None:
    """Expand one split by sampling and translating complete source pairs."""
    spectrograms = np.load(input_split_root / "spectrograms.npy", mmap_mode="r")
    labels = np.load(input_split_root / "labels.npy", mmap_mode="r")
    rows = _read_split_metadata(input_split_root / "metadata.csv")
    pair_rows = _validate_split(input_split_root, spectrograms, labels, rows)
    if target_pair_count < len(pair_rows):
        raise ValueError(
            f"Target pair count {target_pair_count} is smaller than source count "
            f"{len(pair_rows)} for {input_split_root.name}"
        )
    if not 0 <= maximum_frequency_shift_fraction < 1:
        raise ValueError("maximum_frequency_shift_fraction must be in [0, 1)")
    if not 0 <= maximum_time_shift_fraction < 1:
        raise ValueError("maximum_time_shift_fraction must be in [0, 1)")

    output_split_root.mkdir(parents=True)
    output_spectrograms = np.lib.format.open_memmap(
        output_split_root / "spectrograms.npy",
        mode="w+",
        dtype=spectrograms.dtype,
        shape=(2 * target_pair_count, *spectrograms.shape[1:]),
    )
    output_labels = np.lib.format.open_memmap(
        output_split_root / "labels.npy",
        mode="w+",
        dtype=labels.dtype,
        shape=(2 * target_pair_count,),
    )
    source_pair_indices = np.asarray(sorted(pair_rows), dtype=np.int64)
    rng = np.random.default_rng(seed)
    frequency_limit = int(round(spectrograms.shape[1] * maximum_frequency_shift_fraction))
    time_limit = int(round(spectrograms.shape[2] * maximum_time_shift_fraction))

    metadata_fields = list(rows[0])
    for field in _AUGMENTATION_FIELDS:
        if field not in metadata_fields:
            metadata_fields.append(field)
    metadata_path = output_split_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=metadata_fields)
        writer.writeheader()
        for output_pair_index in range(target_pair_count):
            source_pair_index = int(rng.choice(source_pair_indices))
            normal_index, anomaly_index = pair_rows[source_pair_index]
            frequency_shift = int(rng.integers(-frequency_limit, frequency_limit + 1))
            time_shift = int(rng.integers(-time_limit, time_limit + 1))
            output_normal_index = 2 * output_pair_index
            output_anomaly_index = output_normal_index + 1
            output_spectrograms[output_normal_index] = _translate_spectrogram(
                np.asarray(spectrograms[normal_index]), frequency_shift, time_shift
            )
            output_spectrograms[output_anomaly_index] = _translate_spectrogram(
                np.asarray(spectrograms[anomaly_index]), frequency_shift, time_shift
            )
            output_labels[output_normal_index : output_anomaly_index + 1] = (0, 1)

            for output_index, source_index in (
                (output_normal_index, normal_index),
                (output_anomaly_index, anomaly_index),
            ):
                metadata_row: dict[str, Any] = dict(rows[source_index])
                metadata_row.update(
                    sample_index=output_index,
                    pair_index=output_pair_index,
                    source_pair_index=source_pair_index,
                    augmentation_index=output_pair_index,
                    frequency_shift=frequency_shift,
                    time_shift=time_shift,
                )
                writer.writerow(metadata_row)
    output_spectrograms.flush()
    output_labels.flush()


def augment_processed_dataset(
    input_root: Path,
    output_root: Path,
    train_pair_count: int = 10_800,
    test_pair_count: int = 4_062,
    seed: int = 42,
    maximum_frequency_shift_fraction: float = 0.1,
    maximum_time_shift_fraction: float = 0.1,
) -> None:
    """Create a paper-sized paired spectrogram dataset by deterministic augmentation.

    Parameters
    ----------
    input_root : pathlib.Path
        Processed RF-RQKS dataset containing ``train`` and ``test`` splits.
    output_root : pathlib.Path
        New output directory. Existing directories are not overwritten.
    train_pair_count : int
        Number of normal/anomaly pairs in the output training split. Default is 10800.
    test_pair_count : int
        Number of normal/anomaly pairs in the output test split. Default is 4062.
    seed : int
        Seed controlling source-pair sampling and translations. Default is 42.
    maximum_frequency_shift_fraction : float
        Maximum absolute frequency translation as a fraction of image height.
        Default is 0.1.
    maximum_time_shift_fraction : float
        Maximum absolute time translation as a fraction of image width. Default is 0.1.

    Raises
    ------
    FileExistsError
        If ``output_root`` or its temporary build directory already exists.
    ValueError
        If the source dataset is malformed or target counts are invalid.
    """
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")
    if train_pair_count <= 0 or test_pair_count <= 0:
        raise ValueError("Target pair counts must be positive")
    temporary_root = output_root.with_name(f".{output_root.name}.building-{os.getpid()}")
    if temporary_root.exists():
        raise FileExistsError(f"Temporary build directory already exists: {temporary_root}")
    temporary_root.mkdir(parents=True)
    _augment_split(
        input_root / "train",
        temporary_root / "train",
        train_pair_count,
        seed,
        maximum_frequency_shift_fraction,
        maximum_time_shift_fraction,
    )
    _augment_split(
        input_root / "test",
        temporary_root / "test",
        test_pair_count,
        seed + 1,
        maximum_frequency_shift_fraction,
        maximum_time_shift_fraction,
    )
    manifest = {
        "format_version": 1,
        "source_root": str(input_root.resolve()),
        "train_samples": 2 * train_pair_count,
        "test_samples": 2 * test_pair_count,
        "augmentation": "paired spectrogram translations with replacement",
        "seed": seed,
        "maximum_frequency_shift_fraction": maximum_frequency_shift_fraction,
        "maximum_time_shift_fraction": maximum_time_shift_fraction,
        "notes": [
            "Each normal/anomaly pair receives the same translation.",
            "source_pair_index records the original pair used for each augmented pair.",
            "This matches the paper row counts but does not create independent LTE captures.",
        ],
    }
    (temporary_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    temporary_root.rename(output_root)
