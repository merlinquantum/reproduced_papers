"""Tests for RF-RQKS paired spectrogram augmentation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from lib.augmentation import augment_processed_dataset


def _write_source_dataset(root: Path) -> None:
    for split, pair_count in (("train", 2), ("test", 1)):
        split_root = root / split
        split_root.mkdir(parents=True)
        spectrograms = np.arange(pair_count * 2 * 6 * 8, dtype=np.float32).reshape(
            pair_count * 2, 6, 8
        )
        np.save(split_root / "spectrograms.npy", spectrograms)
        np.save(split_root / "labels.npy", np.tile([0, 1], pair_count))
        with (split_root / "metadata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=["sample_index", "pair_index", "label"]
            )
            writer.writeheader()
            for sample_index in range(pair_count * 2):
                writer.writerow(
                    {
                        "sample_index": sample_index,
                        "pair_index": sample_index // 2,
                        "label": sample_index % 2,
                    }
                )


def test_augmentation_reaches_target_counts_and_preserves_pairs(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "augmented"
    _write_source_dataset(source_root)

    augment_processed_dataset(
        source_root, output_root, train_pair_count=5, test_pair_count=3
    )

    for split, pair_count in (("train", 5), ("test", 3)):
        split_root = output_root / split
        assert np.load(split_root / "spectrograms.npy").shape == (2 * pair_count, 6, 8)
        assert np.array_equal(
            np.load(split_root / "labels.npy"), np.tile([0, 1], pair_count)
        )
        with (split_root / "metadata.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        for pair_index in range(pair_count):
            pair_rows = rows[2 * pair_index : 2 * pair_index + 2]
            assert [row["label"] for row in pair_rows] == ["0", "1"]
            assert len({row["source_pair_index"] for row in pair_rows}) == 1
            assert (
                pair_rows[0]["pair_index"]
                == pair_rows[1]["pair_index"]
                == str(pair_index)
            )
