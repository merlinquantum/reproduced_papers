#!/usr/bin/env python3
"""Expand a processed RF-RQKS dataset to the paper's ablation row counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse augmentation command-line arguments.

    Returns
    -------
    argparse.Namespace
        Source, destination, target-count, and augmentation parameters.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-pairs", type=int, default=10_800)
    parser.add_argument("--test-pairs", type=int, default=4_062)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frequency-shift", type=float, default=0.1)
    parser.add_argument("--max-time-shift", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    """Build the augmented processed dataset."""
    sys.path.insert(0, str(PAPER_ROOT))
    from lib.augmentation import augment_processed_dataset

    arguments = parse_arguments()
    augment_processed_dataset(
        arguments.input_root,
        arguments.output_root,
        train_pair_count=arguments.train_pairs,
        test_pair_count=arguments.test_pairs,
        seed=arguments.seed,
        maximum_frequency_shift_fraction=arguments.max_frequency_shift,
        maximum_time_shift_fraction=arguments.max_time_shift,
    )


if __name__ == "__main__":
    main()
