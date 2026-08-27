#!/usr/bin/env python3
"""Generate the paired RF-RQKS spectrogram dataset from measured LTE IQ files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse dataset-generation command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed paths for input, output, and configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root containing measured LTE .bin files",
    )
    parser.add_argument(
        "--output-root", type=Path, required=True, help="New dataset output directory"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PAPER_ROOT / "configs" / "dataset.json",
        help="Dataset JSON configuration",
    )
    return parser.parse_args()


def main() -> None:
    """Build the configured dataset."""
    sys.path.insert(0, str(PAPER_ROOT))
    from lib.dataset import DatasetConfig, build_dataset

    arguments = parse_arguments()
    config = DatasetConfig.from_json(arguments.config)
    build_dataset(arguments.input_root, arguments.output_root, config)


if __name__ == "__main__":
    main()
