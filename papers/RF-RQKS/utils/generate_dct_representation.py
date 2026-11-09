#!/usr/bin/env python3
"""Cache an unnormalized low-index RF-RQKS DCT representation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse DCT-cache command-line arguments.

    Returns
    -------
    argparse.Namespace
        Input dataset, output cache, and representation configuration paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Processed RF-RQKS spectrogram dataset",
    )
    parser.add_argument(
        "--output-root", type=Path, required=True, help="New DCT cache directory"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PAPER_ROOT / "configs" / "dct64x64.json",
        help="DCT representation JSON configuration",
    )
    return parser.parse_args()


def main() -> None:
    """Build the configured DCT representation cache."""
    sys.path.insert(0, str(PAPER_ROOT))
    from lib.representations import (
        DctRepresentationConfig,
        build_dct_representation_cache,
    )

    arguments = parse_arguments()
    config = DctRepresentationConfig.from_json(arguments.config)
    build_dct_representation_cache(arguments.input_root, arguments.output_root, config)


if __name__ == "__main__":
    main()
