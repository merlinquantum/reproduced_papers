"""Build the prepared datasets used by the named configs, and report their manifests.

Usage (from the repository root)::

    python papers/photonic_MMD_generative/utils/prepare_datasets.py \
        --configs fig3b_sushi_original fig4a_pairwise_sushi_original [--data-root data]

Each config is merged onto ``configs/defaults.json``; its ``dataset`` section is
built (or loaded) by ``lib.data.load_dataset``, which writes
``<data_root>/photonic_MMD_generative/prepared/<dataset_id>/``. Prints nothing
but the log; the manifests are the evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_DIR))

from lib import data  # noqa: E402
from lib.experiment_logging import configure_logging  # noqa: E402

from runtime_lib.config import deep_update, load_config  # noqa: E402
from runtime_lib.data_paths import resolve_data_root  # noqa: E402

logger = logging.getLogger("prepare_datasets")


def main() -> None:
    """Build the datasets of the given configs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config names under configs/ (without .json)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Shared data root (default: DATA_DIR or <repo>/data)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file (default: <data_root>/photonic_MMD_generative/prepared/prepare.log)",
    )
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    log_file = (
        Path(args.log_file)
        if args.log_file
        else data_root / data.PAPER_DATA_DIRNAME / "prepared" / "prepare.log"
    )
    configure_logging(log_file)
    defaults = load_config(PAPER_DIR / "configs" / "defaults.json")
    for name in args.configs:
        cfg = deep_update(
            json.loads(json.dumps(defaults)),
            load_config(PAPER_DIR / "configs" / f"{name}.json"),
        )
        cfg["data_root"] = str(data_root)
        start = time.perf_counter()
        dataset = data.load_dataset(cfg)
        manifest = {
            k: v for k, v in dataset["manifest"].items() if k not in ("gene_universe",)
        }
        logger.info(
            "DATASET_READY | config=%s | seconds=%.1f | path=%s | manifest=%s",
            name,
            time.perf_counter() - start,
            dataset["path"],
            json.dumps(manifest, default=str),
        )


if __name__ == "__main__":
    main()
