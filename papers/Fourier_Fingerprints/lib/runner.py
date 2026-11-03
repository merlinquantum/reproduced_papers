"""Dispatch a Fourier Fingerprints configuration to the shared implementation."""

import logging
from pathlib import Path
from typing import Any

from lib.fourier import main as run_fingerprints

logger = logging.getLogger(__name__)


def _run_experiment(cfg: dict[str, Any], run_dir: Path):
    run_fingerprints(
        dimension=cfg.get("dim", cfg.get("dimension")),
        circuits=cfg["circuits"],
        encoding=cfg["encoding"],
        name=cfg["graph_name"],
        rundir=run_dir,
    )


def train_and_evaluate(cfg: dict[str, Any], run_dir):
    run_dir = Path(run_dir)
    _run_experiment(cfg, run_dir)
    logger.info("Finished. Artifacts in: %s", run_dir)


__all__ = ["train_and_evaluate"]
