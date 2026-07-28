import logging
import re
from pathlib import Path
from typing import Any

import merlin
import numpy as np
import sklearn
import torch
from lib.fourier_1D import executer_fourier_fingerprint_1d

logger = logging.getLogger(__name__)


def _run_experiment(cfg: dict[str, Any], run_dir: Path):
    for experiment in cfg["experiments"]:
        logger.info("Running experiment: %s", experiment["name"])
        if experiment["dimension"] == "1D":
            executer_fourier_fingerprint_1d(run_dir, experiment["circuit_index"], encoding=experiment["encoding"])
        elif experiment["dimension"] == "2D":
            raise NotImplementedError("2D experiments are not implemented yet.")
        else :
            raise ValueError(f"Unknown dimension: {experiment['dimension']}. Choose 1D or 2D.")


def train_and_evaluate(cfg: dict[str, Any], run_dir):
    run_dir = Path(run_dir)
    _run_experiment(cfg, run_dir)
    logger.info("Finished. Artifacts in: %s", run_dir)


__all__ = ["train_and_evaluate"]
