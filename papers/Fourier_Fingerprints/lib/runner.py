import logging
import re
from pathlib import Path
from typing import Any

from fourier_1D import main as main_1d
from Fourier_Fingerprints.lib.fourier_2D import main as main_2d

logger = logging.getLogger(__name__)


def _run_experiment(cfg: dict[str, Any], run_dir: Path):
    dim = cfg.get("dim", cfg.get("dimension"))
    circuits = cfg["circuits"]
    facteur_echelle = cfg["encoding"]
    name = cfg["graph_name"]

    if dim == 1:
        main_1d(
            circuits=circuits,
            facteur_echelle=facteur_echelle,
            name=name,
            rundir=run_dir,
        )
    elif dim == 2:
        main_2d(
            circuits=circuits,
            facteur_echelle=facteur_echelle,
            name=name,
            rundir=run_dir,
        )
    else:
        raise ValueError(f"Unsupported Fourier fingerprint dimension: {dim!r}")


def train_and_evaluate(cfg: dict[str, Any], run_dir):
    run_dir = Path(run_dir)
    _run_experiment(cfg, run_dir)
    logger.info("Finished. Artifacts in: %s", run_dir)


__all__ = ["train_and_evaluate"] 