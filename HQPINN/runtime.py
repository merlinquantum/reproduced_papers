from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import torch


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    level_name = os.getenv("HQPINN_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_run_banner(config: dict[str, Any]) -> None:
    LOGGER.info(
        "Starting run: experiment=%s mode=%s backend=%s",
        config.get("experiment"),
        config.get("mode"),
        config.get("backend"),
    )
    LOGGER.debug(
        "Resolved config:\n%s",
        json.dumps(config, indent=2, sort_keys=True),
    )


def seed_everything(seed: int = 0) -> None:
    """Seed Python, PyTorch, and NumPy RNGs when available."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)
