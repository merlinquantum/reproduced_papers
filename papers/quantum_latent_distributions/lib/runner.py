"""Shared-runtime entry point for the quantum-latent-distributions reproduction.

The runner only dispatches: every study lives in :mod:`lib.experiments`, and the
photonic, GAN and metric building blocks live in their own modules so the
notebook can import them directly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from lib.experiments import EXPERIMENTS

logger = logging.getLogger(__name__)

__all__ = ["train_and_evaluate"]


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run the configured experiment and write its artifacts into ``run_dir``.

    Parameters
    ----------
    cfg : dict
        Configuration resolved by the shared runtime (defaults, config overlay,
        global CLI flags, then paper CLI flags).
    run_dir : pathlib.Path
        Timestamped run directory created by the shared runtime.

    Returns
    -------
    dict
        The experiment summary, also written to ``run_dir``.

    Raises
    ------
    ValueError
        If ``cfg["experiment"]`` is not one of the four supported studies.
    """
    experiment = cfg.get("experiment")
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"unknown experiment: {experiment!r} (expected one of {sorted(EXPERIMENTS)})"
        )

    # The shared runtime resolves `dtype` into a (label, torch.dtype) pair.
    dtype_entry = cfg.get("dtype")
    if isinstance(dtype_entry, (list, tuple)) and len(dtype_entry) == 2:
        torch.set_default_dtype(dtype_entry[1])

    threads = cfg.get("torch_threads")
    if threads:
        torch.set_num_threads(int(threads))

    logger.info("Running experiment %r", experiment)
    summary = EXPERIMENTS[experiment](cfg, run_dir)

    (run_dir / "done.json").write_text(
        json.dumps({"experiment": experiment, "status": "complete"}, indent=2),
        encoding="utf-8",
    )
    return summary
