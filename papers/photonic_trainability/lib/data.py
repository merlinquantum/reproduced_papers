"""Synthetic ensembles used as this paper's "data".

The paper uses no external dataset (LOG.md, Data Acquisition Log): every
experiment averages over Haar-random interferometers. ``build_dataloader``
yields those interferometers in batches, reproducibly from the run seed, so the
Monte Carlo code consumes them like any other dataset.

The shared data root (``DATA_DIR`` environment variable or ``--data-root``) is
still resolved, because cached reference sources live under
``<data_root>/photonic_trainability/sources/``; no experiment reads files from
it.
"""

from __future__ import annotations

import os
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .photonic_sim import haar_unitary

PAPER_NAME = "photonic_trainability"


def paper_data_dir(cfg: dict[str, Any]) -> Path:
    """Return ``<data_root>/photonic_trainability``.

    ``cfg["data_root"]`` is set by the shared runtime from ``--data-root`` or
    ``DATA_DIR``; outside the runtime the ``DATA_DIR`` variable is used.

    Raises
    ------
    KeyError
        If neither ``cfg["data_root"]`` nor ``DATA_DIR`` is set.
    """
    data_root = cfg.get("data_root") or os.environ["DATA_DIR"]
    return Path(data_root) / PAPER_NAME


class HaarUnitaryLoader:
    """Re-iterable, seeded batches of Haar-random ``M x M`` unitaries.

    Parameters
    ----------
    modes : int
        Matrix size ``M``.
    count : int
        Total number of unitaries.
    batch_size : int
        Unitaries per batch; the last batch may be smaller.
    seed_sequence : numpy.random.SeedSequence
        Seed material; iterating twice yields the same unitaries.
    """

    def __init__(
        self,
        modes: int,
        count: int,
        batch_size: int,
        seed_sequence: np.random.SeedSequence,
    ) -> None:
        if modes < 1 or count < 1 or batch_size < 1:
            raise ValueError(
                f"Invalid loader sizes modes={modes} count={count} batch_size={batch_size}"
            )
        self.modes = modes
        self.count = count
        self.batch_size = batch_size
        self.seed_sequence = seed_sequence

    def __len__(self) -> int:
        return -(-self.count // self.batch_size)

    def __iter__(self) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self.seed_sequence)
        remaining = self.count
        while remaining > 0:
            size = min(self.batch_size, remaining)
            yield np.stack([haar_unitary(self.modes, rng) for _ in range(size)])
            remaining -= size


def build_dataloader(cfg: dict[str, Any], split: str) -> HaarUnitaryLoader:
    """Build the Haar-unitary loader for one experiment case.

    Parameters
    ----------
    cfg : dict
        Needs ``seed``, ``modes``, ``monte_carlo.unitaries`` and
        ``monte_carlo.batch_size``; ``case_index`` (default 0) separates the
        streams of different cases in one run.
    split : str
        Stream label, e.g. ``"monte_carlo"`` or ``"train"``. Different labels
        give independent streams.

    Returns
    -------
    HaarUnitaryLoader
        Yields arrays of shape ``(batch, M, M)``.

    Raises
    ------
    KeyError
        If a required key is missing.
    """
    monte_carlo = cfg["monte_carlo"]
    seed_sequence = np.random.SeedSequence(
        [int(cfg["seed"]), int(cfg.get("case_index", 0)), zlib.crc32(split.encode())]
    )
    return HaarUnitaryLoader(
        modes=int(cfg["modes"]),
        count=int(monte_carlo["unitaries"]),
        batch_size=int(monte_carlo["batch_size"]),
        seed_sequence=seed_sequence,
    )
