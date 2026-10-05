"""Run artifacts: JSON/CSV writers, file hashes and code provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def utc_now() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically write a JSON object with sorted keys.

    Parameters
    ----------
    path : str | Path
        Destination file.
    value : Mapping
        JSON-serialisable mapping.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: str | Path) -> dict[str, Any]:
    """Return the Git commit and dirty state of ``root``.

    Provenance is metadata, not an input: outside a Git checkout (for example
    an exported source tree) the commit is recorded as ``"unknown"``.

    Parameters
    ----------
    root : str | Path
        Directory inside the repository.

    Returns
    -------
    dict
        ``commit`` (str) and ``dirty`` (bool, or None when unknown).
    """
    command = ["git", "-C", str(Path(root).resolve())]
    try:
        commit = subprocess.check_output(
            [*command, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            [*command, "status", "--porcelain", "--", "."],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        LOGGER.warning("Git provenance unavailable for %s", root)
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def write_predictions(
    path: Path, predictions: Mapping[str, Sequence[float]], data: Mapping[str, Any]
) -> None:
    """Write per-peptide probabilities and fixed-threshold decisions.

    Parameters
    ----------
    path : Path
        Destination ``predictions.csv``.
    predictions : Mapping
        Split name to positive-class probabilities.
    data : Mapping
        Task data from :func:`lib.data.load_task` providing peptides and targets.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "peptide", "truth", "probability", "prediction"])
        for split, probabilities in predictions.items():
            for peptide, truth, probability in zip(
                data[split]["peptides"],
                data[split]["targets"].int().tolist(),
                probabilities,
            ):
                writer.writerow(
                    [split, peptide, truth, probability, int(probability >= 0.5)]
                )


def write_losses(path: Path, losses: Sequence[float]) -> None:
    """Write the per-update training objective to ``losses.csv``."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "training_objective"])
        writer.writerows(enumerate(losses, start=1))
