"""Data loading for the QRC level-generation reproduction.

The Mario level 1-2 sequence and the published QRC-generated sequences come from
Moth's Open Data repository (Level_Generation_with_Quantum_Reservoir_Computing
folder). The original-level sequence is committed as JSON; the reference
generated sequences live alongside as pickled lists keyed by temperature/qubit
count/backend.
"""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Iterable
from pathlib import Path

_PAPER_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PAPER_DIR.parents[1]


def _resolve(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return (_PAPER_DIR / candidate).resolve()


def load_original_level(level_file: str) -> tuple[list[int], int]:
    """Load the original Mario level 1-2 sequence and vocab size."""
    path = _resolve(level_file)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    sequence = list(payload["sequence"])
    return sequence, int(payload["num_features"])


def load_reference_sequences(
    reference_root: str,
    level: str = "1-2",
    n_qubits: int = 6,
    backend: str = "Aer",
    betas: Iterable[float] | None = None,
) -> dict[float, list[list[int]]]:
    """Load published QRC-generated sequences from the Moth open-data dump.

    Parameters
    ----------
    reference_root : str
        Path containing the ``SMB`` and ``Roblox`` subdirectories.
    level : str
        Level identifier (default ``"1-2"`` for Super Mario Bros).
    n_qubits : int
        Qubit count subfolder.
    backend : str
        One of ``"Aer"``, ``"Aer_matrixnoise"``, ``"FakeGarnet"``, ``"FakeJames"``.
    betas : iterable of floats, optional
        Temperatures (paper notation ``T = beta``) to load. If ``None``, every
        pickle matching the pattern is loaded.
    """
    root = _resolve(reference_root)
    folder = root / "SMB" / f"{n_qubits}_qubits" / backend
    if not folder.is_dir():
        raise FileNotFoundError(f"Reference folder missing: {folder}")

    if betas is None:
        wanted = None
    else:
        wanted = {_format_beta(b) for b in betas}

    sequences: dict[float, list[list[int]]] = {}
    for entry in os.listdir(folder):
        if not entry.startswith(f"Sequences_level_{level}_beta_"):
            continue
        if not entry.endswith(f"_{backend}.p"):
            continue
        # Filename like Sequences_level_1-2_beta_1_Aer.p
        beta_str = entry.removeprefix(f"Sequences_level_{level}_beta_")
        beta_str = beta_str.removesuffix(f"_{backend}.p")
        if wanted is not None and beta_str not in wanted:
            continue
        with (folder / entry).open("rb") as handle:
            sequences[float(beta_str)] = pickle.load(handle)
    return sequences


def _format_beta(value: float | str | int) -> str:
    if isinstance(value, str):
        return value
    if float(value).is_integer():
        return str(int(value))
    return str(value)
