"""Portable artifact writing with bounded retries for Windows file sharing."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def replace_file(source: Path, destination: Path) -> None:
    """Replace an artifact, retrying only transient permission errors.

    Parameters
    ----------
    source, destination : Path
        Temporary and final paths on the same filesystem.
    """
    for attempt in range(30):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(min(0.2 * (attempt + 1), 1.0))


def write_json(path: Path, payload: object) -> None:
    """Atomically write JSON without allowing nonfinite numbers.

    Parameters
    ----------
    path : Path
        Destination file.
    payload : object
        JSON-serializable data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    for attempt in range(30):
        try:
            temporary.write_text(text, encoding="utf-8")
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(min(0.2 * (attempt + 1), 1.0))
    replace_file(temporary, path)


def sha256(path: Path) -> str:
    """Return a file's streaming SHA-256 digest.

    Parameters
    ----------
    path : Path
        Input file.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
