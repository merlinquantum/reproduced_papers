"""Small helpers for human-readable experiment logging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


def configure_logging(
    log_file: str | Path,
    *,
    level: int = logging.INFO,
    console: bool = True,
) -> None:
    """Configure UTF-8 file logging and optional console logging in UTC.

    Args:
        log_file: File that receives the experiment log.
        level: Python logging level.
        console: Whether to mirror records to stderr.
    """

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(path, encoding="utf-8")
    ]
    if console:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=level,
        format="%(asctime)sZ [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def utc_now() -> str:
    """Return the current UTC time in a stable format."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically write a JSON object.

    Args:
        path: Destination file.
        value: JSON-serializable mapping to write.
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
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: str | Path) -> dict[str, Any]:
    """Return the current Git commit and dirty-worktree state.

    Args:
        root: Repository containing the experiment code.

    Returns:
        Mapping with ``commit`` and ``dirty`` fields.
    """

    command = ["git", "-C", str(Path(root).resolve())]
    commit = subprocess.check_output(
        [*command, "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            [*command, "status", "--porcelain"],
            text=True,
        ).strip()
    )
    return {"commit": commit, "dirty": dirty}
