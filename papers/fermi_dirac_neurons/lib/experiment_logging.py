"""Small helpers for human-readable experiment logging."""

from __future__ import annotations

import calendar
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


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
    handlers: list[logging.Handler] = [logging.FileHandler(path, encoding="utf-8")]
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

    On a remote GPU machine the code arrives without ``.git``. The
    ``remote-gpu`` runner then sets ``ALPS_CODE_COMMIT`` and
    ``ALPS_CODE_DIRTY`` to the laptop's state, and those are returned.

    Args:
        root: Repository containing the experiment code.

    Returns:
        Mapping with ``commit`` and ``dirty`` fields.
    """

    commit = os.environ.get("ALPS_CODE_COMMIT")
    if commit:
        return {"commit": commit, "dirty": os.environ.get("ALPS_CODE_DIRTY") == "1"}
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


def sweep_progress(status: Mapping[str, Any], *, now: float | None = None) -> str:
    """Summarize how far a sweep has got, for a log line or the progress viewer.

    Each completed run counts as one. A running run adds the finished fraction
    of its ``progress`` object. Planned, failed, invalid and superseded runs
    add nothing, so a retried attempt is never counted twice.

    Args:
        status: Parsed ``sweep_status.json``.
        now: Current Unix time, for tests. Defaults to ``time.time()``.

    Returns:
        Text such as ``7/12 runs completed · 58% done · about 1h 5m left``.
    """

    expected = status.get("expected_runs")
    runs = status.get("runs")
    if not isinstance(expected, int) or expected <= 0 or not isinstance(runs, list):
        return "progress unavailable"

    completed = sum(
        1 for run in runs if isinstance(run, dict) and run.get("status") == "COMPLETED"
    )
    done = float(completed)
    for run in runs:
        if not isinstance(run, dict) or run.get("status") != "RUNNING":
            continue
        progress = run.get("progress")
        if not isinstance(progress, dict):
            continue
        current, total = progress.get("current"), progress.get("total")
        if isinstance(current, (int, float)) and isinstance(total, (int, float)):
            if total > 0 and current >= 0:
                done += min(1.0, current / total)
    done = min(float(expected), done)
    text = (
        f"{min(completed, expected)}/{expected} runs completed · "
        f"{round(100 * done / expected)}% done"
    )

    if status.get("status") != "RUNNING":
        return f"{text} · sweep {str(status.get('status', 'unknown')).lower()}"
    try:
        started = calendar.timegm(
            time.strptime(status["started_at"], "%Y-%m-%dT%H:%M:%SZ")
        )
    except (KeyError, TypeError, ValueError):
        return f"{text} · time left unknown"
    elapsed = (time.time() if now is None else now) - started
    if done <= 0 or elapsed <= 0:
        return f"{text} · time left unknown"
    return f"{text} · about {_duration(elapsed / done * (expected - done))} left"


def _duration(seconds: float) -> str:
    """Format a duration as ``<1m``, ``25m``, ``1h 5m`` or ``2d 3h``."""

    if seconds < 60:
        return "<1m"
    minutes = math.ceil(seconds / 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h" if hours else f"{days}d"
