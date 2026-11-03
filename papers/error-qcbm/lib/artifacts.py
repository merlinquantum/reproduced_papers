"""Atomic run artifacts, JSON serialization and objective-shot accounting."""

import json
import logging
from collections import Counter
from pathlib import Path
from time import sleep


def atomic_replace(source: Path, destination: Path) -> None:
    """Replace an artifact, tolerating short Windows sharing/access conflicts.

    Parameters
    ----------
    source : pathlib.Path
        Fully written and closed temporary file in the destination directory.
    destination : pathlib.Path
        Artifact to replace. Only one writer may own this destination.

    Raises
    ------
    OSError
        If replacement fails. Only Windows errors 5, 32 and 33 are retried,
        with at most eight attempts and 2.25 seconds of total delay. Other
        errors propagate immediately. The temporary file is preserved on failure.

    Notes
    -----
    Retries repeat the same file-system operation, never sampling, estimation,
    serialization or optimizer updates. Source and destination stay on one
    file system to retain atomic replacement semantics.
    """
    for attempt in range(8):
        try:
            source.replace(destination)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 7:
                raise
            delay = min(0.05 * 2**attempt, 0.5)
            logging.getLogger(__name__).warning(
                "Artifact replacement blocked (Windows %s); retry %s/7 in %.2fs: %s",
                error.winerror,
                attempt + 1,
                delay,
                destination.name,
            )
            sleep(delay)


def planned_objective_calls(steps: int, record_every: int) -> dict:
    """Count calls for initial reporting, calibration, updates and evaluation.

    Parameters
    ----------
    steps : int
        Positive number of SPSA updates.
    record_every : int
        Positive reporting interval. The final update is always evaluated.

    Returns
    -------
    dict
        Call counts before any interrupted work or separate retry branches.
    """
    if steps <= 0 or record_every <= 0:
        raise ValueError("Steps and recording interval must be positive")
    parts = {
        "initial_evaluation": 1,
        "gain_calibration": 2,
        "spsa_perturbations": 2 * steps,
        "recorded_evaluations": (steps + record_every - 1) // record_every,
    }
    return {**parts, "total": sum(parts.values())}


def summarize_sampling_ledger(path: Path) -> dict:
    """Account for an append-only ledger without treating replays as free.

    Parameters
    ----------
    path : pathlib.Path
        One attempt directory's ``sampling_ledger.jsonl``. Parent retry ledgers
        must be summarized separately; inherited checkpoints cost no new shots.

    Returns
    -------
    dict
        Attempt counts, known fully sampled shots and an upper bound for calls
        whose sampling completion is unknown. A terminal row without an explicit
        completion marker in a legacy failed call remains uncertain. Replayed
        completed calls are identified by call index and sampling seed.

    Raises
    ------
    ValueError
        If records are inconsistent. A final unterminated JSON line is retained
        as an incomplete record; malformed complete lines are rejected.
    """
    data = path.read_bytes() if path.exists() else b""
    attempts = []
    pending = None
    incomplete_line = False
    for index, raw in enumerate(data.splitlines(keepends=True)):
        if not raw.endswith(b"\n"):
            incomplete_line = True
            break
        try:
            row = json.loads(raw)
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"Malformed sampling ledger line {index + 1}") from error
        key = (row["call_index"], row["sampling_seed"])
        if row["status"] == "started":
            pending = {"key": key, "shots": row["initial_shots"], "terminal": None}
            attempts.append(pending)
        elif row["status"] in ("complete", "failed"):
            if (
                pending is None
                or pending["key"] != key
                or pending["terminal"] is not None
            ):
                raise ValueError("Sampling terminal row has no matching started call")
            if pending["shots"] != row["initial_shots"]:
                raise ValueError("Sampling attempt changed its shot budget")
            pending["terminal"] = row
        else:
            raise ValueError("Unknown sampling ledger status")
    complete = Counter()
    known = unknown = completed = failed = unfinished = 0
    for attempt in attempts:
        row = attempt["terminal"]
        if row is None:
            unfinished += 1
        elif row["status"] == "complete":
            completed += 1
            complete[(attempt["key"], attempt["shots"])] += 1
        else:
            failed += 1
        if row is not None and (
            row["status"] == "complete" or row.get("sampling_completed") is True
        ):
            known += attempt["shots"]
        else:
            unknown += attempt["shots"]
    return {
        "started_attempts": len(attempts),
        "completed_attempts": completed,
        "failed_attempts": failed,
        "unfinished_attempts": unfinished,
        "known_fully_sampled_initial_shots": known,
        "unknown_completion_initial_shots_upper_bound": None
        if incomplete_line
        else unknown,
        "replayed_completed_calls": sum(count - 1 for count in complete.values()),
        "replayed_completed_initial_shots": sum(
            (count - 1) * key[1] for key, count in complete.items()
        ),
        "incomplete_trailing_line": incomplete_line,
        "scope": "This directory only; parent attempt ledgers are not included",
    }


def write_json(path: Path, value: dict) -> None:
    """Atomically replace a UTF-8 JSON progress artifact.

    Parameters
    ----------
    path : pathlib.Path
        Destination in the run directory.
    value : dict
        JSON-compatible progress or result object.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    atomic_replace(temporary, path)
