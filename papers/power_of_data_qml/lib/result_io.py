"""Read and write the curated JSON results in ``results/``.

The aggregates repeat the same keys thousands of times, so they are stored as
compact JSON and gzip-compressed (``.json.gz``), which makes them about ten
times smaller. Both functions also accept a plain ``.json`` path.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    """Load a JSON file, decompressing it first when the name ends in ``.gz``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.json`` or ``.json.gz`` file.

    Returns
    -------
    Any
        The decoded JSON value.
    """
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(value: Any, path: str | Path, **kwargs: Any) -> None:
    """Write ``value`` as compact JSON, gzip-compressed when the name ends in ``.gz``.

    The gzip header carries no timestamp, so identical content gives identical
    bytes and an unchanged result does not show up as a git change.

    Parameters
    ----------
    value : Any
        JSON-serialisable value.
    path : str or pathlib.Path
        Destination ``.json`` or ``.json.gz`` file.
    **kwargs : Any
        Extra keyword arguments for :func:`json.dumps`, e.g. ``sort_keys`` or
        ``default``.
    """
    path = Path(path)
    text = json.dumps(value, separators=(",", ":"), **kwargs) + "\n"
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
                handle.write(text.encode("utf-8"))
    else:
        path.write_text(text, encoding="utf-8")
