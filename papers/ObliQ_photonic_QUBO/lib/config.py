"""Run configs: loading, CLI overrides, and content-addressed storage.

A run is described by a JSON config in ``configs/``. This module owns its whole
lifecycle:

1. **Load** -- :func:`load_config`, by path or bare filename.
2. **Override** -- ``cli.json`` declares the flags, so adding one needs no Python
   change here. Two kinds: ``config_path`` flags overwrite a dotted key inside the
   config (``sweep.size_range``); ``kwarg`` flags are passed straight to
   :func:`solver.run_instance`. A flag the user did not pass stays ``None``
   and is never applied -- which is what keeps an unused override out of the
   config, and therefore out of its hash.
3. **Address** -- a run is identified by a short hash of its config's *experiment
   identity*: the fields that actually determine the results. Execution-only knobs
   (``output``, ``sweep.parallel_workers``), the display-only ``name``, and the
   ``description`` required by the repository's shared runner (see
   ``runtime_lib/config.py``) are excluded, so re-running with more workers, a
   different plot label, or a config edited only to add ``description`` reuses
   the same folder, while changing the seed/solver/options produces a new one.

Layout written by :mod:`lib.benchmark` and read by :mod:`utils.plotter`::

    <output.dir>/<hash>/results.json   # the sweep output
    <output.dir>/<hash>/config.json    # a copy of the resolved config

Because the hash covers config *content*, keeping the shipped ``configs/*.json``
byte-stable outside the ignored fields above is what keeps the shipped
``results/<hash>/`` directories findable. Derived seeds are computed in
:mod:`lib.seeding` rather than stored in a config for exactly that reason.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

RESULTS_FILE = "results.json"
CONFIG_FILE = "config.json"
DEFAULT_OUTPUT_DIR = "results"

#: Type names usable in ``cli.json``.
TYPES: dict[str, Any] = {
    "int": int,
    "float": float,
    "str": str,
    "json": json.loads,
    "csv_int_list": lambda s: [int(x) for x in s.split(",") if x.strip()],
}

# Fields that do not affect the computed results and are stripped before hashing.
# "description" is required by the shared runner's own config loader
# (runtime_lib/config.py) for every file reachable through --config, but it is
# purely documentation -- ignored here for the same reason as "name".
_IGNORED_TOP_LEVEL = ("output", "name", "description")
_IGNORED_SWEEP = ("parallel_workers",)


# ---------------------------------------------------------------------------
# Loading and CLI overrides
# ---------------------------------------------------------------------------
def load_cli_spec(path) -> dict:
    """Load the ``cli.json`` command declaration."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def add_arguments(parser: argparse.ArgumentParser, arguments: list) -> None:
    """Add one command group's declared flags to ``parser``."""
    for arg in arguments:
        apply = arg.get("apply", {})
        if "path" not in apply:
            continue
        kwargs: dict[str, Any] = {
            "help": arg.get("help"),
            # dotted config paths become argparse dests: sweep.seed -> sweep__seed
            "dest": apply["path"].replace(".", "__"),
            "default": None,
        }
        if arg.get("action") == "store_true":
            # default stays None so "not passed" is distinguishable from False.
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = TYPES[arg.get("type", "str")]
        if arg.get("choices"):
            kwargs["choices"] = arg["choices"]
        if arg.get("required"):
            kwargs["required"] = True
        parser.add_argument(*arg["flags"], **kwargs)


def set_config_path(config: dict, dotted: str, value) -> None:
    """Assign ``value`` into ``config`` at a dotted key path, creating dicts."""
    keys = dotted.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply_overrides(config: dict, arguments: list, args: argparse.Namespace) -> None:
    """Overlay the CLI values that were actually provided onto ``config``."""
    for arg in arguments:
        apply = arg.get("apply", {})
        if apply.get("kind") != "config_path" or "path" not in apply:
            continue
        value = getattr(args, apply["path"].replace(".", "__"), None)
        if value is not None:
            set_config_path(config, apply["path"], value)


def collect_kwargs(arguments: list, args: argparse.Namespace) -> dict:
    """Collect the ``kwarg``-kind flags that were provided, as a kwargs dict."""
    kwargs = {}
    for arg in arguments:
        apply = arg.get("apply", {})
        if apply.get("kind") != "kwarg" or "path" not in apply:
            continue
        value = getattr(args, apply["path"].replace(".", "__"), None)
        if value is not None:
            kwargs[apply["path"]] = value
    return kwargs


def load_config(path: str, config_dir: str = "configs") -> dict:
    """Load a run config by path or by bare filename inside ``config_dir``.

    Accepts ``configs/obliq_maxcut.json`` or just ``obliq_maxcut.json``, so both
    forms work from the project directory.

    Raises:
        FileNotFoundError: if neither form exists.
    """
    for candidate in (Path(path), Path(config_dir) / path):
        if candidate.exists():
            with open(candidate, encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError(
        f"Unable to locate config file: {path} (also tried {Path(config_dir) / path})"
    )


# ---------------------------------------------------------------------------
# Content-addressed storage
# ---------------------------------------------------------------------------
def canonical_config(config: dict) -> dict:
    """Return a copy of ``config`` with execution-only / label fields removed."""
    canonical = copy.deepcopy(config)
    for key in _IGNORED_TOP_LEVEL:
        canonical.pop(key, None)
    sweep = canonical.get("sweep")
    if isinstance(sweep, dict):
        for key in _IGNORED_SWEEP:
            sweep.pop(key, None)
    return canonical


def config_hash(config: dict, length: int = 12) -> str:
    """Stable short hash of a config's experiment identity."""
    payload = json.dumps(
        canonical_config(config), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def run_dir(config: dict, output_dir: str | None = None) -> str:
    """Directory holding this config's results and config copy (not created).

    ``output_dir`` defaults to the config's own ``output.dir``, or ``results``.
    """
    if output_dir is None:
        output_dir = config.get("output", {}).get("dir", DEFAULT_OUTPUT_DIR)
    return os.path.join(output_dir, config_hash(config))
