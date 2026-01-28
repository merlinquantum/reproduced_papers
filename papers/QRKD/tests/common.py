from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_lib.cli import build_cli_parser  # noqa: E402
from runtime_lib.config import load_config  # noqa: E402
from runtime_lib.dtypes import resolve_config_dtypes  # noqa: E402


_CLI_SCHEMA_PATH = PROJECT_DIR / "cli.json"
_DEFAULTS_PATH = PROJECT_DIR / "configs" / "defaults.json"


def build_project_cli_parser():
    """Return the ArgumentParser (and CLI defs) for the QRKD project."""

    schema_path = _CLI_SCHEMA_PATH
    if not schema_path.exists():
        schema_path = PROJECT_DIR / "configs" / "cli.json"
    schema = json.loads(schema_path.read_text())
    schema.setdefault("arguments", [])
    return build_cli_parser(schema)


def load_project_defaults() -> dict:
    """Load QRKD's defaults.json into a Python dict."""

    return load_config(_DEFAULTS_PATH)


def load_runtime_ready_config() -> dict:
    """Return a deepcopy of defaults.json with dtype specs resolved."""

    cfg = deepcopy(load_project_defaults())
    return resolve_config_dtypes(cfg)


__all__ = [
    "PROJECT_DIR",
    "REPO_ROOT",
    "build_project_cli_parser",
    "load_project_defaults",
    "load_runtime_ready_config",
]
