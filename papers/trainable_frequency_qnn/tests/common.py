from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
for import_root in (REPO_ROOT, PROJECT_DIR.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from runtime_lib.cli import build_cli_parser  # noqa: E402
from runtime_lib.config import deep_update, load_config  # noqa: E402
from runtime_lib.dtypes import resolve_config_dtypes  # noqa: E402

_CLI_SCHEMA_PATH = PROJECT_DIR / "cli.json"
_DEFAULTS_PATH = PROJECT_DIR / "configs" / "defaults.json"


def build_project_cli_parser():
    schema_path = _CLI_SCHEMA_PATH
    if not schema_path.exists():
        schema_path = PROJECT_DIR / "configs" / "cli.json"
    schema = json.loads(schema_path.read_text())
    schema.setdefault("arguments", [])
    return build_cli_parser(schema)


def load_project_defaults() -> dict:
    return load_config(_DEFAULTS_PATH)


def load_runtime_ready_config(overlay: str | None = None) -> dict:
    """Return defaults merged with ``configs/<overlay>``, dtypes resolved."""
    cfg = deepcopy(load_project_defaults())
    if overlay is not None:
        cfg = deep_update(cfg, load_config(PROJECT_DIR / "configs" / overlay))
    cfg["data_root"] = None
    return resolve_config_dtypes(cfg)


__all__ = [
    "PROJECT_DIR",
    "REPO_ROOT",
    "build_project_cli_parser",
    "load_project_defaults",
    "load_runtime_ready_config",
]
