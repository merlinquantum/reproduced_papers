from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
try:
    from runtime_lib.cli import build_cli_parser
    from runtime_lib.config import load_config
    from runtime_lib.dtypes import resolve_config_dtypes
except ModuleNotFoundError:
    for path in (REPO_ROOT, PROJECT_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from runtime_lib.cli import build_cli_parser
    from runtime_lib.config import load_config
    from runtime_lib.dtypes import resolve_config_dtypes

_CLI_SCHEMA_PATH = PROJECT_DIR / "cli.json"
_DEFAULTS_PATH = PROJECT_DIR / "configs" / "defaults.json"
_GLOBAL_CLI_SCHEMA_PATH = REPO_ROOT / "runtime_lib" / "global_cli.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def build_project_cli_parser():
    schema = _load_json(_CLI_SCHEMA_PATH)
    schema.setdefault("arguments", [])
    if _GLOBAL_CLI_SCHEMA_PATH.exists():
        global_schema = _load_json(_GLOBAL_CLI_SCHEMA_PATH)
        schema["arguments"].extend(global_schema.get("arguments", []))
    return build_cli_parser(schema)


def load_project_defaults() -> dict:
    return load_config(_DEFAULTS_PATH)


def load_runtime_ready_config() -> dict:
    cfg = deepcopy(load_project_defaults())
    return resolve_config_dtypes(cfg)


__all__ = [
    "PROJECT_DIR",
    "REPO_ROOT",
    "build_project_cli_parser",
    "load_project_defaults",
    "load_runtime_ready_config",
]
