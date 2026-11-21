from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
PAPERS_DIR = PROJECT_DIR.parent
# REPO_ROOT provides runtime_lib; PAPERS_DIR makes ``photonic_trainability`` an
# importable package regardless of test order.
for _path in (REPO_ROOT, PAPERS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from runtime_lib.cli import build_cli_parser  # noqa: E402
from runtime_lib.config import load_config  # noqa: E402
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
