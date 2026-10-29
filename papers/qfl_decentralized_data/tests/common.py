from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
DATA_ROOT = REPO_ROOT / "data"
MNIST_DIR = DATA_ROOT / "qfl_decentralized_data" / "mnist" / "raw"

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


def load_runtime_ready_config() -> dict:
    cfg = deepcopy(load_project_defaults())
    return resolve_config_dtypes(cfg)


__all__ = [
    "DATA_ROOT",
    "MNIST_DIR",
    "load_named_config",
    "PROJECT_DIR",
    "REPO_ROOT",
    "build_project_cli_parser",
    "load_project_defaults",
    "load_runtime_ready_config",
]


def load_named_config(name: str, **overrides) -> dict:
    """Defaults overlaid with ``configs/<name>.json`` and top-level section overrides."""
    cfg = deepcopy(load_project_defaults())
    overlay = json.loads((PROJECT_DIR / "configs" / f"{name}.json").read_text())
    cfg = deep_update(cfg, overlay)
    cfg = deep_update(cfg, overrides)
    cfg["data_root"] = str(DATA_ROOT)
    return resolve_config_dtypes(cfg)
