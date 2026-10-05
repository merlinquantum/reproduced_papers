from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
for import_root in (REPO_ROOT, PROJECT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from runtime_lib.cli import build_cli_parser  # noqa: E402
from runtime_lib.config import deep_update, load_config  # noqa: E402
from runtime_lib.dtypes import resolve_config_dtypes  # noqa: E402

_CLI_SCHEMA_PATH = PROJECT_DIR / "cli.json"
_DEFAULTS_PATH = PROJECT_DIR / "configs" / "defaults.json"

BINDING_POSITIONS = [2, 4, 5, 9]
IMMUNOGENIC_POSITIONS = [3, 5, 6, 7]


def build_project_cli_parser():
    schema = json.loads(_CLI_SCHEMA_PATH.read_text())
    schema.setdefault("arguments", [])
    return build_cli_parser(schema)


def load_project_defaults() -> dict:
    return load_config(_DEFAULTS_PATH)


def load_runtime_ready_config() -> dict:
    cfg = deepcopy(load_project_defaults())
    return resolve_config_dtypes(cfg)


def make_config(
    task: str = "binding", epochs: int = 1, resolve_dtypes: bool = True, **model
) -> dict:
    """Return defaults overlaid with a task and model, as the runtime builds it.

    Parameters
    ----------
    task : str
        ``binding`` or ``immunogenic``; selects the paper's residue positions.
        Default value is ``"binding"``.
    epochs : int
        Training budget. Default value is 1.
    resolve_dtypes : bool
        Normalise ``dtype`` as the runtime does before calling the runner.
        Set False for the JSON form the runtime writes to
        ``config_snapshot.json``. Default value is True.
    **model
        Entries of the ``model`` section, for example ``name="merlin"``.
    """
    positions = BINDING_POSITIONS if task == "binding" else IMMUNOGENIC_POSITIONS
    overlay = {
        "dataset": {"task": task, "positions": positions},
        "model": model,
        "training": {"epochs": epochs, "spsa_a": 0.06, "spsa_c": 0.2},
    }
    cfg = deep_update(deepcopy(load_project_defaults()), overlay)
    return resolve_config_dtypes(cfg) if resolve_dtypes else cfg


MODEL_SETTINGS = {
    "merlin": {"name": "merlin"},
    "merlin_reference": {"name": "merlin_reference"},
    "classical": {"name": "classical", "hidden": 12},
    "classical_reference": {"name": "classical_reference", "hidden": 17},
    "gate_source": {"name": "gate_source", "source_labels": "signed"},
    "gate_corrected": {"name": "gate_corrected"},
    "logistic": {"name": "logistic"},
    "rf": {"name": "rf"},
    "logistic_onehot": {"name": "logistic_onehot"},
    "rf_onehot": {"name": "rf_onehot"},
}


__all__ = [
    "BINDING_POSITIONS",
    "IMMUNOGENIC_POSITIONS",
    "MODEL_SETTINGS",
    "PROJECT_DIR",
    "REPO_ROOT",
    "build_project_cli_parser",
    "load_project_defaults",
    "load_runtime_ready_config",
    "make_config",
]
