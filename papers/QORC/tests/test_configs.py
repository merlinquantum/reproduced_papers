from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_lib.config import load_config  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"


def test_defaults_has_description_and_required_keys() -> None:
    cfg = load_config(CONFIG_DIR / "defaults.json")
    assert cfg["description"], "defaults.json must document the experiment"
    for key in ("xp_type", "outdir", "n_photons", "n_modes"):
        assert key in cfg, f"Missing '{key}' in defaults.json"


def test_cli_schema_matches_defaults_path() -> None:
    defaults_path = PROJECT_DIR / "configs" / "defaults.json"
    cli_schema_path = PROJECT_DIR / "cli.json"
    if not cli_schema_path.exists():
        cli_schema_path = PROJECT_DIR / "configs" / "cli.json"

    assert defaults_path.exists(), "defaults.json missing"
    assert cli_schema_path.exists(), "cli.json missing"

    runner_module = importlib.import_module("lib.runner")
    assert hasattr(runner_module, "train_and_evaluate"), (
        "Runner must expose train_and_evaluate()"
    )
