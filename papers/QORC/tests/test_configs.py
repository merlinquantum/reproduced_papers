from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_lib.config import load_config

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"


def test_defaults_has_description_and_required_keys() -> None:
    cfg = load_config(CONFIG_DIR / "defaults.json")
    assert cfg["description"], "defaults.json must document the experiment"
    for key in (
        "xp_type",
        "outdir",
        "n_photons",
        "n_modes",
        "n_epochs",
        "dataset_name",
        "noise_enabled",
        "use_qpu",
    ):
        assert key in cfg, f"Missing '{key}' in defaults.json"
    assert cfg["n_photons"] == 3
    assert cfg["n_modes"] == 20
    assert cfg["n_epochs"] == 200
    assert cfg["dataset_name"] == "mnist"
    assert cfg["noise_enabled"] is False
    assert cfg["use_qpu"] is False


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


def test_qorc_lsvc_comparison_config_has_requested_settings() -> None:
    comparison = load_config(CONFIG_DIR / "comparison_QORC_LSVC_mnist.json")

    assert comparison["xp_type"] == "comparison_qorc_lsvc"
    assert comparison["dataset_name"] == "mnist"
    assert comparison["n_photons"] == 3
    assert comparison["n_modes"] == 20
    assert comparison["n_epochs"] == 200
    assert comparison["noise_enabled"] is False
    assert comparison["use_qpu"] is False
