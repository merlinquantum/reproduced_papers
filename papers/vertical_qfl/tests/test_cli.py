from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_original_configs_carry_paper_hyperparameters():
    import json

    from common import PROJECT_DIR

    cfg = json.loads((PROJECT_DIR / "configs/e1_8clients_original.json").read_text())
    assert cfg["training"]["epochs"] == 30
    assert cfg["training"]["lr"] == 0.002
    assert cfg["dataset"]["batch_size"] == 32
    assert cfg["model"]["n_blocks"] == 5 and cfg["model"]["depth_per_block"] == 100
    assert len(cfg["model"]["client_slices"]) == 8
    assert cfg["dataset"]["max_train_samples"] is None


def test_defaults_are_lightweight():
    cfg = load_runtime_ready_config()
    assert cfg["training"]["epochs"] == 2 and cfg["model"]["depth_per_block"] == 2
