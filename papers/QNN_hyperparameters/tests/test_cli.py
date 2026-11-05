from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_defaults_match_author_pipeline():
    cfg = load_runtime_ready_config()
    assert cfg["mode"] == "qnn"
    assert cfg["dataset"]["name"] in {"kdd_cup", "cover_type", "glass", "rice"}
    assert cfg["dataset"]["n_components"] == 7
    assert cfg["dataset"]["n_train"] == 400
    assert cfg["dataset"]["n_eval"] == 250
    assert cfg["dataset"]["subsample_seed"] == 12
    assert cfg["training"]["shots"] == 1024
    assert cfg["training"]["optimizer"] in {"COBYLA", "SPSA", "NelderMead"}
    assert cfg["model"]["feature_map_reps"] == 2 and cfg["model"]["ansatz_reps"] == 3
