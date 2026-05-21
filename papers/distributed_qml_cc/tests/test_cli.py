from __future__ import annotations

from pathlib import Path

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config

from runtime_lib.cli import apply_cli_overrides


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_overrides_apply(tmp_path):
    parser, arg_defs = build_project_cli_parser()
    ns = parser.parse_args(
        ["--scheme", "cc", "--n-layers", "3", "--iterations", "10"]
    )
    cfg = {"model": {"params": {}}, "training": {}}
    cfg = apply_cli_overrides(
        cfg, ns, arg_defs, PROJECT_DIR, Path(tmp_path)
    )
    assert cfg["model"]["params"]["scheme"] == "cc"
    assert cfg["model"]["params"]["n_layers"] == 3
    assert cfg["training"]["n_iterations"] == 10


def test_defaults_have_required_keys():
    cfg = load_runtime_ready_config()
    assert cfg["pipeline"] == "quantum"
    assert cfg["model"]["params"]["scheme"] in {"non", "nc", "cc", "qc"}
    assert "n_layers" in cfg["model"]["params"]
