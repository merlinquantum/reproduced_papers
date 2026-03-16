from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_project_defaults


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_defaults_have_model_name():
    cfg = load_project_defaults()
    assert "model" in cfg
    assert cfg["model"].get("name")
