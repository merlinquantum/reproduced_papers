from __future__ import annotations

import json

import pytest
from common import PROJECT_DIR, build_project_cli_parser


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_every_config_is_valid_json_with_required_sections():
    for path in (PROJECT_DIR / "configs").glob("*.json"):
        assert isinstance(json.loads(path.read_text()), dict), path.name
