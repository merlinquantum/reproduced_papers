from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_project_defaults


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_flags_parse():
    parser, _ = build_project_cli_parser()
    args = parser.parse_args(
        ["--seeds", "0,1,2", "--max-episodes", "7", "--encoding-activation", "tanh"]
    )
    assert args.seeds == [0, 1, 2]
    assert args.max_episodes == 7
    assert args.encoding_activation == "tanh"


def test_defaults_are_a_lightweight_smoke_config():
    cfg = load_project_defaults()
    assert cfg["training"]["max_episodes"] <= 5
    assert len(cfg["agents"]["seeds"]) <= 2
