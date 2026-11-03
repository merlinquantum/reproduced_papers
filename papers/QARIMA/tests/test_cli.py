from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_defaults_have_required_keys():
    cfg = load_runtime_ready_config()
    for key in ("dataset", "refiners", "candidates", "loss", "vqc", "order_sweep"):
        assert key in cfg, f"missing config key {key!r}"
    assert cfg["dataset"]["name"] in {
        "sunspots",
        "co2",
        "ausbeer",
        "woolyarn",
        "sydney",
    }
