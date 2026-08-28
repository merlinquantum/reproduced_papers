from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_defaults_config_is_well_formed():
    cfg = load_runtime_ready_config()
    assert cfg["model"]["params"]["n_qubits"] == 6
    assert cfg["model"]["params"]["depth"] == 32
    assert cfg["training"]["total_steps"] > 0
