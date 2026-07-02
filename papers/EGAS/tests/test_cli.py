from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_defaults_config_has_required_keys():
    cfg = load_runtime_ready_config()
    assert cfg["task"] in {"egas_eval", "wasserstein", "fig1"}
    assert cfg["dataset"]["n_qubits"] == 8
    assert cfg["egas"]["seq_len"] == 28
