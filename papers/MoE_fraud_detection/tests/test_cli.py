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
    assert cfg["cv"]["n_repeats"] >= 1
    assert cfg["model"]["n_qubits"] == 6
    assert cfg["model"]["vqc"]["n_layers"] == 6
    assert cfg["model"]["autoencoder"]["hidden_dims"] == [256, 128, 64]
    assert cfg["training"]["epochs"] >= 1
    assert isinstance(cfg["evaluation"]["router_thresholds"], list)
    assert len(cfg["evaluation"]["router_thresholds"]) >= 1
