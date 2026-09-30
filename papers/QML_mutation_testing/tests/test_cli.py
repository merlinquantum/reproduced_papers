from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_defaults_declare_photonic_model():
    cfg = load_runtime_ready_config()
    params = cfg["model"]["params"]
    assert params["n_photons"] >= 2, "a photonic reproduction needs at least 2 photons"
    assert params["n_modes"] >= params["n_classes"]
    assert cfg["dataset"]["n_suite_samples"] == 20, "the paper uses 20 test samples"
    assert len(cfg["experiment"]["seeds"]) >= 3, "report mean +/- std over >= 3 seeds"
