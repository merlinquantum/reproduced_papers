from __future__ import annotations

import json

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_project_defaults

from runtime_lib.config import deep_update, load_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_parses_paper_flags():
    parser, _ = build_project_cli_parser()
    args = parser.parse_args(
        [
            "--experiment",
            "study",
            "--system-sizes",
            "3,4,5",
            "--seeds",
            "0,1",
            "--grid",
            "paper",
        ]
    )
    assert (
        args.system_sizes == [3, 4, 5] and args.seeds == [0, 1] and args.grid == "paper"
    )


@pytest.mark.parametrize(
    "config_path",
    sorted((PROJECT_DIR / "configs").glob("*.json")),
    ids=lambda p: p.name,
)
def test_every_config_overlays_defaults(config_path):
    merged = deep_update(load_project_defaults(), load_config(config_path))
    assert merged["experiment"] in {
        "study",
        "training_size",
        "convention_check",
        "photonic_scaling",
    }
    assert "<<" not in json.dumps(merged), "placeholder left in a runnable config"
    for block in (
        "study",
        "conventions",
        "kernels",
        "engineered",
        "models",
        "simulation",
        "training_size",
        "photonic",
        "photonic_scaling",
    ):
        assert block in merged
