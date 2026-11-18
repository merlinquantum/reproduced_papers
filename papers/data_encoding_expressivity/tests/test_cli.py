from __future__ import annotations

import json

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_parses_paper_flags():
    parser, _ = build_project_cli_parser()
    args = parser.parse_args(
        ["--steps", "5", "--lr", "0.1", "--batch-size", "10", "--seeds", "0,1,2"]
    )
    assert (
        args.steps == 5
        and args.lr == 0.1
        and args.batch_size == 10
        and args.seeds == [0, 1, 2]
    )


def test_defaults_run_the_spectrum_experiment():
    cfg = load_runtime_ready_config()
    assert cfg["experiment"] == "spectrum"
    assert cfg["dtype"].label == "float64"


@pytest.mark.parametrize(
    "path", sorted((PROJECT_DIR / "configs").glob("*.json")), ids=lambda p: p.name
)
def test_every_config_names_a_known_experiment_and_has_its_section(path):
    cfg = json.loads(path.read_text())
    experiment = cfg.get("experiment")
    assert experiment in {
        "fit",
        "coefficients",
        "spectrum",
        "universality",
        "photonic_spectrum",
        "coefficient_reach",
    }
    assert experiment in cfg, (
        f"{path.name} selects {experiment!r} but has no {experiment!r} section"
    )
    assert "<<" not in path.read_text(), f"{path.name} contains a placeholder"
