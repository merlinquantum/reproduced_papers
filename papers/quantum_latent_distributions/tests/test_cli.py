"""CLI schema and config wiring."""

from __future__ import annotations

import json

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_project_defaults

CONFIG_DIR = PROJECT_DIR / "configs"


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_exposes_the_documented_flags():
    parser, arg_defs = build_project_cli_parser()
    flags = {flag for entry in arg_defs for flag in entry["flags"]}
    assert {"--experiment", "--latents", "--architecture", "--normalize"} <= flags
    args = parser.parse_args(["--experiment", "qm9", "--latents", "boson,gaussian"])
    assert args.experiment == "qm9"


def test_defaults_are_complete():
    cfg = load_project_defaults()
    for key in ("experiment", "latent", "dataset", "model", "training", "evaluation"):
        assert key in cfg, f"defaults.json is missing {key!r}"
    assert cfg["latent"]["normalize"] == "center", (
        "the paper centres the latents; changing this default changes the results"
    )


@pytest.mark.parametrize(
    "name",
    [
        "sampler_validation.json",
        "mixture_of_gaussians.json",
        "synthetic_datasets.json",
        "synthetic_datasets_fast.json",
        "qm9.json",
        "smoke.json",
    ],
)
def test_named_configs_are_valid_json_without_placeholders(name):
    text = (CONFIG_DIR / name).read_text(encoding="utf-8")
    assert "<<" not in text, "runnable configs must not contain placeholders"
    cfg = json.loads(text)
    assert "description" in cfg
