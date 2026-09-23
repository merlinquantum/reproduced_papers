"""Every shipped config must load and satisfy the runner's contract."""

import json
from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.json"))


def test_configs_exist():
    assert CONFIGS, f"no configs found under {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_has_required_keys(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for section, keys in (
        ("problem", ("family", "m", "instances")),
        ("circuit", ("delays",)),
        ("training", ("updates", "samples", "lr_theta", "lr_alpha", "shift")),
        ("method", ("name",)),
    ):
        assert section in cfg, f"{path.name} lacks {section}"
        for key in keys:
            assert key in cfg[section], f"{path.name} lacks {section}.{key}"
    assert cfg["problem"]["family"] in ("knapsack", "tsp")
    assert "<<" not in path.read_text(encoding="utf-8"), (
        "runnable configs must not carry placeholders"
    )


@pytest.mark.parametrize(
    "path", [p for p in CONFIGS if p.stem.endswith("_original")], ids=lambda p: p.stem
)
def test_paper_accurate_configs_use_paper_hyperparameters(path):
    """The `_original` configs are the reproducibility artifact and must stay exact.

    The paper states N = 200 updates, S = 50 samples, learning rates 0.01 and
    0.05, and delay lines 1-3-9. Reduced runs belong in `_reduced` configs or CLI
    overrides, never here.
    """
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["training"]["updates"] == 200
    assert cfg["training"]["samples"] == 50
    assert cfg["training"]["lr_theta"] == 0.01
    assert cfg["training"]["lr_alpha"] == 0.05
    assert cfg["circuit"]["delays"] == [1, 3, 9]
    assert cfg["problem"]["instances"] == 100
