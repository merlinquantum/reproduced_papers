# ruff: noqa: N999
from __future__ import annotations

import json

from photonic_QCNN.lib import runtime_cli
from photonic_QCNN.lib.run_MNIST import _prepare_output_dir, _prepare_random_states
from photonic_QCNN.lib.runner import load_config


def test_apply_batch_size_updates_all_runs():
    cfg: dict[str, object] = {}
    runtime_cli.apply_batch_size(cfg, 7, {})
    runs = cfg.get("runs", {})
    assert set(runs.keys()) == set(runtime_cli.DATASET_CHOICES)
    assert all(entry.get("batch_size") == 7 for entry in runs.values())


def test_apply_n_runs_updates_all_runs():
    cfg: dict[str, object] = {}
    runtime_cli.apply_n_runs(cfg, 3, {})
    runs = cfg.get("runs", {})
    assert set(runs.keys()) == set(runtime_cli.DATASET_CHOICES)
    assert all(entry.get("n_runs") == 3 for entry in runs.values())


def test_load_config_merges_overrides(tmp_path):
    overrides = {
        "training": {"epochs": 99},
        "runs": {"MNIST": {"batch_size": 8}},
    }
    cfg_path = tmp_path / "override.json"
    cfg_path.write_text(json.dumps(overrides), encoding="utf-8")
    config = load_config(cfg_path)
    assert config["training"]["epochs"] == 99
    assert config["runs"]["MNIST"]["batch_size"] == 8


def test_prepare_output_dir_flattens_dataset(tmp_path):
    output_dir = _prepare_output_dir(tmp_path, "MNIST")
    assert output_dir == tmp_path / "MNIST"
    assert output_dir.exists()


def test_prepare_random_states_extends():
    cfg = {"n_runs": 4, "random_states": [1], "seed": 123}
    states = _prepare_random_states(cfg)
    assert len(states) == 4
