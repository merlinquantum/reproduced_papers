"""Tests for QPE_GNN shared-runtime configuration and CLI integration."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

PAPER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from runtime_lib.cli import apply_cli_overrides, build_cli_parser  # noqa: E402
from runtime_lib.config import load_config  # noqa: E402

sys.path.insert(0, str(PAPER_ROOT))
from lib.runner import _validate_config  # noqa: E402


def test_all_configs_have_shared_runtime_descriptions():
    for config_path in sorted((PAPER_ROOT / "configs").glob("*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert isinstance(config.get("description"), str), config_path.name
        assert config["description"].strip(), config_path.name


def test_all_configs_declare_valid_feasibility_or_pass_validation():
    defaults = load_config(PAPER_ROOT / "configs" / "defaults.json")
    for config_path in sorted((PAPER_ROOT / "configs").glob("*.json")):
        config = copy.deepcopy(defaults)
        config.update(load_config(config_path))
        if config["feasibility"]["status"] == "infeasible":
            with pytest.raises(ValueError, match="experiment config is infeasible"):
                _validate_config(config)
        else:
            _validate_config(config)


def test_cli_applies_model_training_and_dataset_overrides():
    cli_schema = json.loads((PAPER_ROOT / "cli.json").read_text(encoding="utf-8"))
    global_schema = json.loads(
        (REPOSITORY_ROOT / "runtime_lib" / "global_cli.json").read_text(
            encoding="utf-8"
        )
    )
    cli_schema["arguments"].extend(copy.deepcopy(global_schema["arguments"]))
    parser, argument_definitions = build_cli_parser(cli_schema)
    arguments = parser.parse_args(
        [
            "--model",
            "gcn",
            "--hidden-dim",
            "24",
            "--head",
            "graph_class",
            "--epochs",
            "3",
            "--train-frac",
            "0.7",
            "--length-range",
            "4,6",
            "--dataset-seed",
            "11",
            "--times",
            "0.1",
            "0.5",
            "--attention-dropout",
            "0.3",
            "--warmup-epochs",
            "2",
            "--pooling",
            "sum",
            "--parameter-budget",
            "500000",
            "--rrwp-dim",
            "21",
            "--qpe-dim",
            "20",
        ]
    )
    config = load_config(PAPER_ROOT / "configs" / "defaults.json")

    resolved_config = apply_cli_overrides(
        config,
        arguments,
        argument_definitions,
        PAPER_ROOT,
        REPOSITORY_ROOT,
    )

    assert resolved_config["model"] == "gcn"
    assert resolved_config["hidden_dim"] == 24
    assert resolved_config["head"] == "graph_class"
    assert resolved_config["epochs"] == 3
    assert resolved_config["train_frac"] == 0.7
    assert resolved_config["times"] == [0.1, 0.5]
    assert resolved_config["attention_dropout"] == 0.3
    assert resolved_config["warmup_epochs"] == 2
    assert resolved_config["pooling"] == "sum"
    assert resolved_config["parameter_budget"] == 500000
    assert resolved_config["rrwp_dim"] == 21
    assert resolved_config["qpe_dim"] == 20
    assert resolved_config["dataset_kwargs"]["length_range"] == [4, 6]
    assert resolved_config["dataset_kwargs"]["seed"] == 11


def test_synthetic_original_is_explicitly_infeasible():
    config = load_config(PAPER_ROOT / "configs" / "synthetic_original.json")

    assert config["feasibility"]["status"] == "infeasible"
    try:
        _validate_config(config)
    except ValueError as error:
        assert "dense Ising solver" in str(error)
    else:
        raise AssertionError("synthetic_original.json must fail validation")


def test_paper_synthetic_configs_share_graphs_and_splits():
    config_names = (
        "synthetic_quantum_original",
        "synthetic_rrwp_original",
        "synthetic_laplacian_original",
        "synthetic_gcn_original",
    )
    configs = [
        load_config(PAPER_ROOT / "configs" / f"{config_name}.json")
        for config_name in config_names
    ]
    for config in configs:
        assert config["dataset_kwargs"]["per_class"] == 400
        assert config["dataset_kwargs"]["length_range"] == [100, 400]
        assert config["dataset_kwargs"]["seed"] == 314159
        assert config["dataset_kwargs"]["crossing_range"] == [2, 2]
        assert config["split_seed"] == 1729
        assert config["epochs"] == 200
        assert config["seeds"] == [0, 1, 2, 3]
        if config["encoding"] != "none":
            assert config["dataset_kwargs"]["pe_dim"] == 20


def test_pattern_and_cluster_configs_use_node_classification():
    for dataset_name in ("pattern", "cluster"):
        for config_suffix in ("smoke", "original"):
            config = load_config(
                PAPER_ROOT / "configs" / f"{dataset_name}_{config_suffix}.json"
            )
            assert config["head"] == "node_class"


def test_original_benchmark_configs_use_full_grit_contracts():
    expected = {
        "zinc": (10, 64, 8, 32, 2000, "sum", 21, 20, 0.2, 50),
        "mnist": (3, 52, 4, 16, 200, "mean", 18, 18, 0.5, 5),
        "cifar10": (3, 52, 4, 16, 200, "mean", 18, 18, 0.5, 5),
        "pattern": (10, 64, 8, 32, 100, "none", 21, 20, 0.2, 5),
        "cluster": (16, 48, 8, 16, 100, "none", 32, 32, 0.5, 5),
    }
    expected_encodings = {
        "rrwp": "rrwp",
        "cqrw1": "rrwp+cqrw1",
        "qirw2": "rrwp+qirw2",
    }
    for dataset_name, contract in expected.items():
        (
            depth,
            node_dim,
            num_heads,
            batch_size,
            epochs,
            pooling,
            rrwp_dim,
            qpe_dim,
            attention_dropout,
            warmup_epochs,
        ) = contract
        for variant, encoding in expected_encodings.items():
            config = load_config(
                PAPER_ROOT / "configs" / f"{dataset_name}_{variant}_original.json"
            )
            assert config["model"] == "grit"
            assert config["encoding"] == encoding
            assert "pe_dim" not in config
            assert config["rrwp_dim"] == rrwp_dim
            assert config["depth"] == depth
            assert config["node_dim"] == node_dim
            assert config["num_heads"] == num_heads
            assert config["batch_size"] == batch_size
            assert config["epochs"] == epochs
            assert config["pooling"] == pooling
            assert config["attention_dropout"] == attention_dropout
            assert config["warmup_epochs"] == warmup_epochs
            assert config["seeds"] == [0, 1, 2, 3]
            if variant == "rrwp":
                assert "qpe_dim" not in config
            else:
                assert config["qpe_dim"] == qpe_dim
            if variant == "cqrw1":
                assert config["qpe_initial_distribution"] == "local"
                assert config["qpe_min_time"] == 0.1
                assert config["qpe_max_time"] == pytest.approx(math.pi)
            if variant == "qirw2":
                assert config["qpe_initial_distribution"] == "adjacency"


def test_combined_encoding_rejects_implicit_dimension_split():
    config = load_config(PAPER_ROOT / "configs" / "defaults.json")
    config.update(
        {
            "encoding": "rrwp+qirw2",
            "pe_dim": 41,
            "qpe_initial_distribution": "adjacency",
        }
    )

    with pytest.raises(ValueError, match="requires rrwp_dim and qpe_dim"):
        _validate_config(config)
