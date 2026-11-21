"""Tiny end-to-end runs of every experiment type through the runner."""

from __future__ import annotations

import json

import pytest
from common import PROJECT_DIR, load_runtime_ready_config
from photonic_trainability.lib import runner


def _config(name, **overrides):
    cfg = load_runtime_ready_config()
    for key in ("cases", "observables", "monte_carlo"):
        cfg.pop(key, None)
    cfg.update(json.loads((PROJECT_DIR / "configs" / name).read_text()))
    cfg.update(overrides)
    return cfg


TINY_CASES = {
    "first_moment": (
        "fig10_first_moment_original.json",
        {"weights": [5, 6, 7, 8, 9, 10, 11], "densities": [0.5]},
    ),
    "variance_scaling": (
        "fig3_pnm_variances_original.json",
        {"modes": [10, 20, 30, 40, 50, 60]},
    ),
    "ratio_scaling": (
        "ratio_scaling_nu1_original.json",
        {"modes": [4, 6, 8]},
    ),
    "output_probability": (
        "output_probability_original.json",
        {
            "monte_carlo": {"unitaries": 40, "batch_size": 40, "batches": 4},
            "cases": [
                {
                    "name": "M=4",
                    "modes": 4,
                    "input": "alternating",
                    "bin_fraction_K": 2,
                    "random_bins": 1,
                }
            ],
        },
    ),
    "pnp_variances": (
        "fig5cd_pnp_variances_original.json",
        {"modes": [10], "instances": 2},
    ),
    "sample_complexity": (
        "sample_complexity_original.json",
        {
            "pairs": 50,
            "shot_factors": [1, 10],
            "gradient": {
                "modes": 4,
                "input": "alternating",
                "observable": {"type": "monomial", "exponents_by_mode": {"0": 2}},
                "inits": 3,
            },
        },
    ),
    "network_polynomial": (
        "network_polynomial_original.json",
        {
            "expansion_cases": [{"modes": 3, "hidden_layers": 2, "bias": True}],
            "cost_modes": [10],
            "cost_layers": [2],
        },
    ),
    "training_mesh": (
        "fig7_nn_training_reduced.json",
        {
            "modes": 4,
            "iterations": 2,
            "seeds": [0, 1],
            "learning_rate_grid": [0.01, 0.1],
            "shots": 100,
        },
    ),
    "training_loop": (
        "fig9_loop_training_substitute.json",
        {
            "modes": 6,
            "input": [1, 0, 1, 0, 0, 0],
            "observable": {
                "type": "random_pnp",
                "terms": 10,
                "min_order": 1,
                "max_order": 2,
                "coefficient_range": [-5, 5],
            },
            "iterations": 2,
            "seeds": [0],
            "learning_rate_grid": [0.01],
            "shots": 50,
        },
    ),
}


@pytest.mark.parametrize("case", sorted(TINY_CASES))
def test_experiment_runs_and_writes_contract_artifacts(case, tmp_path):
    name, overrides = TINY_CASES[case]
    cfg = _config(name, **overrides)
    runner.train_and_evaluate(cfg, tmp_path)
    status = json.loads((tmp_path / "run_status.json").read_text())
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    for artifact in (metrics.get("artifacts") or {}).values():
        if artifact:
            assert (tmp_path / artifact).exists(), artifact


def test_training_rejects_oversized_state_space(tmp_path):
    cfg = _config("fig7_nn_training_reduced.json", max_fock_dimension=10)
    with pytest.raises(ValueError, match="max_fock_dimension"):
        runner.train_and_evaluate(cfg, tmp_path)
