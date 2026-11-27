"""Phase 5 fair comparison: selection split, tuned training, B4 reference, selection rule."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys

import common  # noqa: F401  (sets sys.path)
import numpy as np
import pytest
from common import PROJECT_DIR, load_runtime_ready_config
from fermi_dirac_neurons.lib.data import selection_split
from fermi_dirac_neurons.lib.linear_reference import (
    fit_logistic,
    fit_ridge,
    logistic_loss,
    pauli_labels_up_to_weight,
)
from fermi_dirac_neurons.lib.training import train_models

VALIDATOR = os.environ.get("ALPS_LOGGING_VALIDATOR", "")


def test_selection_split_is_seeded_disjoint_and_complete():
    train, selection = selection_split(13, 0.2, seed=4)
    assert len(selection) == 3 and len(train) == 10
    assert set(train).isdisjoint(selection)
    assert sorted(set(train) | set(selection)) == list(range(13))
    again = selection_split(13, 0.2, seed=4)
    np.testing.assert_array_equal(train, again[0])
    assert not np.array_equal(selection, selection_split(13, 0.2, seed=5)[1])
    assert len(selection_split(2, 0.2, seed=0)[1]) == 1
    with pytest.raises(ValueError, match="fraction"):
        selection_split(10, 1.0, seed=0)


class _Quadratic:
    """Toy model f = 0, loss = |theta - 1|^2, for optimizer checks."""

    labels = ["a", "b"]
    n_parameters = 2

    def __init__(self, blow_up_at=None):
        self.calls = 0
        self.blow_up_at = blow_up_at

    def loss_and_gradient(self, theta, data, loss):
        self.calls += 1
        if self.blow_up_at is not None and self.calls > self.blow_up_at:
            return math.nan, np.full(2, math.nan)
        return float(np.sum((theta - 1.0) ** 2)), 2.0 * (theta - 1.0)


class _Data:
    targets = np.zeros(1)


def _train(tmp_path, models, training, rates=None):
    import fermi_dirac_neurons.lib.training as training_module

    original = training_module.evaluate_model
    training_module.evaluate_model = lambda model, theta, data, kind, loss: {
        "loss": float(np.sum((theta - 1.0) ** 2))
    }
    try:
        return train_models(
            models,
            {name: np.zeros(2) for name in models},
            _Data(),
            {"train": _Data()},
            {"loss": "squared", "epochs": 50, "eval_every": 0, **training},
            "regression",
            tmp_path / "history.csv",
            learning_rates=rates,
        )
    finally:
        training_module.evaluate_model = original


def test_per_model_learning_rates_and_adam(tmp_path):
    result = _train(
        tmp_path,
        {"slow": _Quadratic(), "still": _Quadratic()},
        {"learning_rate": 0.1},
        rates={"still": 0.0},
    )
    assert result["learning_rates"] == {"slow": 0.1, "still": 0.0}
    assert result["final_parameters"]["still"] == [0.0, 0.0]
    assert all(value > 0.9 for value in result["final_parameters"]["slow"])
    assert result["optimizer"] == "gd"
    adam = _train(
        tmp_path, {"m": _Quadratic()}, {"learning_rate": 0.05, "optimizer": "adam"}
    )
    assert adam["optimizer"] == "adam"
    assert all(abs(value - 1.0) < 0.2 for value in adam["final_parameters"]["m"])
    assert adam["training_seconds"]["m"] > 0


def test_divergence_freezes_one_model_and_keeps_the_others(tmp_path):
    result = _train(
        tmp_path,
        {"bad": _Quadratic(blow_up_at=10), "good": _Quadratic()},
        {"learning_rate": 0.1, "on_nonfinite": "freeze"},
    )
    assert result["diverged_at_epoch"] == {"bad": 10, "good": None}
    assert math.isnan(result["final_train_loss_before_last_update"]["bad"])
    assert all(np.isfinite(result["final_parameters"]["bad"]))
    assert result["final_train_loss_before_last_update"]["good"] < 1e-6
    with pytest.raises(FloatingPointError):
        _train(tmp_path, {"bad": _Quadratic(blow_up_at=3)}, {"learning_rate": 0.1})


@pytest.mark.parametrize(("n", "w"), [(2, 1), (3, 2), (4, 2), (3, 3)])
def test_pauli_label_count(n, w):
    labels = pauli_labels_up_to_weight(n, w)
    expected = sum(math.comb(n, k) * 3**k for k in range(1, w + 1))
    assert len(labels) == len(set(labels)) == expected
    assert all(sum(letter != "I" for letter in label) <= w for label in labels)


def test_ridge_is_exact_for_a_linear_target_and_logistic_gradient_is_right():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 5))
    targets = 0.3 + features @ np.array([1.0, -2.0, 0.5, 0.0, 0.25])
    coefficients = fit_ridge(features, targets, 1e-12)
    np.testing.assert_allclose(
        coefficients, [0.3, 1.0, -2.0, 0.5, 0.0, 0.25], atol=1e-8
    )
    labels = np.where(features[:, 0] + 0.1 * rng.normal(size=40) > 0, 1.0, -1.0)
    fitted = fit_logistic(features, labels, 1e-2)
    for index in range(len(fitted)):
        shift = np.zeros_like(fitted)
        shift[index] = 1e-6
        numeric = (
            logistic_loss(fitted + shift, features, labels)
            + 0.5e-2 * np.sum((fitted + shift)[1:] ** 2)
            - logistic_loss(fitted - shift, features, labels)
            - 0.5e-2 * np.sum((fitted - shift)[1:] ** 2)
        ) / 2e-6
        assert abs(numeric) < 1e-4  # stationary point of the penalized objective


def test_full_tomography_reference_is_exact_with_informationally_complete_data():
    """Tr[O rho] is linear in rho: full-weight Pauli features fit it from enough random states."""
    from fermi_dirac_neurons.lib.linear_reference import evaluate_linear_reference
    from fermi_dirac_neurons.lib.models import Dataset
    from fermi_dirac_neurons.lib.pauli import pauli_string

    rng = np.random.default_rng(1)

    def random_states(count):
        vectors = rng.normal(size=(count, 4)) + 1j * rng.normal(size=(count, 4))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.einsum("ma,mb->mab", vectors, vectors.conj())

    observable = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    observable = observable + observable.conj().T
    splits = {}
    for name, count in (("train", 60), ("selection", 10), ("validation", 50)):
        states = random_states(count)
        targets = np.real(np.einsum("ab,mba->m", observable, states))
        splits[name] = Dataset(states, targets)
    labels = pauli_labels_up_to_weight(2, 2)
    terms = np.array([pauli_string(label) for label in labels])
    (record,) = evaluate_linear_reference(splits, labels, terms, "regression", [1e-10])
    assert record["n_features"] == 15
    assert record["validation_mse"] < 1e-12


def test_reference_task_runs_on_a_paper_task(tmp_path):
    _, metrics = _run(tmp_path, "p5_ref_fig9_heis_3q")
    assert {record["n_features"] for record in metrics["variants"]["b4_w3"]} == {63}
    assert {record["n_features"] for record in metrics["variants"]["b4_w2"]} == {36}


def test_phase5_task_configs_combine_paper_and_fair_baselines():
    for task, expected in (
        (
            "fig7_tanh_7q",
            {"quantum_tfim", "classical_im", "b2_pauli_features", "b3_quantum_linear"},
        ),
        (
            "fig8_logistic_5q",
            {"quantum_heisenberg", "classical_fcim", "b2_pauli_features"},
        ),
        (
            "fig9_heis_2q",
            {
                "quantum_heis_tanh",
                "quantum_heis_linear",
                "classical_fcim",
                "b2_pauli_features",
            },
        ),
    ):
        config = json.loads(
            (PROJECT_DIR / "configs" / f"p5_sel_{task}_eta3.json").read_text()
        )
        assert {model["name"] for model in config["models"]} == expected
        assert config["training"]["learning_rate"] == 3
        assert config["training"]["on_nonfinite"] == "freeze"
        assert config["data"]["selection_fraction"] == 0.2
        original = json.loads(
            (PROJECT_DIR / "configs" / f"{task}_original.json").read_text()
        )
        assert config["data"] | {"selection_fraction": None} == original["data"] | {
            "selection_fraction": None
        }
    photonic = json.loads(
        (PROJECT_DIR / "configs" / "p5_phsel_m6_n3_lr0.03.json").read_text()
    )
    assert photonic["training"]["optimizer"] == "adam"
    assert photonic["training"]["learning_rate"] == 0.03
    assert photonic["photonic"]["selection_fraction"] == 0.2


def test_selection_rule():
    from fermi_dirac_neurons.utils.select_p5 import select

    values = {
        ("task", "model"): {
            0.1: {0: -2.0, 1: -2.1, 2: -1.9},
            1.0: {0: -3.0, 1: math.nan, 2: -3.0},  # diverged once: ineligible
            3.0: {0: -2.5, 1: -2.4, 2: -2.6},
            10.0: {0: -2.5, 1: -2.4, 2: -2.6},  # exact tie: smaller rate wins
        },
        ("task", "broken"): {0.1: {0: math.nan, 1: -1.0, 2: -1.0}},
    }
    rows = {row["model"]: row for row in select(values, [0, 1, 2])}
    assert rows["model"]["selected_learning_rate"] == 3.0
    assert rows["broken"]["status"] == "FAILED"


def test_linear_reference_memory_estimate_covers_terms():
    from fermi_dirac_neurons.utils import run_sweep

    estimate = run_sweep.estimate_memory_gib(
        run_sweep.resolved_config("p5_ref_fig8_logistic_7q")
    )
    states = 1500 * 4**7 * 16 / 2**30
    assert estimate > 0.4 + states


def _run(tmp_path, name, epochs=None):
    from fermi_dirac_neurons.lib import runner

    from runtime_lib.config import deep_update

    overlay = json.loads((PROJECT_DIR / "configs" / f"{name}.json").read_text())
    cfg = deep_update(load_runtime_ready_config(), copy.deepcopy(overlay))
    if "models" in overlay:
        cfg["models"] = overlay["models"]
    if epochs is not None:
        cfg["training"]["epochs"] = epochs
        cfg["training"]["eval_every"] = 2
    run_dir = tmp_path / name
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)
    assert (
        json.loads((run_dir / "run_status.json").read_text())["status"] == "COMPLETED"
    )
    return run_dir, json.loads((run_dir / "metrics.json").read_text())


@pytest.mark.parametrize(
    ("name", "training"),
    [
        ("p5_sel_fig7_tanh_2q_eta1", True),
        ("p5_sel_fig8_logistic_2q_eta1", True),
        ("p5_ref_fig8_logistic_2q", False),
        ("p5_phsel_m4_n2_lr0.01", True),
    ],
)
def test_phase5_tasks_write_contract_artifacts(tmp_path, name, training):
    run_dir, metrics = _run(tmp_path, name, epochs=5 if training else None)
    if training:
        for record in metrics["models"].values():
            assert {"selection", "validation", "train"} <= set(record["final"])
            assert record["diverged_at_epoch"] is None
            assert record["training_seconds"] > 0
        status = json.loads((run_dir / "run_status.json").read_text())
        sizes = {
            name: split["n_states"]
            for name, split in status["dataset"]["splits"].items()
        }
        assert sizes["selection"] >= 1 and sizes["train"] > sizes["selection"]
    else:
        assert set(metrics["variants"]) == {"b4_w2"}
    if VALIDATOR and os.path.exists(VALIDATOR):
        command = [
            sys.executable,
            VALIDATOR,
            "--run-dir",
            str(run_dir),
            "--require-evaluation",
        ]
        if training:
            command.append("--require-training")
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def test_summary_excludes_diverged_models_but_rejects_other_nonfinite(tmp_path):
    from fermi_dirac_neurons.utils.summarize_sweep import summarize

    def metrics(diverged, value):
        return {
            "task": "train",
            "models": {
                "m": {
                    "n_parameters": 2,
                    "diverged_at_epoch": diverged,
                    "final": {"validation": {"mse": value}},
                }
            },
        }

    runs = []
    for seed, (diverged, value) in enumerate([(None, 0.1), (7, math.inf), (None, 0.3)]):
        path = tmp_path / f"m{seed}.json"
        path.write_text(json.dumps(metrics(diverged, value)))
        runs.append(
            {
                "run_id": f"r{seed}",
                "config": "c",
                "seed": seed,
                "status": "COMPLETED",
                "metrics_path": path.name,
            }
        )
    status = {"sweep_id": "s", "expected_runs": 3, "runs": runs}
    summary = json.loads(summarize(tmp_path, status).read_text())
    (row,) = summary["rows"]
    assert row["diverged_seeds"] == [1]
    assert row["validation_mse_median"] == pytest.approx(0.2)
    (tmp_path / "m1.json").write_text(json.dumps(metrics(None, math.inf)))
    with pytest.raises(ValueError, match="Non-finite"):
        summarize(tmp_path, status)
