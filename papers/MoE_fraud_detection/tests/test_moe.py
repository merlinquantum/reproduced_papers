from __future__ import annotations

import numpy as np
import pytest
from lib.moe import (
    build_router_targets,
    combine_predictions,
    evaluate_binary,
    routed_fraction,
    youden_j_threshold,
)


def test_youden_j_threshold_perfect_separator():
    # Probabilities perfectly separate the two classes at 0.5.
    y = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    tau = youden_j_threshold(probs, y)
    # Any threshold strictly between 0.3 and 0.7 achieves J=1; roc_curve's
    # thresholds are drawn from the score values themselves.
    assert 0.3 <= tau <= 0.9


def test_build_router_targets_matches_eq15_by_hand():
    y_true = np.array([1, 1, 0, 0])
    # Expert 1 (primary): correct, correct, correct, wrong
    p1_hat = np.array([0.9, 0.9, 0.1, 0.9])
    tau1 = 0.5
    # Expert 2 (secondary): wrong, correct, wrong, correct
    p2_hat = np.array([0.1, 0.9, 0.9, 0.9])
    tau2 = 0.5

    z = build_router_targets(p1_hat, y_true, p2_hat, tau1, tau2, y_true)

    # i=0: p1 correct -> z=0 regardless of p2
    # i=1: p1 correct -> z=0
    # i=2: p1 correct -> z=0 (even though p2 wrong)
    # i=3: p1 WRONG (pred=1, true=0), p2 correct (pred=1? true=0 -> pred=1 != true -> wrong actually)
    expected = np.array([0, 0, 0, 0])
    np.testing.assert_array_equal(z, expected)


def test_build_router_targets_flags_expert2_saves_expert1_miss():
    y_true = np.array([1, 0])
    # Expert 1 wrong on both
    p1_hat = np.array([0.1, 0.9])
    tau1 = 0.5
    # Expert 2 correct on both
    p2_hat = np.array([0.9, 0.1])
    tau2 = 0.5

    z = build_router_targets(p1_hat, y_true, p2_hat, tau1, tau2, y_true)
    np.testing.assert_array_equal(z, np.array([1, 1]))


def test_build_router_targets_rejects_mismatched_labels():
    y_true = np.array([1, 0])
    y1 = np.array([1, 1])
    with pytest.raises(ValueError):
        build_router_targets(
            np.array([0.9, 0.1]), y1, np.array([0.9, 0.1]), 0.5, 0.5, y_true
        )


def test_combine_predictions_hard_mixture():
    p1_hat = np.array([0.1, 0.2, 0.3])
    p2_hat = np.array([0.9, 0.8, 0.7])
    router_probs = np.array([0.4, 0.6, 0.9])
    gamma = 0.5
    p_comb = combine_predictions(p1_hat, p2_hat, router_probs, gamma)
    # router_probs > 0.5 -> [False, True, True] -> use p1, p2, p2
    np.testing.assert_allclose(p_comb, [0.1, 0.8, 0.7])


def test_routed_fraction():
    router_probs = np.array([0.1, 0.6, 0.7, 0.4])
    frac = routed_fraction(router_probs, 0.5)
    assert frac == pytest.approx(0.5)


def test_run_single_split_router_split_options():
    from lib.data import generate_cv_splits, load_raw_data
    from lib.pipeline import _run_single_split

    X, y, _ = load_raw_data({})
    splits = generate_cv_splits(X, y, seed=42, n_repeats=1, max_splits=1)
    split = splits[0]
    base_cfg = {
        "model": {
            "backend": "gate",
            "n_qubits": 6,
            "xgboost": {"n_estimators": 50, "max_depth": 3},
        },
        "training": {"epochs": 1, "lr": 1e-3},
        "dataset": {"batch_size": 32},
        "evaluation": {"router_thresholds": [0.6]},
    }
    for router_split in ["analysis", "validation"]:
        cfg = {**base_cfg, "model": {**base_cfg["model"], "router_split": router_split}}
        result = _run_single_split(cfg, split, seed=42)
        assert result["router_split"] == router_split
        assert "aucpr" in result["xgboost_baseline"]

    with pytest.raises(ValueError):
        bad_cfg = {**base_cfg, "model": {**base_cfg["model"], "router_split": "bogus"}}
        _run_single_split(bad_cfg, split, seed=42)


def test_xgboost_early_stopping_uses_eval_set():
    from lib.pipeline import _make_xgb_classifier

    cfg = {
        "model": {
            "xgboost": {"n_estimators": 50, "max_depth": 3, "early_stopping_rounds": 5}
        }
    }
    clf = _make_xgb_classifier(cfg, seed=42, early_stopping=True)
    assert clf.get_params()["early_stopping_rounds"] == 5
    clf_no_es = _make_xgb_classifier(cfg, seed=42, early_stopping=False)
    assert clf_no_es.get_params()["early_stopping_rounds"] is None


def test_evaluate_binary_known_values():
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    result = evaluate_binary(probs, y)
    assert set(result.keys()) == {
        "aucpr",
        "ap",
        "precision",
        "recall",
        "routed_fraction",
    }
    # Perfect ranking: AP should be 1.0.
    assert result["ap"] == pytest.approx(1.0)
    assert result["aucpr"] == pytest.approx(1.0, abs=1e-6)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_youden_threshold_finite_on_anticorrelated_scores():
    from lib.moe import youden_j_threshold

    # Scores perfectly anti-correlated with labels: no real threshold has
    # positive Youden's J, so the pre-fix argmax landed on roc_curve's
    # synthetic np.inf "reject everything" threshold and silently corrupted
    # the router targets built from ``probs >= tau``.
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.9, 0.8, 0.2, 0.1])
    tau = youden_j_threshold(probs, y)
    assert np.isfinite(tau)
    # The returned threshold must be achievable by at least one sample.
    assert (probs >= tau).any()


def test_youden_threshold_unchanged_on_well_ordered_scores():
    from lib.moe import youden_j_threshold

    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    tau = youden_j_threshold(probs, y)
    assert np.isfinite(tau)
    # Perfect separation: the best finite threshold classifies perfectly.
    assert ((probs >= tau).astype(int) == y).all()
