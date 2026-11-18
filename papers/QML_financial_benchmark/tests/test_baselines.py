"""Focused tests for the additional fair baselines (Phase 1 plan, verified in Phase 5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from common import PROJECT_DIR, REPO_ROOT
from lib import data
from lib.models import MLPClassifier, count_parameters
from lib.tasks.directional import matched_ann_width
from lib.tasks.volatility import run_volatility, suggest_params


@pytest.mark.parametrize("budget, features", [(70, 3), (176, 7), (2170, 64), (9, 3)])
def test_matched_ann_width_is_closest_one_hidden_layer_count(budget, features):
    width = matched_ann_width(budget, features)
    gaps = {
        w: abs(count_parameters(MLPClassifier(features, [w])) - budget)
        for w in range(1, 200)
    }
    assert gaps[width] == min(gaps.values())


def test_wide_rbf_searches_gamma_continuously_while_paper_rbf_is_restricted():
    section = json.loads((PROJECT_DIR / "configs" / "e6_original.json").read_text())[
        "volatility"
    ]
    wide_study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    paper_study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    wide = [
        suggest_params(wide_study.ask(), "svr_rbf_wide", section)["gamma"]
        for _ in range(30)
    ]
    paper = {
        suggest_params(paper_study.ask(), "svr_rbf", section)["gamma"]
        for _ in range(30)
    }
    low, high = section["tuning"]["rbf_wide_gamma"]
    assert all(low <= gamma <= high for gamma in wide) and len(set(wide)) == 30
    assert paper <= {"scale", "auto"}


def test_persistence_and_har_forecasts(tmp_path: Path):
    cfg = json.loads((PROJECT_DIR / "configs" / "e6_reduced.json").read_text())
    section = cfg["volatility"] | {"models": ["persistence", "har"]}
    status = {}
    run_volatility(
        {"data_root": str(REPO_ROOT / "data")},
        section,
        tmp_path,
        status,
        lambda *args: None,
    )
    predictions = pd.read_csv(
        tmp_path / "test_predictions.csv", parse_dates=["date"]
    ).set_index("date")
    frame = data.build_volatility_frame(section["dataset"], REPO_ROOT / "data")
    target = frame["target"].to_numpy()
    n_train, step = (
        section["dataset"]["initial_train_days"],
        section["dataset"]["refit_every"],
    )
    # Every forecast is floored at its refit block's smallest positive training RV.
    floors = np.array(
        [
            target[: n_train + step * (i // step)][
                target[: n_train + step * (i // step)] > 0
            ].min()
            for i in range(len(predictions))
        ]
    )
    rv_now = frame.loc[predictions.index, "rv_now"].to_numpy()
    assert np.allclose(predictions["persistence"], np.maximum(rv_now, floors))
    # HAR-RV on the first refit block equals an independent least-squares fit.
    regressors = np.column_stack(
        [
            np.ones(len(frame)),
            frame["return_now"] ** 2,
            frame["rv_now"],
            frame["rv_month"],
        ]
    )
    coefficients = np.linalg.lstsq(regressors[:n_train], target[:n_train], rcond=None)[
        0
    ]
    first_block = predictions.index[:step]
    expected = regressors[frame.index.get_indexer(first_block)] @ coefficients
    assert np.allclose(
        predictions.loc[first_block, "har"], np.maximum(expected, floors[:step])
    )
