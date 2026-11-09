"""Learning-rate schedule, early stopping and metric summaries."""

from __future__ import annotations

import common  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import pytest
import torch

from papers.QuLTSF.lib.training import (
    EarlyStopping,
    scheduled_learning_rate,
    summarise_errors,
)


def test_type3_schedule_matches_upstream():
    # Upstream: lr kept while epoch < 3, then lr * 0.9 ** (epoch - 3).
    rates = [scheduled_learning_rate("type3", 1e-4, epoch) for epoch in range(1, 7)]
    expected = [1e-4, 1e-4, 1e-4, 0.9e-4, 0.81e-4, 0.729e-4]
    np.testing.assert_allclose(rates, expected)


def test_type1_schedule_halves_every_epoch():
    rates = [scheduled_learning_rate("type1", 1e-4, epoch) for epoch in range(1, 4)]
    np.testing.assert_allclose(rates, [1e-4, 5e-5, 2.5e-5])


def test_unknown_schedule_fails():
    with pytest.raises(ValueError):
        scheduled_learning_rate("cosine", 1e-4, 1)


def test_early_stopping_needs_strict_improvement():
    model = torch.nn.Linear(1, 1)
    stopper = EarlyStopping(patience=2)
    assert stopper.step(1.0, model, 1)
    assert not stopper.step(1.0, model, 2)  # equal is not an improvement
    assert not stopper.should_stop
    assert not stopper.step(1.5, model, 3)
    assert stopper.should_stop
    assert stopper.best_epoch == 1


def test_summarise_errors_reports_upstream_drop_last_subset():
    errors = {"squared": np.arange(10.0), "absolute": np.ones(10)}
    summary = summarise_errors(errors, batch_size=4)
    assert summary["n_windows"] == 10
    assert summary["n_windows_drop_last"] == 8
    assert summary["mse"] == pytest.approx(4.5)
    assert summary["mse_drop_last"] == pytest.approx(3.5)
