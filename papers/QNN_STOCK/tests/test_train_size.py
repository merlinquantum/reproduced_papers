"""Training-size subsets for the F step."""

import copy

import numpy as np
import pytest
from common import load_project_defaults
from lib.data import build_task, training_subset


def _cfg(**dataset):
    cfg = copy.deepcopy(load_project_defaults())
    cfg["dataset"].update(dataset)
    return cfg


def test_full_split_when_size_is_null():
    windows = np.arange(40).reshape(10, 4)
    assert training_subset(windows, _cfg(train_size=None)) is windows


def test_contiguous_block_reproducible_per_seed():
    windows = np.arange(400).reshape(100, 4)
    cfg = _cfg(train_size=10, train_subset_seed="seed")
    cfg["seed"] = 7
    first = training_subset(windows, cfg)
    second = training_subset(windows, copy.deepcopy(cfg))
    assert first.shape == (10, 4)
    np.testing.assert_array_equal(first, second)
    start = int(first[0, 0]) // 4
    np.testing.assert_array_equal(first, windows[start : start + 10])
    cfg["seed"] = 8
    assert not np.array_equal(training_subset(windows, cfg), first)


def test_size_out_of_range_rejected():
    with pytest.raises(ValueError):
        training_subset(np.zeros((5, 4)), _cfg(train_size=6, train_subset_seed=1))


def test_test_split_unchanged_by_subset():
    full = build_task(_cfg(train_size=None), "AAPL")
    small = build_task(_cfg(train_size=50, train_subset_seed=3), "AAPL")
    assert len(small.train_windows) == 50
    np.testing.assert_array_equal(small.test_windows, full.test_windows)
