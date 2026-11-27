from __future__ import annotations

import numpy as np
from common import load_runtime_ready_config
from photonic_trainability.lib.data import build_dataloader


def _loader_cfg(**overrides):
    cfg = load_runtime_ready_config()
    cfg.update({"modes": 5, "monte_carlo": {"unitaries": 7, "batch_size": 3}})
    cfg.update(overrides)
    return cfg


def test_one_batch_has_expected_shape_and_is_unitary():
    loader = build_dataloader(_loader_cfg(), split="train")
    batch = next(iter(loader))
    assert batch.shape == (3, 5, 5), batch.shape
    identity = np.eye(5)
    for unitary in batch:
        np.testing.assert_allclose(unitary.conj().T @ unitary, identity, atol=1e-12)


def test_loader_covers_count_and_is_deterministic():
    loader = build_dataloader(_loader_cfg(), split="train")
    first = np.concatenate(list(loader))
    second = np.concatenate(list(loader))
    assert first.shape == (7, 5, 5)
    assert len(loader) == 3
    np.testing.assert_array_equal(first, second)


def test_splits_and_cases_give_independent_streams():
    train = next(iter(build_dataloader(_loader_cfg(), split="train")))
    other_split = next(iter(build_dataloader(_loader_cfg(), split="monte_carlo")))
    other_case = next(iter(build_dataloader(_loader_cfg(case_index=1), split="train")))
    assert not np.allclose(train, other_split)
    assert not np.allclose(train, other_case)


def test_haar_ensemble_first_moment_of_entries():
    # For Haar U in U(M): E|U_ij|^2 = 1/M and E|U_ij|^4 = 2/(M(M+1)).
    cfg = _loader_cfg(modes=4, monte_carlo={"unitaries": 4000, "batch_size": 500})
    unitaries = np.concatenate(list(build_dataloader(cfg, split="train")))
    squared = np.abs(unitaries[:, 0, 1]) ** 2
    assert abs(squared.mean() - 1 / 4) < 0.01
    assert abs((squared**2).mean() - 2 / 20) < 0.01
