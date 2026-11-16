from __future__ import annotations

import numpy as np
import pytest
from common import load_runtime_ready_config
from distributed_secure_kernels.lib.data import (
    build_dataloader,
    build_dataset,
    random_oversample,
)


def _cfg(name, protocol="paper"):
    cfg = load_runtime_ready_config()
    settings = {
        "wine": (None, 0.8, "standard"),
        "parkinsons": (0.6, 0.8, "standard"),
        "framingham": (0.6, 0.8, "standard"),
        "digits": (None, 100, "standard"),
    }[name]
    cfg["dataset"] = {
        "name": name,
        "oversample_ratio": settings[0],
        "subset_size": settings[1],
        "scaling": settings[2],
        "protocol": protocol,
    }
    return cfg


@pytest.mark.parametrize(
    "name, shape",
    [
        ("wine", (142, 13)),
        ("parkinsons", (188, 22)),
        ("framingham", (4066, 14)),
        ("digits", (100, 64)),
    ],
)
def test_one_batch_matches_authors_subset_sizes(name, shape):
    # Sizes printed in the authors' logs: Parkinson's (188, 22), Framingham (4066, 4066) kernel.
    features, labels = next(iter(build_dataloader(_cfg(name), split="train")))
    assert features.shape == shape
    assert labels.shape == (shape[0],)


def test_majority_rates_match_the_authors_collapsed_results():
    # Several logged quantum results equal these rates (0.6249 Framingham, 0.6276 Parkinson's).
    framingham = build_dataset(_cfg("framingham")).metadata["class_counts"]
    assert framingham == {"0": 2541, "1": 1525}
    assert round(2541 / 4066, 4) == 0.6249


def test_oversampling_follows_imblearn_count_rule():
    labels = np.array([0] * 48 + [1] * 147)  # Parkinson's class counts
    features = np.arange(len(labels), dtype=float)[:, None]
    over_x, over_y = random_oversample(features, labels, 0.6, np.random.default_rng(0))
    assert len(over_y) == 235  # int(147 * 0.6 - 48) = 40 added
    np.testing.assert_array_equal(over_x[:195, 0], features[:, 0])  # originals first
    assert set(over_x[195:, 0]) <= set(np.flatnonzero(labels == 0))


def test_paper_protocol_standardises_the_subset_and_corrected_does_not():
    paper = build_dataset(_cfg("wine")).features
    np.testing.assert_allclose(paper.mean(axis=0), 0.0, atol=1e-12)
    corrected = build_dataset(_cfg("wine", "corrected")).features
    assert abs(corrected.mean()) > 1.0  # raw Wine features


def test_corrected_protocol_does_not_oversample_before_the_split():
    data = build_dataset(_cfg("parkinsons", "corrected"))
    assert data.features.shape == (156, 22)  # 80 % of the 195 original rows
