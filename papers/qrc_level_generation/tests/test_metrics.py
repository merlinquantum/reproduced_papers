from __future__ import annotations

import sys

from common import PROJECT_DIR

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lib import baselines, metrics  # noqa: E402


def test_originality_zero_for_original_self():
    original = [1, 2, 3, 4, 5, 1, 2, 3]
    rates = metrics.originality_rate([original], original, max_length=5)
    # Every window from the original level appears in the original level by
    # definition: originality must be zero everywhere.
    for value in rates.values():
        assert value == 0.0


def test_originality_one_for_unseen():
    original = [0, 0, 0]
    fake = [1, 2, 3, 4, 5]
    rates = metrics.originality_rate([fake], original, max_length=2)
    assert rates[2] == 1.0


def test_baselines_lengths_match_request():
    original = [0, 1, 2, 1, 0, 2, 1] * 5
    seqs = baselines.make_baseline_sequences(
        original=original,
        num_features=3,
        length=20,
        n_samples=5,
        seed=42,
    )
    assert all(len(s) == 20 for s in seqs["markov"])
    assert all(len(s) == 20 for s in seqs["uncorrelated"])


def test_mario_rules_detects_broken_pipe():
    # Two halves of a pipe must be adjacent (13 then 14).
    rules = metrics.mario_rules("1-2")
    # Sequence with broken transition (14 with no preceding 13)
    seq = [0, 14, 6]
    accum = metrics.find_broken_rules(seq, rules)
    assert "2" in accum
    # First slot is total count, second is broken.
    assert int(accum["2"][1]) > 0
