from __future__ import annotations

import sys

from common import PROJECT_DIR

sys.path.insert(0, str(PROJECT_DIR / "utils"))
from compare_fair import decide_paired, decide_unpaired  # noqa: E402


def test_paired_rule_requires_significance_for_ten_seeds():
    baseline = {seed: 1.0 + 0.01 * seed for seed in range(1, 11)}
    method = {seed: value * 0.5 for seed, value in baseline.items()}
    result = decide_paired(method, baseline)
    assert result["a_lower_seeds"] == 10
    assert result["wilcoxon_p"] < 0.05
    assert result["decision"] == "A_LOWER"


def test_paired_rule_five_seeds_is_only_consistent():
    baseline = {seed: 1.0 + 0.01 * seed for seed in range(1, 6)}
    method = {seed: value * 0.5 for seed, value in baseline.items()}
    assert decide_paired(method, baseline)["decision"] == "A_LOWER_CONSISTENT"
    assert decide_paired(baseline, method)["decision"] == "B_LOWER_CONSISTENT"


def test_paired_rule_within_tie_tolerance_is_no_separation():
    baseline = {seed: 1.0 + 0.01 * seed for seed in range(1, 11)}
    method = {seed: value * 0.97 for seed, value in baseline.items()}
    assert decide_paired(method, baseline)["decision"] == "NO_SEPARATION"


def test_paired_rule_three_seeds_never_decides():
    baseline = {1: 1.0, 2: 1.1, 3: 1.2}
    method = {seed: value * 0.01 for seed, value in baseline.items()}
    assert decide_paired(method, baseline)["decision"] == "NO_SEPARATION"


def test_unpaired_rule_needs_non_overlapping_ranges():
    method = {1: 0.10, 2: 0.12, 3: 0.11}
    assert decide_unpaired(method, {1: 0.5, 2: 0.6})["decision"] == "A_LOWER"
    assert decide_unpaired(method, {1: 0.11, 2: 0.6})["decision"] == "NO_SEPARATION"
