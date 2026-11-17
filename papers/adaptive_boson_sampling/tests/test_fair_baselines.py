"""Phase 5 fair-baseline configs mirror the ABS runs they are compared with."""

from __future__ import annotations

import json

import numpy as np
import pytest
from common import PROJECT_DIR
from lib.data import build_dataset
from lib.kernels import gaussian_kernel, median_heuristic_gamma

CONFIGS = PROJECT_DIR / "configs"
BASELINE_TWINS = {
    "classify_moons_xorder_classical.json": "classify_moons_xorder_original.json",
    "classify_moons_xorder_noise015_classical.json": (
        "classify_moons_xorder_noise015.json"
    ),
    "classify_moons_assignments_classical.json": (
        "classify_moons_assignments_original.json"
    ),
}


def _load(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text())


@pytest.mark.parametrize(("baseline", "abs_twin"), BASELINE_TWINS.items())
def test_baseline_matches_abs_protocol(baseline, abs_twin):
    ours, theirs = _load(baseline), _load(abs_twin)
    assert ours["dataset"] == theirs["dataset"]
    assert ours["dataset_repetitions"] == theirs["dataset_repetitions"]
    assert ours["evaluation"] == theirs["evaluation"]
    assert all(model["kind"] != "abs_digitized" for model in ours["models"])


def test_assignment_baseline_keeps_only_assignment_dependent_models():
    kinds = {
        (model["kind"], model.get("source"))
        for model in _load("classify_moons_assignments_classical.json")["models"]
    }
    assert kinds == {("index_gaussian", "outcome_index"), ("random_rank_d", None)}


def test_outcome_kernel_follows_the_assignment():
    # The information-matched baseline sees the same outcome index as the ABS
    # kernel, so changing the cluster-to-outcome rule must change its Gram matrix.
    base = _load("classify_moons_xorder_original.json")["dataset"]
    grams = []
    for assignment in ("x_order", "arc_order"):
        dataset = build_dataset({**base, "assignment": assignment})
        index = np.arange(15, dtype=float)[:, None]
        kernel = gaussian_kernel(index, median_heuristic_gamma(index))
        grams.append(kernel[np.ix_(dataset.outcome_index, dataset.outcome_index)])
    assert grams[0].shape == (200, 200)
    assert not np.allclose(grams[0], grams[1])


def _entry(per_variant: dict) -> dict:
    return {"partitions": {key: np.asarray(v, float) for key, v in per_variant.items()}}


def test_paired_difference_uses_shared_variants_and_tie_rule():
    from utils.compare_baselines import paired_difference

    rng = np.random.default_rng(0)
    first, second = {}, {}
    for variant in range(9):
        base = rng.uniform(0.5, 0.7, 100)
        first[("v", variant)] = base + 0.2
        second[("v", variant)] = base
    second[("extra", 0)] = np.zeros(100)
    clear = paired_difference(_entry(first), _entry(second))
    assert clear["n"] == 9 and clear["paired_over"] == "variants"
    assert clear["difference"] == pytest.approx(0.2)
    assert clear["verdict"] == "first higher"

    tiny = {key: value - 0.19 for key, value in first.items()}
    assert paired_difference(_entry(tiny), _entry(second))["verdict"] == (
        "no clear separation"
    )


def test_paired_difference_single_variant_pairs_over_partitions():
    from utils.compare_baselines import paired_difference

    accuracies = np.tile([1.0, 2 / 3, 1 / 3, 1.0], 25)
    result = paired_difference(
        _entry({("v", 0): accuracies}), _entry({("v", 0): accuracies[::-1]})
    )
    assert result["paired_over"] == "partitions" and result["n"] == 100
