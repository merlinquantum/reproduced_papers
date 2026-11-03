"""Verify the official SDK boundary, fixed bin order and failure semantics."""

from itertools import product

import numpy as np
import perceval as pcvl
import pytest
from lib.recycling import recycling_mitigation, recycling_provenance


@pytest.mark.parametrize("modes,photons", [(4, 2), (6, 3), (12, 4)])
def test_official_distribution_is_aligned_to_the_complete_fixed_support(modes, photons):
    counts = {
        tuple([1] * photons + [0] * (modes - photons)): 23,
        tuple([1] * (photons - 1) + [0] * (modes - photons + 1)): 71,
        tuple([0] * (modes - photons + 2) + [1] * (photons - 2)): 113,
    }
    expected = pcvl.photon_recycling(
        pcvl.BSCount({pcvl.BasicState(list(s)): n for s, n in counts.items()}), photons
    )
    result = recycling_mitigation(counts, modes, photons)
    if modes <= 6:
        assert result["states"] == tuple(
            s for s in product((0, 1), repeat=modes) if sum(s) == photons
        )
    np.testing.assert_array_equal(
        result["probabilities"],
        [expected.get(pcvl.BasicState(list(s)), 0.0) for s in result["states"]],
    )
    assert np.isfinite(result["probabilities"]).all()
    assert (result["probabilities"] >= 0).all()
    assert result["probabilities"].sum() == pytest.approx(1)
    assert result["sector_counts"] == [23, 71, 113]
    assert result["diagnostics"].items() >= recycling_provenance().items()


def test_complete_ledger_reaches_official_api_once_without_resampling(monkeypatch):
    counts = {
        (1, 1, 1, 0, 0): 2,
        (1, 1, 0, 0, 0): 3,
        (0, 0, 0, 0, 1): 4,
        (0, 0, 0, 0, 0): 17,
    }
    official = pcvl.photon_recycling
    observed = []

    def capture(sdk_counts, ideal_photon_count):
        observed.append(
            ({tuple(s): n for s, n in sdk_counts.items()}, ideal_photon_count)
        )
        return official(sdk_counts, ideal_photon_count)

    monkeypatch.setattr(pcvl, "photon_recycling", capture)
    result = recycling_mitigation(counts, 5, 3)
    assert observed == [(counts, 3)]
    assert result["sector_counts"] == [2, 3, 4]
    assert result["ignored_count"] == 17
    assert result["postselected"].sum() == 1


def test_sdk_failure_is_not_replaced_by_another_estimator(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("SDK fit failed")

    monkeypatch.setattr(pcvl, "photon_recycling", fail)
    with pytest.raises(RuntimeError, match="SDK fit failed"):
        recycling_mitigation(
            {(1, 1, 0, 0): 10, (1, 0, 0, 0): 20, (0, 0, 0, 0): 30}, 4, 2
        )


def test_missing_sectors_fail_clearly():
    with pytest.raises(ValueError, match="sector_counts=\\[10, 0, 0\\]"):
        recycling_mitigation({(1, 1, 0, 0): 10}, 4, 2)


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_each_required_sector_is_mandatory_even_with_zero_count_keys(missing):
    counts = {(1, 1, 0, 0): 10, (1, 0, 0, 0): 20, (0, 0, 0, 0): 30}
    counts[list(counts)[missing]] = 0
    with pytest.raises(ValueError, match="nonempty zero/one/two"):
        recycling_mitigation(counts, 4, 2)


@pytest.mark.parametrize(
    "counts",
    [
        {(2, 0, 0, 0): 5},
        {(1, 0, 0, 0): -1},
        {(1, 0, 0, 0): 1.5},
        {(1, 1, 1, 0): 2},
        {(1.0, 0, 0, 0): 2},
    ],
)
def test_invalid_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        recycling_mitigation(counts, 4, 2)


@pytest.mark.parametrize("probability", [float("nan"), -1.0, 0.5])
def test_invalid_sdk_output_is_not_silently_repaired(monkeypatch, probability):
    monkeypatch.setattr(
        pcvl,
        "photon_recycling",
        lambda *args, **kwargs: {pcvl.BasicState([1, 1, 0, 0]): probability},
    )
    with pytest.raises(RuntimeError, match="invalid probabilities"):
        recycling_mitigation(
            {(1, 1, 0, 0): 10, (1, 0, 0, 0): 20, (0, 0, 0, 0): 30}, 4, 2
        )
