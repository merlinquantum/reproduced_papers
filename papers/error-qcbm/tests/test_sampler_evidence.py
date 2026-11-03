"""Reject stale, reduced-budget and mismatched production sampler evidence."""

import copy

import pytest
from lib.backend import DEMO_INPUT_STATE
from lib.sampler_validation import (
    SCHEMA,
    combine_evidence,
    fingerprint,
    validate_full_evidence,
)


def unit_report():
    """Build gate-test metadata only; never use this fixture as real evidence."""
    cases = {}
    for name in (
        "belenos_postselection",
        "belenos_recycling",
        "lossless_partial",
        "lossless_ideal",
    ):
        comparisons = []
        for label in ("initial_fresh_reference", "trained_probe_fresh_reference"):
            comparisons.append(
                {
                    "label": label,
                    "phase_sha256": label,
                    "counts_identical": True,
                    "rng_traces_identical": True,
                    "caller_rng_restored": True,
                    "probabilities_identical": True,
                    "loss_identical": True,
                    "max_count_difference": 0,
                    "stock_chunks": 50,
                    "vectorized_chunks": 50,
                    "stock_shots": 50000000,
                    "vectorized_shots": 50000000,
                }
            )
        cases[name] = {
            "shots": 50000000,
            "chunk_shots": 1000000,
            "profile": {
                "transmittance": 0.028 if name.startswith("belenos_") else 1.0,
                "indistinguishability": 1.0 if name == "lossless_ideal" else 0.729,
            },
            "comparisons": comparisons,
        }
    return {
        "schema": SCHEMA,
        "status": "passed",
        "environment": fingerprint(),
        "input_state": list(DEMO_INPUT_STATE),
        "cases": cases,
    }


@pytest.mark.parametrize(
    "change", ["version", "shots", "profile", "rng", "counts", "phases", "case"]
)
def test_gate_rejects_unqualified_or_corrupted_evidence(change):
    report = unit_report()
    if change == "version":
        report["environment"]["packages"]["torch"] = "different-version"
    elif change == "case":
        del report["cases"]["lossless_ideal"]
    else:
        case = report["cases"]["lossless_ideal"]
        if change == "shots":
            case["shots"] = 10000
        elif change == "profile":
            case["profile"]["indistinguishability"] = 0.84
        elif change == "rng":
            case["comparisons"][0]["rng_traces_identical"] = False
        elif change == "counts":
            case["comparisons"][0]["max_count_difference"] = 1
        else:
            case["comparisons"][0]["phase_sha256"] = "different-initialization"
    with pytest.raises(ValueError):
        validate_full_evidence(report)


def test_combination_requires_disjoint_complete_compatible_profiles():
    report = unit_report()
    a, b = copy.deepcopy(report), copy.deepcopy(report)
    a["cases"] = {k: v for k, v in a["cases"].items() if k.startswith("belenos_")}
    b["cases"] = {k: v for k, v in b["cases"].items() if not k.startswith("belenos_")}
    assert combine_evidence([a, b])["cases"] == report["cases"]
    with pytest.raises(ValueError, match="Duplicate"):
        combine_evidence([a, a, b])
    with pytest.raises(ValueError, match="all paper-scale"):
        combine_evidence([a])
