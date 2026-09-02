from __future__ import annotations

import math
import sys

from common import PROJECT_DIR

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import aggregate as agg  # noqa: E402


def _fake_run(orig_l2: float, orig_l10: float, broken2: float, broken3: float) -> dict:
    return {
        "qrc": {
            "QRC_T=1.0": {
                "originality": {"2": orig_l2, "5": 0.5, "10": orig_l10, "20": 1.0},
                "broken_rate_per_rule": {"2": broken2, "3": broken3},
            },
        },
        "baselines": {
            "markov": {
                "originality": {"2": 0.01, "10": 0.99},
                "broken_rate_per_rule": {"2": 0.0, "3": 0.0},
            },
            "uncorrelated": {
                "originality": {"2": 0.6, "10": 1.0},
                "broken_rate_per_rule": {"2": 0.7, "3": 0.05},
            },
        },
    }


def test_aggregate_point_mean_and_std():
    seed_metrics = [
        _fake_run(0.2, 0.7, 0.1, 0.0),
        _fake_run(0.3, 0.8, 0.2, 0.0),
        _fake_run(0.4, 0.9, 0.3, 0.0),
    ]
    aggregated = agg.aggregate_point(seed_metrics)
    assert "1.0" in aggregated
    o = aggregated["1.0"]["originality_L2"]
    assert math.isclose(o["mean"], 0.3, abs_tol=1e-9)
    # Population std of [0.2, 0.3, 0.4] is sqrt(((0.1)^2 + 0 + (0.1)^2)/3) ~ 0.08165
    assert math.isclose(o["std"], math.sqrt(0.02 / 3), abs_tol=1e-9)
    assert o["n"] == 3


def test_aggregate_point_handles_missing_metric_gracefully():
    seed_metrics = [
        {
            "qrc": {
                "QRC_T=1.0": {
                    "originality": {"2": float("nan")},
                    "broken_rate_per_rule": {},
                }
            }
        }
    ]
    aggregated = agg.aggregate_point(seed_metrics)
    assert aggregated["1.0"]["broken_rate_2"]["n"] == 0


def test_baseline_summary_picks_up_markov():
    seed_metrics = [_fake_run(0.1, 0.8, 0.2, 0.0)]
    out = agg.baseline_summary(seed_metrics)
    assert "markov" in out
    assert math.isclose(out["markov"]["broken_rate_2"]["mean"], 0.0)
