from __future__ import annotations

import pandas as pd
import pytest
from utils.report_search import _effective_convergence_runs
from utils.search_reservoirs import _extended_convergence_specs, stage3_specs

TUNED = {"modes": 8, "photons": 4, "scale": 37.6991, "leak": 0.3, "mem": 3}


def _stage4_results(capped_count: int) -> pd.DataFrame:
    specifications = stage3_specs(TUNED, [0, 1, 2], 3000, arm="conv")
    rows = []
    for index, specification in enumerate(specifications):
        rows.append(
            {
                "stage": "s4",
                "cfg": specification["cfg"],
                "tag": specification["tag"],
                "seed": specification["seed"],
                "epochs": 3000 if index < capped_count else 1200,
                "mse_test": float(index + 1),
            }
        )
    return pd.DataFrame(rows)


def test_extended_convergence_retries_only_capped_cells_without_overwriting():
    previous_results = _stage4_results(capped_count=42)

    specifications = _extended_convergence_specs(
        TUNED, [0, 1, 2], 10000, previous_results, 3000
    )

    assert len(specifications) == 42
    assert all(specification["epochs"] == 10000 for specification in specifications)
    assert all(
        specification["runid"].startswith("s4_cap10000__")
        for specification in specifications
    )


def test_extended_convergence_requires_complete_stage4_matrix():
    incomplete_results = _stage4_results(capped_count=42).iloc[:-1]

    with pytest.raises(ValueError, match="complete stage-4 matrix"):
        _extended_convergence_specs(TUNED, [0, 1, 2], 10000, incomplete_results, 3000)


def test_extended_convergence_requires_larger_cap():
    with pytest.raises(ValueError, match="must be greater"):
        _extended_convergence_specs(
            TUNED, [0, 1, 2], 3000, _stage4_results(capped_count=42), 3000
        )


def test_report_prefers_extended_result_for_each_completed_cell():
    stage4_results = _stage4_results(capped_count=1)
    extended_result = stage4_results.iloc[[0]].copy()
    extended_result["stage"] = "s4_cap10000"
    extended_result["epochs"] = 7200
    extended_result["mse_test"] = 0.25

    effective_results, largest_cap = _effective_convergence_runs(
        pd.concat([stage4_results, extended_result], ignore_index=True)
    )

    replaced = effective_results[
        (effective_results.cfg == extended_result.iloc[0].cfg)
        & (effective_results.tag == extended_result.iloc[0].tag)
        & (effective_results.seed == extended_result.iloc[0].seed)
    ]
    assert len(effective_results) == len(stage4_results)
    assert replaced.iloc[0].mse_test == 0.25
    assert largest_cap == 10000
