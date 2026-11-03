"""Check paired identities, strict cutoffs, duplicates and total-attempt denominators."""

import numpy as np
import pandas as pd
import pytest
from pep_q_gan.lib.evaluation import (
    SCORES,
    aggregate_attempts,
    score_attempts,
    summarize_panels,
    validate_attempts,
)


class Predictor:
    def predict(self, peptides, alleles, sample_names, **kwargs):
        assert kwargs == {
            "include_affinity_percentile": True,
            "verbose": 0,
            "throw": True,
        }
        assert (
            len(peptides) == 2
        )  # Repeated valid attempts are evaluated once per pair.
        result = pd.DataFrame(
            {"peptide": peptides, "peptide_num": [0, 1], "sample_name": sample_names}
        )
        for col in SCORES:
            result[col] = (
                [0.49, 0.5] if col == "presentation_percentile" else [1.0, 1.0]
            )
        return result.iloc[::-1]


def inputs():
    return pd.DataFrame(
        {
            "prior": ["dist"] * 4,
            "seed": [42] * 4,
            "hla": ["HLA-A02:01"] * 4,
            "attempt": [1, 2, 3, 4],
            "peptide": ["AAAAAAAAA", "AAAAAAAAA", "CCCCCCCCC", "XXXXXXXXX"],
        }
    )


def test_duplicates_threshold_and_invalid_denominator():
    scored = score_attempts(inputs(), Predictor(), ["HLA-A02:01"], 4)
    assert scored.peptide.tolist() == inputs().peptide.tolist()
    assert np.isnan(scored.presentation_percentile.iloc[-1])
    row = aggregate_attempts(scored, ["HLA-A02:01"], 4).iloc[0]
    assert row.generated == 4 and row.scorable == 3
    assert row["ps_rank_lt_0.5_count"] == 2
    assert row["ps_rank_lt_0.5_rate"] == 0.5
    assert row["ps_rank_lt_0.5_unique_count"] == 1


@pytest.mark.parametrize("change", ["drop", "duplicate", "hla", "prior"])
def test_reject_incomplete_or_mixed_scope(change):
    frame = inputs()
    if change == "drop":
        frame = frame.iloc[:-1]
    elif change == "duplicate":
        frame.loc[3, "attempt"] = 1
    elif change == "hla":
        frame.loc[3, "hla"] = "HLA-B07:02"
    else:
        frame.loc[3, "prior"] = "gbs"
    with pytest.raises(ValueError):
        validate_attempts(frame, ["HLA-A02:01"], 4)


def test_reject_mispaired_predictor():
    class Wrong(Predictor):
        def predict(self, **kwargs):
            result = super().predict(**kwargs)
            result["sample_name"] = "HLA-B07:02"
            return result

    with pytest.raises(ValueError, match="identities"):
        score_attempts(inputs(), Wrong(), ["HLA-A02:01"], 4)


def test_all_invalid_attempts_retained():
    frame = inputs()
    frame["peptide"] = "X"
    scored = score_attempts(frame, None, ["HLA-A02:01"], 4)
    row = aggregate_attempts(scored, ["HLA-A02:01"], 4).iloc[0]
    assert row.generated == 4 and row.scorable == 0 and row["ps_rank_lt_0.5_rate"] == 0


def test_seed_sem_is_not_over_hla_rows():
    rows = []
    for seed, hits in [(42, 1), (44, 3)]:
        for hla in ["HLA-A02:01", "HLA-B07:02"]:
            rows.append(
                {
                    "prior": "dist",
                    "seed": seed,
                    "hla": hla,
                    "generated": 4,
                    "scorable": 4,
                    "ps_rank_lt_0_5_count": hits,
                }
            )
    metrics = pd.DataFrame(rows).rename(
        columns={"ps_rank_lt_0_5_count": "ps_rank_lt_0.5_count"}
    )
    totals, summary = summarize_panels(metrics, ["HLA-A02:01", "HLA-B07:02"], 4)
    assert totals["ps_rank_lt_0.5_count"].tolist() == [2, 6]
    assert summary.iloc[0]["mean"] == 4 and summary.iloc[0]["sem"] == 2
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_panels(metrics.iloc[:-1], ["HLA-A02:01", "HLA-B07:02"], 4)


def test_shared_runner_evaluation_dispatch(monkeypatch, tmp_path):
    from pep_q_gan.lib import evaluation
    from pep_q_gan.lib.runner import train_and_evaluate

    def evaluate(settings, run_dir, **options):
        assert settings == {"input": "existing.csv"}
        assert options == {"device": "cpu", "threads": 1}
        assert run_dir == tmp_path
        return {"status": "dispatch_verified"}

    monkeypatch.setattr(evaluation, "evaluate_existing", evaluate)
    cfg = {
        "stage": "evaluate",
        "evaluation": {"input": "existing.csv"},
        "device": "cpu",
        "sampling": {"threads": 1},
    }
    assert train_and_evaluate(cfg, tmp_path)["status"] == "dispatch_verified"
