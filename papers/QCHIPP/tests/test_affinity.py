"""Supplied affinity-rank diagnostic on the released test splits."""

from lib.affinity import affinity_diagnostic
from lib.data import DATASET_SHA256


def test_affinity_diagnostic_covers_each_population():
    result = affinity_diagnostic()
    rows = {row["task"]: row for row in result["rows"]}
    assert result["source_sha256"] == DATASET_SHA256
    assert {task: row["test_count"] for task, row in rows.items()} == {
        "binding": 999,
        "mixed": 1000,
        "immunogenic": 1000,
    }
    # Affinity rank predicts binding well but not T-cell response among binders.
    assert rows["binding"]["auroc"] > 0.9
    assert rows["immunogenic"]["auroc"] < 0.5
