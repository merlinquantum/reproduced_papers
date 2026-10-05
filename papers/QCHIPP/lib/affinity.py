"""Diagnostic: how well DataS1's supplied affinity ranks separate each population.

DataS1 ships ``MT.BA_Rank``, a binding-affinity percentile rank predicted by an
external tool. It is a prediction, never ground truth. Scoring it against the
experimental labels on each released test split, with no fitting, shows how
much signal a standard affinity predictor carries for binding versus T-cell
response among confirmed binders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from .data import DATASET_SHA256, TASK_COLUMNS, ensure_dataset, read_rows, task_rows

# Conventional strong-binder cut-off on the percentile rank.
STRONG_BINDER_RANK = 2.0
DIAGNOSTIC_TASKS = ("binding", "mixed", "immunogenic")


def affinity_diagnostic(data_root: str | Path | None = None) -> dict:
    """Score ``-MT.BA_Rank`` against experimental labels on each test split.

    Parameters
    ----------
    data_root : str | Path | None
        Shared data root, as in :func:`lib.data.ensure_dataset`.
        Default value is None.

    Returns
    -------
    dict
        One row per population with AUROC and average precision of the
        negative rank, and F1 / accuracy of the rank < 2 decision.
    """
    rows = read_rows(ensure_dataset(data_root))
    results = []
    for task in DIAGNOSTIC_TASKS:
        split_column, label_column = TASK_COLUMNS[task]
        selected, _ = task_rows(rows, task)
        test = [r for r in selected if r[split_column] == "test"]
        truth = np.array([r[label_column] == "Positive" for r in test], dtype=int)
        rank = np.array([float(r["MT.BA_Rank"]) for r in test])
        predicted_binder = rank < STRONG_BINDER_RANK
        results.append(
            {
                "task": task,
                "test_count": len(test),
                "auroc": float(roc_auc_score(truth, -rank)),
                "average_precision": float(average_precision_score(truth, -rank)),
                "f1_rank_lt2": float(f1_score(truth, predicted_binder)),
                "accuracy_rank_lt2": float(accuracy_score(truth, predicted_binder)),
            }
        )
    return {
        "scope": (
            "supplied affinity-rank diagnostic on the released test splits; "
            "no fitting or test-set selection"
        ),
        "source_sha256": DATASET_SHA256,
        "rows": results,
    }
