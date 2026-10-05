"""Restore real-data checkpoints and combine the two task heads."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from common import MODEL_SETTINGS, PROJECT_DIR, make_config
from lib.inference import QCHIPP
from lib.runner import train_and_evaluate


@pytest.mark.parametrize("family", list(MODEL_SETTINGS))
def test_trained_heads_restore_predictions_and_combine(family):
    outdir = PROJECT_DIR / "outdir"
    outdir.mkdir(exist_ok=True)
    # Inference only loads checkpoints from this paper's outdir.
    with tempfile.TemporaryDirectory(dir=outdir) as temporary:
        runs = {}
        for task in ("binding", "immunogenic"):
            runs[task] = Path(temporary) / task
            train_and_evaluate(make_config(task, **MODEL_SETTINGS[family]), runs[task])
            # The shared runtime writes this snapshot; inference reads it back.
            snapshot = make_config(task, resolve_dtypes=False, **MODEL_SETTINGS[family])
            (runs[task] / "config_snapshot.json").write_text(json.dumps(snapshot))
        with (runs["binding"] / "predictions.csv").open() as handle:
            rows = [r for r in csv.DictReader(handle) if r["split"] == "test"][:4]
        model = QCHIPP(runs["binding"], runs["immunogenic"])
        result = model.predict([r["peptide"] for r in rows])
        np.testing.assert_allclose(
            result["binding_probability"],
            [float(r["probability"]) for r in rows],
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            result["positive"],
            (result["binding_probability"] >= 0.5)
            & (result["immunogenic_probability"] >= 0.5),
        )
