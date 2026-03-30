from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = None
for parent in Path(__file__).resolve().parents:
    if parent.name == "q_random_kitchen_sinks":
        PROJECT_ROOT = parent
        break
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate q_random_kitchen_sinks project root.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from lib import training  # noqa: E402


def test_resolve_scaling_variants() -> None:
    assert training._resolve_scaling("1/sqrt(R)", r=4) == 0.5
    assert training._resolve_scaling("sqrt(R)", r=9) == 3.0
    assert training._resolve_scaling("sqrt(R)+3", r=4) == 5.0
    assert training._resolve_scaling(2.5, r=3) == 2.5


def test_hybrid_training_data_generated() -> None:
    train_proj = np.linspace(0, 1, 6).reshape(3, 2)
    test_proj = np.linspace(1, 2, 4).reshape(2, 2)
    cfg = {"hybrid_model_data": "generated"}
    synth_train, synth_test = training._hybrid_training_data(train_proj, test_proj, cfg)
    assert synth_train.shape[1] == 2
    assert synth_test.shape[1] == 2
