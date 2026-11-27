from __future__ import annotations

import json

import pytest
import torch
from common import AUTHORS_SETTING_1, PROJECT_DIR, REPO_ROOT, SEMEION_FILE
from lib.data import build_dataloader, build_splits

pytestmark = pytest.mark.skipif(
    not SEMEION_FILE.exists(), reason="Semeion data not downloaded"
)


def _cfg():
    cfg = json.loads((PROJECT_DIR / "configs/e1_8clients_original.json").read_text())
    cfg["data_root"] = str(REPO_ROOT / "data")
    return cfg


def test_one_training_batch_has_expected_shape():
    batch = next(iter(build_dataloader(_cfg(), "train")))
    assert batch[0].shape == (32, 256), batch[0].shape
    assert batch[1].shape == (32,)


def test_split_sizes_and_angle_encoding():
    splits = build_splits(_cfg())
    assert len(splits["train"][1]) == 1274 and len(splits["test"][1]) == 319
    assert set(splits["train"][0].unique().tolist()) <= {0.0, torch.pi / 4}


@pytest.mark.skipif(
    not AUTHORS_SETTING_1.exists(), reason="authors' assets not downloaded"
)
def test_split_equals_authors_setting_1():
    splits = build_splits(_cfg())
    reference = torch.load(
        AUTHORS_SETTING_1 / "data_tensors.pt", weights_only=True, map_location="cpu"
    )
    assert torch.allclose(splits["train"][0], reference["data_tr"])
    assert torch.equal(splits["test"][1], reference["label_te"])
