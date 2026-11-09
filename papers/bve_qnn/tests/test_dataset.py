from __future__ import annotations

import sys

import numpy as np
from common import PROJECT_DIR, REPO_ROOT

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def test_committed_subset_exists_and_is_reshape_compatible():
    from lib.data import DEFAULT_DATASET_FILENAME

    path = REPO_ROOT / "data" / "bve_qnn" / DEFAULT_DATASET_FILENAME
    assert path.exists(), f"missing committed subset {path}"
    assert path.stat().st_size < 50_000, "subset should stay tiny for git"

    data = np.load(path)
    features = data["supervised_features"]
    psi = data["psi_qcl_training"]
    assert features.ndim == 2 and features.shape[1] == 4
    assert features.shape[0] == int(np.prod(psi.shape))
    assert data["supervised_targets"].shape[0] == features.shape[0]


def test_defaults_config_points_at_subset():
    from lib.data import DEFAULT_DATASET_FILENAME, load_dataset, resolve_dataset_path

    cfg = {
        "dataset": {"filename": DEFAULT_DATASET_FILENAME},
        "data_root": str(REPO_ROOT / "data"),
    }
    path = resolve_dataset_path(cfg)
    assert path.name == DEFAULT_DATASET_FILENAME
    loaded = load_dataset(cfg)
    assert loaded["features_tensor"].shape[0] == int(np.prod(loaded["psi_shape"]))


def test_explicit_dataset_root_is_not_silently_ignored(tmp_path):
    from lib.data import DEFAULT_DATASET_FILENAME, resolve_dataset_path

    cfg = {
        "dataset": {"root": str(tmp_path), "filename": DEFAULT_DATASET_FILENAME},
        "data_root": str(REPO_ROOT / "data"),
    }

    assert resolve_dataset_path(cfg) == tmp_path / DEFAULT_DATASET_FILENAME
