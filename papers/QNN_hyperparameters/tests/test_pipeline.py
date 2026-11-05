from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from common import PROJECT_DIR, REPO_ROOT
from QNN_hyperparameters.lib.circuits import paper_grid
from QNN_hyperparameters.lib.data import build_dataloader, prepare_data
from QNN_hyperparameters.lib.sweep import expand_candidates

DATA_ROOT = REPO_ROOT / "data"
HAVE_RICE = (DATA_ROOT / "QNN_hyperparameters" / "rice" / "rice_features.csv").exists()
RESULTS_DIR = (
    DATA_ROOT
    / "QNN_hyperparameters"
    / "sources"
    / "author_repo"
    / "reports"
    / "results"
)


def _config(name: str) -> dict:
    from runtime_lib.config import deep_update

    cfg = json.loads((PROJECT_DIR / "configs" / "defaults.json").read_text())
    if name != "defaults":
        cfg = deep_update(
            cfg, json.loads((PROJECT_DIR / "configs" / f"{name}.json").read_text())
        )
    cfg["data_root"] = str(DATA_ROOT)
    return cfg


def test_paper_grid_has_84_configurations():
    grid = paper_grid()
    assert len(grid) == 84
    assert grid[0] == {
        "feature_map": "ZFeatureMap",
        "feature_map_entanglement": None,
        "ansatz": "EfficientSU2",
        "ansatz_entanglement": "full",
    }


@pytest.mark.parametrize(
    "name, expected",
    [
        ("mechanism_factorial", 288),
        ("rice_grid_original", 504),
        ("rice_grid_author_signs", 252),
        ("rice_grid_noisy_reduced", 48),
        ("rice_grid_noisy_author_signs", 48),
        ("photonic_mechanism", 144),
        ("baselines_classical", 420),
        ("mechanism_smoke", 12),
    ],
)
def test_sweep_sizes_match_plan(name, expected):
    candidates = expand_candidates(_config(name))
    assert len(candidates) == expected
    assert len({c["run_id"] for c in candidates}) == expected


@pytest.mark.skipif(not HAVE_RICE, reason="Rice data not acquired")
def test_rice_loader_shapes_and_alignment():
    cfg = _config("defaults")
    X, y = next(iter(build_dataloader(cfg, "train")))
    assert X.shape == (400, 7) and y.shape == (400,)
    assert X.min() >= 0 and X.max() <= 1  # minmax default
    raw = prepare_data({**cfg, "dataset": {**cfg["dataset"], "input_scaling": "raw"}})
    assert raw.X["val"].shape == (250, 7) and raw.X["test"].shape == (250, 7)
    assert raw.X["train"].min() < -1  # PCA scores are not in [0, 1]
    assert set(raw.y["train"]) == {0, 1}


@pytest.mark.skipif(not HAVE_RICE, reason="Rice data not acquired")
def test_pca_signs_u_based_flips_only_signs():
    cfg = _config("defaults")
    raw = {**cfg["dataset"], "input_scaling": "raw"}
    installed = prepare_data({**cfg, "dataset": raw}).X["train"]
    author = prepare_data({**cfg, "dataset": {**raw, "pca_signs": "u_based"}})
    signs = np.sign((installed * author.X["train"]).sum(axis=0))
    assert np.allclose(np.abs(installed), np.abs(author.X["train"]))
    # Under the author's scikit-learn 1.2.2 convention, components 2 and 5
    # of Rice have the opposite sign (LOG.md, E-C gap diagnosis).
    assert signs.tolist() == [1, -1, 1, 1, -1, 1, 1]


@pytest.mark.skipif(not RESULTS_DIR.exists(), reason="author results not cached")
def test_released_results_parse():
    from QNN_hyperparameters.lib.reanalysis import load_released_results

    results = load_released_results(Path(RESULTS_DIR))
    # 197 files; the 19 Nelder-Mead files of the post-hoc arms (beta_mu,
    # normal_beta_dist, zero) are header-only because the author run skips them.
    assert len(list(Path(RESULTS_DIR).glob("*_results.csv"))) == 197
    assert results["source_file"].nunique() == 178
    assert set(results["ansatz"]) == {
        "EfficientSU2",
        "PauliTwoDesign",
        "RealAmplitudes",
        "TwoLocal",
    }
    rice = results[
        (results["dataset"] == "rice")
        & (~results["noisy"])
        & (results["init"] == "beta")
    ]
    assert rice["index"].max() == 83
