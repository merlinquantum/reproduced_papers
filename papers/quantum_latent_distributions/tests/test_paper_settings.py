"""The paper-accurate configs must keep the values the paper actually states.

These are the numbers that were wrong in the first pass of this reproduction, so
they get pinned: Appendix D specifies RMSProp at 5e-4, batches of 500 over 40k
iterations, and ReLU hidden activations; latent re-injection is an Appendix E
(QM9) detail and must not leak into the toy studies.
"""

from __future__ import annotations

import json

import pytest
from common import PROJECT_DIR

CONFIGS = PROJECT_DIR / "configs"


def _load(name):
    return json.loads((CONFIGS / name).read_text())


def test_synthetic_config_matches_appendix_d():
    cfg = _load("synthetic_datasets.json")
    assert cfg["training"]["optimizer"] == "rmsprop"
    assert cfg["training"]["lr"] == 5e-4
    assert cfg["training"]["n_critic"] == 5
    assert cfg["training"]["iterations"] == 40_000
    assert cfg["dataset"]["batch_size"] == 500
    assert cfg["model"]["activation"] == "relu"
    assert cfg["model"]["generator_hidden"] == [512, 512]
    assert cfg["model"]["critic_hidden"] == [512, 512]
    assert cfg["evaluation"]["repeats"] == 12  # "estimated over 12 runs"


def test_qm9_config_matches_appendix_e():
    cfg = _load("qm9.json")
    assert cfg["training"]["optimizer"] == "adam"
    assert cfg["training"]["lr"] == 1e-4
    assert cfg["training"]["iterations"] == 20_000
    assert cfg["dataset"]["batch_size"] == 256
    assert cfg["model"]["generator_hidden"] == [64, 176, 288, 400, 512]
    assert cfg["model"]["latent_reinjection"] is True
    assert cfg["model"]["activation"] == "leaky_relu"
    assert cfg["evaluation"]["repeats"] == 5  # paper uses 20; see README


def test_mixture_config_matches_appendix_c_and_the_released_code():
    cfg = _load("mixture_of_gaussians.json")
    assert cfg["training"]["iterations"] == 5000
    assert cfg["evaluation"]["repeats"] == 5
    assert cfg["latent"]["dim"] == 16
    assert cfg["latent"]["architecture"] == "1-3-9"
    assert cfg["latent"]["normalize"] == "center"
    assert cfg["model"]["generator_hidden"] == [256, 256]
    assert cfg["model"]["activation"] == "leaky_relu"
    # from src/gaussians_utils.py in the authors' release
    assert cfg["mixture"]["n_components"] == 7
    assert cfg["mixture"]["radius"] == 5.0
    assert cfg["mixture"]["radial_std"] == 0.2
    assert cfg["mixture"]["tangential_std"] == 0.05
    assert cfg["training"]["lr"] == 5e-4
    assert cfg["training"]["betas"] == [0.0, 0.9]
    assert cfg["dataset"]["batch_size"] == 32


@pytest.mark.parametrize(
    "name", ["synthetic_datasets.json", "mixture_of_gaussians.json"]
)
def test_latent_reinjection_is_a_qm9_only_detail(name):
    assert _load(name)["model"]["latent_reinjection"] is False
