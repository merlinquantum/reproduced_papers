from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from common import PROJECT_DIR
from lib.kernels import fidelity_kernel
from lib.photonic_merlin import (
    PhotonicQNNClassifier,
    photonic_feature_states,
    spread_input_state,
)
from lib.tasks.directional import build_model, qnn_candidates


def test_spread_input_state_places_single_photons():
    assert spread_input_state(7, 3) == [1, 0, 1, 0, 0, 1, 0]
    assert sum(spread_input_state(10, 2)) == 2
    with pytest.raises(ValueError):
        spread_input_state(4, 1)


def test_photonic_kernel_is_a_valid_nontrivial_gram_matrix():
    features = np.random.default_rng(0).uniform(0, 1, size=(12, 10))
    states = photonic_feature_states(
        features, n_photons=2, layers=2, angle_scale=math.pi, mesh_seed=1234
    )
    assert states.shape == (12, math.comb(10 + 2 - 1, 2))
    gram = fidelity_kernel(states, states)
    assert np.allclose(np.diag(gram), 1.0)
    assert np.allclose(gram, gram.T)
    assert np.linalg.eigvalsh(gram).min() > -1e-10
    assert gram[~np.eye(12, dtype=bool)].max() < 0.999


def test_photonic_kernel_uses_the_explicit_angle_scale_and_seed():
    features = np.random.default_rng(1).uniform(0, 1, size=(6, 10))
    base = fidelity_kernel(
        *(photonic_feature_states(features, 2, 1, math.pi, 1234),) * 2
    )
    scaled = fidelity_kernel(*(photonic_feature_states(features, 2, 1, 1.0, 1234),) * 2)
    reseeded = fidelity_kernel(
        *(photonic_feature_states(features, 2, 1, math.pi, 99),) * 2
    )
    again = fidelity_kernel(
        *(photonic_feature_states(features, 2, 1, math.pi, 1234),) * 2
    )
    assert not np.allclose(base, scaled)
    assert not np.allclose(base, reseeded)
    assert np.allclose(base, again)


@pytest.mark.parametrize(
    "hybrid, readout", [(False, False), (False, True), (True, False), (True, True)]
)
def test_photonic_qnn_shapes_space_and_gradients(hybrid, readout):
    torch.manual_seed(0)
    model = PhotonicQNNClassifier(
        7, 7, 3, 2, hybrid=hybrid, multi_mode_readout=readout, angle_scale=math.pi
    )
    assert model.circuit.output_size == math.comb(
        7, 3
    )  # UNBUNCHED: threshold-detection outcomes
    assert int(model.last_mode_mask.sum()) == math.comb(6, 2)
    logits = model(torch.rand(5, 7))
    assert logits.shape == (5,) and torch.isfinite(logits).all()
    logits.sum().backward()
    circuit_grads = [p.grad for p in model.circuit.parameters() if p.requires_grad]
    assert circuit_grads and any(
        g is not None and g.abs().sum() > 0 for g in circuit_grads
    )


def test_photonic_qnn_is_deterministic_under_seed():
    inputs = torch.rand(4, 7)
    outputs = []
    for _ in range(2):
        torch.manual_seed(3)
        outputs.append(PhotonicQNNClassifier(7, 7, 2, 1, True, True, math.pi)(inputs))
    assert torch.allclose(outputs[0], outputs[1])


def test_merlin_config_reaches_the_model():
    config = json.loads((PROJECT_DIR / "configs" / "e2_merlin.json").read_text())[
        "directional"
    ]["models"]
    defaults = json.loads((PROJECT_DIR / "configs" / "defaults.json").read_text())[
        "directional"
    ]["models"]
    models_cfg = {**defaults, "qnn": {**defaults["qnn"], **config["qnn"]}}
    candidates = qnn_candidates(models_cfg["qnn"], n_features=7)
    assert len(candidates) == 16 and all(
        c["name"].startswith("pqnn_") for c in candidates
    )
    model = build_model(candidates[-1], 7, models_cfg)
    assert isinstance(model, PhotonicQNNClassifier)
    assert model.layers == candidates[-1]["layers"] and model.preprocess is not None
