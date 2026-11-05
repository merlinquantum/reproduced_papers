from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
import torch
from common import PROJECT_DIR  # noqa: F401  (puts the repo on sys.path)
from QNN_hyperparameters.lib.photonic import (
    PhotonicClassifier,
    build_photonic_layer,
    spread_input_state,
)
from QNN_hyperparameters.lib.sweep_analysis import photonic_effects


def _classifier(n_features=3, photons=2, depth=1, init="merlin_default", seed=0):
    torch.manual_seed(seed)
    return PhotonicClassifier(
        {"init": init},
        {"photons": photons, "depth": depth},
        n_features,
        "cpu",
        np.random.default_rng(seed),
    )


def _separable(n_features=3, rows=20, seed=3):
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            rng.normal(0.2, 0.05, (rows, n_features)),
            rng.normal(0.9, 0.05, (rows, n_features)),
        ]
    )
    return X, np.array([0] * rows + [1] * rows)


def test_spread_input_state_places_photons_over_the_modes():
    assert spread_input_state(14, 2) == [1] + [0] * 6 + [1] + [0] * 6
    assert [i for i, v in enumerate(spread_input_state(14, 4)) if v] == [0, 3, 7, 10]
    assert [i for i, v in enumerate(spread_input_state(10, 4)) if v] == [0, 2, 5, 7]
    with pytest.raises(ValueError):
        spread_input_state(4, 1)


def test_layer_has_twice_the_modes_one_phase_tensor_and_mode_readout():
    layer, state = build_photonic_layer(7, 4, 2, torch.device("cpu"))
    assert len(state) == 14 and sum(state) == 4
    # Three trainable MZI meshes on 14 modes, 14 * 13 phases each.
    assert {name: p.numel() for name, p in layer.named_parameters()} == {
        "mesh": 3 * 14 * 13
    }
    out = layer(torch.rand(5, 7 * 2))
    # Mode expectations under threshold detection: one value per mode,
    # summing to at most the photon number.
    assert out.shape == (5, 14)
    assert torch.all(out.sum(dim=1) <= 4 + 1e-5)


def test_reuploaded_inputs_all_reach_the_circuit():
    layer, _ = build_photonic_layer(3, 2, 2, torch.device("cpu"))
    x = torch.full((1, 6), 0.3)
    changed = x.clone()
    changed[0, 4] += 0.5  # second upload of feature 1 only
    with torch.no_grad():
        assert not torch.allclose(layer(x), layer(changed))


def test_initial_phases_are_applied_and_deterministic():
    X, y = _separable()
    model = _classifier(init="zeros")
    model.fit(X, y, {"optimizer": "Adam", "steps": 1, "lr": 0.0})
    assert torch.all(model.phases == 0)
    beta = _classifier(init="beta")
    result = beta.fit(X, y, {"optimizer": "Adam", "steps": 1, "lr": 0.0})
    assert "beta_a" in result.initial_point_details
    assert torch.all((beta.phases >= 0) & (beta.phases <= 1))
    # As in the runner: seed, build, fit. The readout head is drawn from the
    # seeded torch generator inside ``fit``.
    trained = []
    for _ in range(2):
        model = _classifier(seed=7)
        model.fit(X, y, {"optimizer": "Adam", "steps": 2, "lr": 0.05})
        trained.append(model)
    assert torch.equal(trained[0].phases, trained[1].phases)
    assert torch.equal(trained[0].head.weight, trained[1].head.weight)


def test_adam_learns_with_gradients_on_phases_and_head(caplog):
    X, y = _separable()
    model = _classifier(photons=2, depth=2)
    with caplog.at_level("INFO"):
        result = model.fit(X, y, {"optimizer": "Adam", "steps": 40, "lr": 0.1})
    assert result.evaluations == 40
    assert result.losses[-1] < result.losses[0]
    assert model.phases.grad is not None and torch.any(model.phases.grad != 0)
    assert model.head.weight.grad is not None
    assert caplog.text.count("TRAIN_STEP_COMPLETED") == 40
    assert model.evaluate(X, y)["accuracy"] > 0.9
    assert model.num_weights == model.num_phases + 6 * 2 + 2


def test_invalid_settings_fail_clearly():
    with pytest.raises(ValueError, match="photonic init"):
        _classifier(init="uniform")
    X, y = _separable()
    with pytest.raises(ValueError, match="Adam"):
        _classifier().fit(X, y, {"optimizer": "COBYLA"})
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            PhotonicClassifier(
                {"init": "zeros"},
                {"photons": 2, "depth": 1},
                3,
                "cuda:0",
                np.random.default_rng(0),
            )


def test_photonic_analysis_recovers_a_planted_scaling_effect():
    rng = np.random.default_rng(0)
    rows = []
    for data, scaling, init, photons, depth, seed in itertools.product(
        ["rice", "glass"],
        ["raw", "minmax"],
        ["default", "beta", "zeros"],
        ["n2", "n4"],
        ["d1", "d2"],
        [1, 2, 3],
    ):
        rows.append(
            {
                "data": data,
                "scaling": scaling,
                "init": init,
                "photons": photons,
                "depth": depth,
                "seed": seed,
                "val_accuracy": 0.5 + 0.3 * (scaling == "minmax") + rng.normal(0, 0.01),
            }
        )
    effects = photonic_effects(pd.DataFrame(rows))
    # Per dataset: 3 scaling + 4 init + 2 photon + 2 depth comparisons.
    assert len(effects) == 22
    scaling = effects[effects["effect"] == "scaling"]
    assert (scaling["n_pairs"] == 12).all()
    assert np.allclose(scaling["mean_diff"], 0.3, atol=0.01)
    assert (scaling["holm_p"] < 0.05).all()
    assert effects.loc[effects["effect"] == "init", "negligible"].all()
    assert (effects.loc[effects["effect"] == "photons", "n_pairs"] == 18).all()
