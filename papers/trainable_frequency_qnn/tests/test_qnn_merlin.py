from __future__ import annotations

import math

import common  # noqa: F401  (puts the paper package on sys.path)
import numpy as np
import pytest
import torch
from common import load_runtime_ready_config
from trainable_frequency_qnn.lib.models import build_network, parameter_summary
from trainable_frequency_qnn.lib.qnn_merlin import (
    PhotonicFrequencyModel,
    spread_input_state,
)
from trainable_frequency_qnn.lib.spectrum import (
    dominant_frequencies,
    prediction_spectrum,
)


def _toy(
    mode: str = "fixed",
    readout: str = "threshold_clicks",
    seed: int = 0,
    init: float = 1.0,
):
    return PhotonicFrequencyModel(
        n_modes=6,
        n_photons=3,
        n_inputs=1,
        layout="toy",
        encoding_modes_per_input=3,
        feature_map="simple",
        frequency_mode=mode,
        frequency_init=init,
        readout=readout,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(seed),
    )


def test_spread_input_state_and_photon_limit():
    assert spread_input_state(6, 3) == [1, 0, 1, 0, 1, 0]
    assert spread_input_state(8, 4) == [1, 0, 1, 0, 1, 0, 1, 0]
    with pytest.raises(ValueError):
        spread_input_state(6, 4)


def test_readout_selects_computation_space():
    # FOCK keeps bunched outcomes (C(8, 3) = 56); UNBUNCHED keeps C(6, 3) = 20.
    assert len(_toy(readout="threshold_clicks").layer.output_keys) == 56
    assert len(_toy(readout="unbunched_clicks").layer.output_keys) == 20


def test_click_observable_counts_threshold_clicks():
    model = _toy()
    observable = dict(zip(model.layer.output_keys, model.click_observable.tolist()))
    assert observable[(1, 0, 1, 0, 1, 0)] == 3.0
    assert observable[(0, 1, 0, 1, 0, 1)] == -3.0
    # Threshold detectors: three photons in mode 0 are a single click.
    assert observable[(3, 0, 0, 0, 0, 0)] == 1.0
    assert observable[(2, 1, 0, 0, 0, 0)] == 0.0


def test_shapes_parameters_and_gradients():
    model = _toy("trainable")
    inputs = torch.linspace(-3, 3, 5, dtype=torch.float64)[:, None]
    outputs = model(inputs)
    assert outputs.shape == (5,)
    summary = parameter_summary(model)
    assert summary["frequency"] == 3 and summary["ansatz"] > 0
    outputs.sum().backward()
    assert torch.any(model.frequency_weights.grad != 0)
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.layer.parameters()
    )


def test_paired_seeds_and_determinism():
    inputs = torch.linspace(-3, 3, 7, dtype=torch.float64)[:, None]
    fixed, trainable, scaled = (
        _toy(mode)(inputs) for mode in ("fixed", "trainable", "global_scale")
    )
    assert torch.allclose(fixed, trainable) and torch.allclose(fixed, scaled)
    assert torch.allclose(_toy(seed=4)(inputs), _toy(seed=4)(inputs))
    assert not torch.allclose(_toy(seed=4)(inputs), _toy(seed=5)(inputs))


def test_ff_threshold_spectrum_is_integer_up_to_photon_number():
    # Known answer: 3 photons through 3 unit-weight phase shifters give gaps {1, 2, 3}.
    model = _toy(seed=3)
    assert model.accessible_frequencies() == [1.0, 2.0, 3.0]
    angular, magnitudes = prediction_spectrum(model, 64, 8192, torch.float64, "cpu")
    resolution = angular[1] - angular[0]
    assert magnitudes[angular > 3 + 2 * resolution].max() < 1e-4
    for peak in dominant_frequencies(angular, magnitudes, 3):
        assert abs(peak - round(peak)) < 2 * resolution


def test_unbunched_readout_leaks_above_circuit_spectrum():
    # Same circuit, two detection models: post-selected UNBUNCHED probabilities are
    # renormalised, so the output is a ratio of trigonometric polynomials with power
    # above the circuit spectrum (max frequency 3); threshold clicks are not.
    leakage = {}
    for readout in ("threshold_clicks", "unbunched_clicks"):
        angular, magnitudes = prediction_spectrum(
            _toy(readout=readout, seed=3), 64, 8192, torch.float64, "cpu"
        )
        leakage[readout] = magnitudes[angular > 3.5].max()
    assert leakage["unbunched_clicks"] > 10 * leakage["threshold_clicks"]


def test_frequency_init_and_trainable_spectrum():
    model = _toy("trainable", init=1.5)
    assert torch.allclose(
        model.encoding_frequencies(), torch.full((3,), 1.5, dtype=torch.float64)
    )
    with torch.no_grad():
        model.frequency_weights.copy_(
            torch.tensor([0.89, 1.05, 1.04], dtype=torch.float64)
        )
    gaps = model.accessible_frequencies()
    assert min(abs(g - 1.2) for g in gaps) < 1e-9
    assert any(abs(g - round(g)) > 1e-3 for g in gaps)


def test_navier_stokes_layout_and_input_derivatives():
    model = PhotonicFrequencyModel(
        n_modes=6,
        n_photons=3,
        n_inputs=3,
        layout="ns_parallel",
        encoding_modes_per_input=2,
        feature_map="simple",
        frequency_mode="trainable",
        frequency_init=1.0,
        readout="threshold_clicks",
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(0),
    )
    assert model.n_frequency_parameters() == 6
    inputs = torch.rand((4, 3), dtype=torch.float64, requires_grad=True)
    output = model(inputs)
    first = torch.autograd.grad(output.sum(), inputs, create_graph=True)[0]
    third = torch.autograd.grad(
        torch.autograd.grad(first[:, 0].sum(), inputs, create_graph=True)[0][
            :, 0
        ].sum(),
        inputs,
    )[0]
    assert torch.isfinite(third).all()
    # Finite-difference check of d output / d x.
    step = 1e-5
    shift = torch.zeros(1, 3, dtype=torch.float64)
    shift[0, 0] = step
    with torch.no_grad():
        numeric = (model(inputs + shift) - model(inputs - shift)) / (2 * step)
    assert torch.allclose(first[:, 0], numeric, atol=1e-7)


@pytest.mark.parametrize(
    "config, n_modes, n_photons, mode",
    [
        ("fig2b_tf_merlin.json", 6, 3, "trainable"),
        ("fig3_tower_merlin.json", 8, 4, "fixed"),
        ("ns_table1_ffs_merlin.json", 6, 3, "global_scale"),
    ],
)
def test_configs_reach_the_model(config, n_modes, n_photons, mode):
    cfg = load_runtime_ready_config(config)
    n_inputs = 3 if cfg["task"] == "navier_stokes" else 1
    model = build_network(
        cfg["model"], n_inputs, torch.float64, torch.Generator().manual_seed(0)
    )
    assert isinstance(model, PhotonicFrequencyModel)
    assert (model.n_modes, model.n_photons, model.frequency_mode) == (
        n_modes,
        n_photons,
        mode,
    )
    assert model.n_photons >= 2
    assert math.isclose(float(model.encoding_frequencies()[0]), 1.0)
    assert np.isfinite(
        model(torch.zeros((2, n_inputs), dtype=torch.float64)).detach().numpy()
    ).all()
