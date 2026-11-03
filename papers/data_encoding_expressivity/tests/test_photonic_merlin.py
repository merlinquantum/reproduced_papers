"""Tests for the Phase 4 MerLin adaptation (``lib/photonic_merlin.py``).

The last two tests pin down MerLin's ``UNBUNCHED`` semantics that motivated
using ``FOCK`` with an emulated threshold readout (LOG.md, Phase 4 plan,
probes 1-3).
"""

from __future__ import annotations

import json
import warnings

import merlin as ml
import numpy as np
import perceval as pcvl
import pytest
import torch
from common import PROJECT_DIR
from lib.fourier import fourier_coefficients, periodic_grid, spectrum_support
from lib.photonic_merlin import (
    PhotonicFourierModel,
    build_photonic_model,
    photonic_generator_eigenvalues,
    photonic_positive_frequencies,
    photonic_predicted_spectrum,
    single_photon_input_state,
)
from lib.training import build_fit_model, model_frequencies

warnings.filterwarnings("ignore", category=UserWarning)


def _cfg(
    n_photons,
    n_modes,
    encoded_modes=(0,),
    layers=1,
    space="fock",
    scales=None,
    screens=False,
    readout="threshold_mode0",
):
    return {
        "layout": "photonic",
        "n_photons": n_photons,
        "n_modes": n_modes,
        "encoded_modes": list(encoded_modes),
        "mode_scales": list(scales)
        if scales is not None
        else [1.0] * len(encoded_modes),
        "encoding_layers": layers,
        "encoding_scaling": 1.0,
        "phase_screens": screens,
        "readout": readout,
        "computation_space": space,
    }


def _support(model, n_points=48):
    grid = torch.tensor(periodic_grid(n_points), dtype=torch.float64)
    support = set()
    with torch.no_grad():
        for column in [*model.probabilities(grid).numpy().T, model(grid).numpy()]:
            frequencies, coefficients = fourier_coefficients(column)
            support.update(spectrum_support(frequencies, coefficients, 1e-9))
    return sorted(support)


def test_input_state_spreads_single_photons_and_needs_an_empty_mode():
    assert single_photon_input_state(3, 5) == [1, 1, 1, 0, 0]
    with pytest.raises(ValueError):
        single_photon_input_state(3, 3)


@pytest.mark.parametrize(
    "cfg,expected",
    [
        (_cfg(1, 2), [-1, 0, 1]),
        (_cfg(3, 4), list(range(-3, 4))),
        (_cfg(2, 3, layers=3), list(range(-6, 7))),
        (_cfg(2, 4, encoded_modes=(0, 1)), [-2, -1, 0, 1, 2]),
        (_cfg(2, 3, encoded_modes=(0, 1, 2)), [0]),
    ],
)
def test_predicted_spectrum_from_photon_number_generator(cfg, expected):
    assert [round(v) for v in photonic_predicted_spectrum(cfg)] == expected


def test_generator_eigenvalues_are_photon_counts_on_encoded_modes():
    assert photonic_generator_eigenvalues(3, 4, [0], [1.0], 1.0) == [0, 1, 2, 3]
    assert photonic_generator_eigenvalues(2, 3, [0, 1, 2], [1.0] * 3, 1.0) == [2]
    assert photonic_generator_eigenvalues(
        1, 6, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0
    ) == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]


@pytest.mark.parametrize(
    "cfg",
    [_cfg(2, 3), _cfg(3, 4), _cfg(2, 3, layers=2), _cfg(2, 4, encoded_modes=(0, 1))],
)
def test_fock_model_support_equals_prediction(cfg):
    model = build_photonic_model(cfg, torch.float64)
    model.reset_parameters(torch.Generator().manual_seed(3), 2 * np.pi)
    assert _support(model) == [round(v) for v in photonic_predicted_spectrum(cfg)]


def test_all_modes_encoded_is_a_global_phase():
    model = build_photonic_model(_cfg(2, 3, encoded_modes=(0, 1, 2)), torch.float64)
    model.reset_parameters(torch.Generator().manual_seed(0), 2 * np.pi)
    with torch.no_grad():
        values = model(torch.linspace(-6, 6, 20, dtype=torch.float64))
    assert torch.allclose(values, values[0].expand_as(values), atol=1e-12)


def test_shapes_readout_range_and_threshold_definition():
    model = build_photonic_model(_cfg(2, 3), torch.float64)
    model.reset_parameters(torch.Generator().manual_seed(1), 2 * np.pi)
    inputs = torch.linspace(-6, 6, 7, dtype=torch.float64)
    with torch.no_grad():
        probabilities = model.probabilities(inputs)
        values = model(inputs)
    assert probabilities.shape == (7, len(model.output_keys))
    assert values.shape == (7,)
    assert torch.allclose(probabilities.sum(1), torch.ones(7, dtype=torch.float64))
    empty = [k[0] == 0 for k in model.output_keys]
    expected = 2 * probabilities[:, empty].sum(1) - 1
    assert torch.allclose(values, expected)
    assert bool((values.abs() <= 1 + 1e-12).all())


def test_fock_space_is_used_and_contains_bunched_outcomes():
    fock = build_photonic_model(_cfg(2, 3), torch.float64)
    unbunched = build_photonic_model(_cfg(2, 3, space="unbunched"), torch.float64)
    assert len(fock.output_keys) == 6 and any(max(k) == 2 for k in fock.output_keys)
    assert len(unbunched.output_keys) == 3 and all(
        max(k) == 1 for k in unbunched.output_keys
    )


def test_gradients_reach_every_mesh_parameter():
    model = build_photonic_model(_cfg(2, 3, layers=2), torch.float64)
    model.reset_parameters(torch.Generator().manual_seed(2), 2 * np.pi)
    model(torch.linspace(-3, 3, 5, dtype=torch.float64)).pow(2).sum().backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(
        g is not None and torch.isfinite(g).all() for g in gradients
    )
    assert any(float(g.abs().max()) > 0 for g in gradients)


def test_seeding_is_deterministic():
    outputs = []
    for _ in range(2):
        model = build_photonic_model(_cfg(3, 4), torch.float64)
        model.reset_parameters(torch.Generator().manual_seed(11), 2 * np.pi)
        with torch.no_grad():
            outputs.append(model(torch.linspace(-6, 6, 9, dtype=torch.float64)))
    assert torch.equal(outputs[0], outputs[1])


def test_config_values_reach_the_model_and_b1_frequencies():
    cfg = json.loads((PROJECT_DIR / "configs" / "e2_fig4_merlin.json").read_text())
    case = next(c for c in cfg["fit"]["cases"] if c["name"] == "fig4_photonic_n2_L3")
    model = build_fit_model("photonic", case["model"], torch.float64)
    assert isinstance(model, PhotonicFourierModel)
    assert (model.n_photons, model.n_modes, model.encoding_layers) == (2, 3, 3)
    assert model.encoded_modes == [0] and model.computation_space == "fock"
    assert model_frequencies(case["model"]) == ([1, 2, 3, 4, 5, 6], 1.0)
    assert photonic_positive_frequencies(case["model"]) == [1, 2, 3, 4, 5, 6]


def _probe_layer(space):
    circuit = pcvl.Circuit(3)
    circuit.add(0, pcvl.BS())
    circuit.add(0, pcvl.PS(pcvl.P("px")))
    circuit.add(1, pcvl.BS())
    circuit.add(0, pcvl.BS())
    return ml.QuantumLayer(
        input_size=1,
        circuit=circuit,
        input_state=[1, 1, 0],
        n_photons=2,
        trainable_parameters=[],
        input_parameters=["px"],
        computation_space=space,
        dtype=torch.float64,
    )


def test_merlin_unbunched_is_postselected_and_renormalised_fock():
    inputs = torch.tensor([[0.0], [0.7], [1.9]], dtype=torch.float64)
    fock_layer = _probe_layer(ml.ComputationSpace.FOCK)
    unbunched_layer = _probe_layer(ml.ComputationSpace.UNBUNCHED)
    with torch.no_grad():
        fock = fock_layer(inputs)
        unbunched = unbunched_layer(inputs)
    keys = [tuple(int(n) for n in key) for key in fock_layer.output_keys]
    collision_free = torch.tensor([max(key) <= 1 for key in keys])
    postselected = fock[:, collision_free]
    renormalised = postselected / postselected.sum(1, keepdim=True)
    assert torch.allclose(unbunched, renormalised, atol=1e-12)
    # The discarded bunched probability depends on x, so renormalisation is not a constant factor.
    discarded = 1 - postselected.sum(1)
    assert float(discarded.max() - discarded.min()) > 0.1


def test_merlin_unbunched_output_is_not_band_limited():
    n_points = 256
    grid = torch.tensor(periodic_grid(n_points), dtype=torch.float64).unsqueeze(1)
    with torch.no_grad():
        fock = _probe_layer(ml.ComputationSpace.FOCK)(grid).numpy()
        unbunched = _probe_layer(ml.ComputationSpace.UNBUNCHED)(grid).numpy()
    allowed = {-2, -1, 0, 1, 2}
    for column in fock.T:
        frequencies, coefficients = fourier_coefficients(column)
        assert set(spectrum_support(frequencies, coefficients, 1e-9)) <= allowed
    frequencies, coefficients = fourier_coefficients(unbunched[:, 0])
    beyond = np.abs(coefficients[np.abs(frequencies) == 4])
    assert float(beyond.max()) > 1e-3


def _mesh_rank(n_modes, phase_screens):
    """Jacobian rank of the trainable mode unitary built by the model's mesh helper."""
    import merlin as ml_local

    builder = ml_local.CircuitBuilder(n_modes=n_modes)
    model = PhotonicFourierModel.__new__(PhotonicFourierModel)
    model.phase_screens, model.n_modes = phase_screens, n_modes
    PhotonicFourierModel._add_screened_mesh(model, builder, "t")
    circuit = builder.to_pcvl_circuit()
    params = circuit.get_parameters()
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 2 * np.pi, len(params))

    def unitary(v):
        for parameter, value in zip(params, v):
            parameter.set_value(float(value))
        return np.array(circuit.compute_unitary(), dtype=complex)

    base = unitary(values)
    columns = []
    for index in range(len(params)):
        shifted = values.copy()
        shifted[index] += 1e-6
        delta = (unitary(shifted) - base) / 1e-6
        columns.append(np.concatenate([delta.real.ravel(), delta.imag.ravel()]))
    return np.linalg.matrix_rank(np.array(columns).T, tol=1e-4)


@pytest.mark.parametrize("n_modes", [3, 4, 6])
def test_phase_screens_make_the_mesh_universal(n_modes):
    # The bare MerLin 0.2.3 mesh misses the m - 1 input-side relative phases;
    # a screen placed before it supplies exactly those.
    assert _mesh_rank(n_modes, False) == n_modes * (n_modes - 1)
    assert _mesh_rank(n_modes, True) == n_modes * n_modes - 1


def test_multiscale_single_photon_spectrum_is_theorem_setting():
    cfg = _cfg(
        1, 6, encoded_modes=(1, 2, 3, 4, 5), scales=(1, 2, 3, 4, 5), screens=True
    )
    assert [round(v) for v in photonic_predicted_spectrum(cfg)] == list(range(-5, 6))
    model = build_photonic_model(cfg, torch.float64)
    model.reset_parameters(torch.Generator().manual_seed(1), 2 * np.pi)
    assert _support(model, n_points=64) == list(range(-5, 6))


def test_multiscale_two_photons_reach_degree_six():
    cfg = _cfg(2, 4, encoded_modes=(1, 2), scales=(1, 3), screens=True)
    assert max(photonic_positive_frequencies(cfg)) == 6
    model = build_photonic_model(cfg, torch.float64)
    model.reset_parameters(torch.Generator().manual_seed(2), 2 * np.pi)
    assert _support(model, n_points=64) == list(range(-6, 7))


def test_trainable_readout_starts_at_zero_and_receives_gradients():
    model = build_photonic_model(
        _cfg(2, 3, screens=True, readout="trainable_linear"), torch.float64
    )
    model.reset_parameters(torch.Generator().manual_seed(0), 2 * np.pi)
    inputs = torch.linspace(-3, 3, 7, dtype=torch.float64)
    output = model(inputs)
    assert torch.allclose(output, torch.zeros_like(output))
    (output - 0.5).pow(2).sum().backward()
    assert float(model.readout_layer.weight.grad.abs().max()) > 0
    assert len(model.click_patterns) == model.readout_layer.in_features


def test_previous_photonic_behaviour_is_unchanged_with_default_keys():
    # Pre-scaled inputs at encoding scale 1 equal the Phase 4 encoding (s = 1, k = 1).
    cfg = _cfg(2, 3, layers=2)
    model = build_photonic_model(cfg, torch.float64)
    model.reset_parameters(torch.Generator().manual_seed(5), 2 * np.pi)
    assert model.readout_layer is None and not model.phase_screens
    assert _support(model) == list(range(-4, 5))


def test_invalid_photonic_options_raise():
    with pytest.raises(ValueError):
        build_photonic_model(_cfg(2, 3, readout="nope"), torch.float64)
    with pytest.raises(ValueError):
        build_photonic_model(
            _cfg(2, 4, encoded_modes=(1, 2), scales=(1,)), torch.float64
        )
    cfg = _cfg(2, 3)
    del cfg["phase_screens"]
    with pytest.raises(KeyError):
        build_photonic_model(cfg, torch.float64)
