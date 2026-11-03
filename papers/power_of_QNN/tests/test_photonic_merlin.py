from __future__ import annotations

import json
import math

import pytest
import torch
from common import PROJECT_DIR

pytest.importorskip("merlin")
import perceval as pcvl  # noqa: E402
from power_of_QNN.lib.fisher import empirical_fisher  # noqa: E402
from power_of_QNN.lib.models import build_model  # noqa: E402
from power_of_QNN.lib.photonic_merlin import PhotonicEncodingModel, mzi_positions  # noqa: E402

CPU = torch.device("cpu")


def _model(
    encoding: str, d: int = 12, detection: str = "threshold", n_inputs: int = 4
) -> PhotonicEncodingModel:
    return PhotonicEncodingModel(n_inputs, encoding, d, detection)


def _batch(
    model: PhotonicEncodingModel, rows: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, model.input_size, dtype=torch.float64, generator=generator)
    theta = (
        2
        * math.pi
        * torch.rand(rows, model.d, dtype=torch.float64, generator=generator)
    )
    return x, theta


def test_mzi_positions_wrap_over_brick_columns():
    assert mzi_positions(6, 7) == [0, 2, 4, 1, 3, 0, 2]
    assert len(mzi_positions(12, 50)) == 50


@pytest.mark.parametrize("encoding", ["easy", "hard"])
def test_shapes_normalization_and_chip_layout(encoding):
    model = _model(encoding, d=40)
    assert (model.n_modes, model.n_photons, model.input_state) == (
        6,
        3,
        [1, 0, 1, 0, 1, 0],
    )
    x, theta = _batch(model, 5)
    probabilities = model.probabilities(x, theta)
    assert probabilities.shape == (5, 2)
    assert probabilities.dtype == torch.float64
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(5, dtype=torch.float64))
    trainable = sum(
        parameter.numel() for parameter in model.layer(CPU, torch.float64).parameters()
    )
    assert trainable == 0, (
        "trainable phases must enter per sample, not as layer parameters"
    )


@pytest.mark.parametrize("encoding", ["easy", "hard"])
def test_matches_perceval_slos_with_threshold_click_parity(encoding):
    # Independent reference: set every named Perceval parameter, simulate with SLOS,
    # and apply the click-parity readout by hand.
    model = _model(encoding, d=16)
    x, theta = _batch(model, 1, seed=3)
    ours = model.probabilities(x, theta)[0].detach().numpy()
    values = {}
    for repetition in range(model.repetitions):
        for feature in range(model.input_size):
            values[f"a{repetition}{feature:03d}"] = 2 * float(x[0, feature])
        if encoding == "hard":
            for feature in range(model.input_size - 1):
                values[f"b{repetition}{feature:03d}"] = (
                    2
                    * (math.pi - float(x[0, feature]))
                    * (math.pi - float(x[0, feature + 1]))
                )
    for index in range(model.d):
        values[f"t{index:04d}"] = float(theta[0, index])
    circuit = model.circuit.copy()
    parameters = {parameter.name: parameter for parameter in circuit.get_parameters()}
    assert set(parameters) == set(values)
    for name, value in values.items():
        parameters[name].set_value(value)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    backend.set_circuit(circuit)
    backend.set_input_state(pcvl.BasicState(model.input_state))
    odd = 0.0
    for state, probability in backend.prob_distribution().items():
        clicks = sum(1 for mode in range(1, model.n_modes, 2) if state[mode] > 0)
        odd += probability * (clicks % 2)
    assert ours[1] == pytest.approx(odd, abs=1e-10)


def test_unbunched_is_renormalized_collision_free_fock():
    threshold = _model("hard", d=16, detection="threshold")
    postselected = _model("hard", d=16, detection="unbunched")
    x, theta = _batch(threshold, 3, seed=5)
    layer = postselected.layer(CPU, torch.float64)
    assert all(max(state) <= 1 for state in layer.output_keys)
    assert any(
        max(state) > 1 for state in threshold.layer(CPU, torch.float64).output_keys
    )
    fock = threshold.layer(CPU, torch.float64)(*threshold._inputs(x, theta))
    free = torch.tensor(
        [max(state) <= 1 for state in threshold.layer(CPU, torch.float64).output_keys]
    )
    mask = torch.tensor(
        [
            sum(1 for m in range(1, 6, 2) if s[m] > 0) % 2 == 1
            for s in threshold.layer(CPU, torch.float64).output_keys
        ]
    )
    expected_odd = fock[:, free & mask].sum(1) / fock[:, free].sum(1)
    assert torch.allclose(postselected.probabilities(x, theta)[:, 1], expected_odd)
    assert torch.allclose(
        threshold.collision_free_probability(x, theta), fock[:, free].sum(1)
    )


@pytest.mark.parametrize("encoding", ["easy", "hard"])
def test_gradients_flow_and_rows_are_independent(encoding):
    model = _model(encoding)
    x, theta = _batch(model, 3, seed=7)
    theta.requires_grad_(True)
    probabilities = model.probabilities(x, theta)
    (gradient,) = torch.autograd.grad(probabilities[:, 1].sum(), theta)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    single = model.probabilities(x[1:2], theta[1:2].detach())
    assert torch.allclose(single[0], probabilities[1].detach())
    assert torch.equal(model.probabilities(x, theta.detach()), probabilities.detach())


def test_data_enters_the_model():
    model = _model("easy")
    x, theta = _batch(model, 1, seed=9)
    assert not torch.allclose(
        model.probabilities(x, theta), model.probabilities(x + 0.3, theta)
    )


def test_fisher_pipeline_accepts_photonic_model():
    model = _model("hard")
    x, theta = _batch(model, 4, seed=11)
    fisher = empirical_fisher(model, theta[:2], x, chunk_size=4)
    assert fisher.shape == (2, model.d, model.d)
    assert torch.allclose(fisher, fisher.transpose(1, 2))
    assert torch.linalg.eigvalsh(fisher).min() > -1e-10


def test_configs_reach_the_model():
    cfg = json.loads(
        (PROJECT_DIR / "configs" / "spectrum_scaling_merlin.json").read_text()
    )
    for entry in cfg["models"]:
        model = build_model(entry)
        assert (model.input_size, model.d, model.detection) == (
            entry["n_inputs"],
            entry["d"],
            entry["detection"],
        )
        assert model.n_modes == entry["n_inputs"] + entry["extra_modes"]
        assert model.describe()["computation_space"] == "FOCK"
    unbunched = json.loads(
        (PROJECT_DIR / "configs" / "effdim_d40_merlin_unbunched.json").read_text()
    )["models"][0]
    assert (
        build_model(unbunched).describe()["postselection"]
        == "collision-free outcomes only"
    )


def test_rejects_odd_d():
    with pytest.raises(ValueError):
        PhotonicEncodingModel(4, "easy", 7, "threshold")


def _live_parameters(model: PhotonicEncodingModel, seed: int = 13) -> torch.Tensor:
    x, theta = _batch(model, 8, seed=seed)
    theta.requires_grad_(True)
    probabilities = model.probabilities(x, theta)
    gradients = torch.stack(
        [
            torch.autograd.grad(probabilities[:, k].sum(), theta, retain_graph=True)[0]
            for k in range(2)
        ]
    )
    return gradients.abs().amax(dim=(0, 1)) > 1e-12


def test_bs_ps_mesh_has_dead_final_phases_and_ps_bs_does_not():
    # Amendment 2 defect: a phase after the last beam splitter on its mode cannot change
    # detection probabilities.
    planned = PhotonicEncodingModel(4, "hard", 40, "threshold")
    corrected = PhotonicEncodingModel(4, "hard", 40, "threshold", mesh_order="ps_bs")
    assert not bool(_live_parameters(planned).all())
    assert bool(_live_parameters(corrected).all())


def test_interleaved_layout_splits_the_mesh_and_keeps_d():
    model = PhotonicEncodingModel(
        4, "hard", 40, "threshold", mesh_order="ps_bs", layout="interleaved"
    )
    assert model._mesh_blocks() == [7, 7, 6]
    names = [
        parameter.name
        for parameter in model.circuit.get_parameters()
        if parameter.name.startswith("t")
    ]
    assert len(names) == 40 and names == sorted(names)
    x, theta = _batch(model, 2)
    assert torch.allclose(
        model.probabilities(x, theta).sum(dim=1), torch.ones(2, dtype=torch.float64)
    )


def test_pnr_parity_differs_from_click_parity_only_on_bunched_outcomes():
    clicks = PhotonicEncodingModel(4, "hard", 16, "threshold")
    counts = PhotonicEncodingModel(4, "hard", 16, "threshold", readout="pnr_parity")
    layer = clicks.layer(CPU, torch.float64)
    mask_clicks = clicks._odd_click_mask(layer, CPU)
    mask_counts = counts._odd_click_mask(layer, CPU)
    for state, a, b in zip(
        layer.output_keys, mask_clicks.tolist(), mask_counts.tolist()
    ):
        if max(state[1::2]) <= 1:
            assert a == b


@pytest.mark.skipif(
    not hasattr(__import__("merlin"), "EmitterState"),
    reason="EmitterState is in the photonic_engines fork only",
)
@pytest.mark.parametrize("input_kind", ["cluster", "ghz"])
def test_emitter_inputs_run_with_gradients(input_kind):
    model = PhotonicEncodingModel(
        4, "hard", 16, "threshold", mesh_order="ps_bs", input_kind=input_kind
    )
    x, theta = _batch(model, 3, seed=17)
    theta.requires_grad_(True)
    probabilities = model.probabilities(x, theta)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3, dtype=torch.float64))
    (gradient,) = torch.autograd.grad(probabilities[:, 1].sum(), theta)
    assert gradient.abs().sum() > 0
    separable = PhotonicEncodingModel(4, "hard", 16, "threshold", mesh_order="ps_bs")
    assert not torch.allclose(
        probabilities.detach(), separable.probabilities(x, theta.detach())
    )


@pytest.mark.skipif(
    not hasattr(__import__("merlin"), "EmitterState"),
    reason="EmitterState is in the photonic_engines fork only",
)
def test_ghz_input_equals_explicit_ghz_state_vector():
    import merlin as ML

    model = PhotonicEncodingModel(
        4, "hard", 16, "threshold", mesh_order="ps_bs", input_kind="ghz"
    )
    x, theta = _batch(model, 2, seed=19)
    ghz = pcvl.StateVector(pcvl.BasicState([1, 0, 1, 0, 1, 0])) + pcvl.StateVector(
        pcvl.BasicState([0, 1, 0, 1, 0, 1])
    )
    ghz.normalize()
    layer = ML.QuantumLayer(
        circuit=model.circuit.copy(),
        input_parameters=model.input_groups,
        input_state=ghz,
        measurement_strategy=ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK),
        dtype=torch.float64,
    )
    expected = layer(*model._inputs(x, theta))
    assert torch.allclose(
        model.layer(CPU, torch.float64)(*model._inputs(x, theta)), expected, atol=1e-12
    )


def test_qloq_single_photon_reproduces_qubit_qnn_exactly():
    from power_of_QNN.lib.qloq_merlin import QloqQubitQNN
    from power_of_QNN.lib.quantum import QubitQNN

    generator = torch.Generator().manual_seed(23)
    for feature_map, reps in (("zz", 2), ("z", 1)):
        qloq, qubit = (
            QloqQubitQNN(3, feature_map, reps, 2),
            QubitQNN(3, feature_map, reps, 2),
        )
        x = torch.randn(4, 3, dtype=torch.float64, generator=generator)
        theta = (
            2
            * math.pi
            * torch.rand(4, qubit.d, dtype=torch.float64, generator=generator)
        )
        assert torch.allclose(
            qloq.probabilities(x, theta), qubit.probabilities(x, theta), atol=1e-12
        )
        assert qloq.n_modes == 8 and qloq.n_photons == 1


def test_feedforward_fisher_matches_finite_differences():
    from power_of_QNN.lib.feedforward_merlin import FeedForwardPhotonicModel

    model = FeedForwardPhotonicModel("easy")
    generator = torch.Generator().manual_seed(29)
    x = torch.randn(2, 4, dtype=torch.float64, generator=generator)
    theta = 2 * math.pi * torch.rand(model.d, dtype=torch.float64, generator=generator)
    ours = model.summed_fisher_shared(x, theta)
    base = model.probabilities_shared(x, theta).detach()
    step = 1e-6
    derivatives = []
    for index in range(model.d):
        shifted = theta.clone()
        shifted[index] += step
        derivatives.append(
            (model.probabilities_shared(x, shifted).detach() - base) / step
        )
    derivative = torch.stack(derivatives, dim=-1)  # (B, 2, d)
    expected = torch.einsum(
        "bki,bkj->ij",
        derivative / base.sqrt().unsqueeze(-1),
        derivative / base.sqrt().unsqueeze(-1),
    )
    assert torch.allclose(ours, expected, atol=1e-4 * expected.abs().max())
    assert torch.allclose(base.sum(dim=1), torch.ones(2, dtype=torch.float64))
