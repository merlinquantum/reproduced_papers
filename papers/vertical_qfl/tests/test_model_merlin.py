"""Focused tests of the MerLin photonic adaptation (Phase 4)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import merlin as ml
import pytest
import torch
from common import PROJECT_DIR, REPO_ROOT, SEMEION_FILE
from lib import runner
from lib.data import build_splits
from lib.model_merlin import PhotonicParty, PhotonicVerticalFederatedModel

from runtime_lib import run_from_project

N_MODES, N_PHOTONS = 8, 4


def _model(
    slices,
    server_encoding,
    scale=math.pi,
    server_modes=8,
    server_head=0,
    space="unbunched",
    photons=N_PHOTONS,
    reuploads=1,
    client_readout="mode_expectations",
):
    return PhotonicVerticalFederatedModel(
        client_slices=slices,
        server_encoding=server_encoding,
        n_modes=N_MODES,
        n_photons=photons,
        client_encoding_scale=scale,
        server_encoding_scale=math.pi,
        n_client_stages=4,
        server_modes=server_modes,
        server_head=server_head,
        computation_space=space,
        reuploads=reuploads,
        client_readout=client_readout,
    )


def _pixels(batch=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(batch, 256, generator=generator) > 0.5).double() * math.pi / 4


def test_shapes_and_probability_output():
    model = _model(list(range(8)), "server_wide")
    log_probabilities = model(_pixels())
    assert log_probabilities.shape == (3, 10)
    assert torch.allclose(log_probabilities.exp().sum(dim=1), torch.ones(3), atol=1e-4)


def test_client_message_conserves_photon_number():
    party = _model([0], "server_narrow").clients[0]
    message = party((torch.rand(5, 32) > 0.5).float())
    assert message.shape == (5, 8)
    # mode expectations of 4 photons in 8 modes sum to the photon number
    assert torch.allclose(message.sum(dim=1), torch.full((5,), 4.0), atol=1e-4)


def test_parameter_counts_follow_the_stage_layout():
    wide = _model(list(range(8)), "server_wide").trainable_parameter_counts()
    narrow = _model([0], "server_narrow").trainable_parameter_counts()
    assert wide["client_0"] == 5 * 56 and wide["server"] == 9 * 56
    assert wide["total"] == 8 * 280 + 504
    assert narrow["server"] == 5 * 56 and narrow["total"] == 560


def test_gradients_reach_every_parameter():
    model = _model(list(range(8)), "server_wide")
    labels = torch.tensor([0, 1, 2])
    torch.nn.CrossEntropyLoss()(model(_pixels()), labels).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    assert all(p.grad.abs().sum() > 0 for p in model.clients[3].parameters())
    assert sum(p.grad.abs().sum() for p in model.server.parameters()) > 0


def test_configured_space_photons_and_input_state_are_used():
    layer = _model([0], "server_narrow").clients[0].layer
    assert layer.computation_space == ml.ComputationSpace.UNBUNCHED
    assert layer.n_photons == N_PHOTONS
    assert list(layer.input_state) == [1, 0, 1, 0, 1, 0, 1, 0]
    assert layer.measurement_strategy.computation_space == ml.ComputationSpace.UNBUNCHED


def test_encoding_scale_is_applied_with_letter_names():
    """Guards the MerLin quirk: digit-ending encoding names silently ignore `scale`."""
    torch.manual_seed(0)
    first = PhotonicParty(4, "mode_expectations", math.pi, N_MODES, N_PHOTONS)
    torch.manual_seed(0)
    second = PhotonicParty(4, "mode_expectations", math.pi / 2, N_MODES, N_PHOTONS)
    values = (torch.rand(3, 32) > 0.5).float()
    assert not torch.allclose(first(values), second(values))
    torch.manual_seed(0)
    reference = PhotonicParty(4, "mode_expectations", 1.0, N_MODES, N_PHOTONS)
    assert torch.allclose(first(values), reference(values * math.pi), atol=1e-5)


def test_every_pixel_of_a_client_slice_influences_its_message():
    party = PhotonicParty(4, "mode_expectations", math.pi, N_MODES, N_PHOTONS)
    values = (torch.rand(2, 32) > 0.5).float()
    base = party(values).detach()
    for index in range(32):
        flipped = values.clone()
        flipped[:, index] = 1 - flipped[:, index]
        assert (party(flipped).detach() - base).abs().sum() > 1e-6, index


def test_narrow_server_reuploads_the_client_message():
    model = _model([0], "server_narrow")
    assert model.server.n_inputs == 32 and model.server_repeats == 4


def test_same_seed_gives_same_output():
    torch.manual_seed(7)
    first = _model([0], "server_narrow")(_pixels())
    torch.manual_seed(7)
    second = _model([0], "server_narrow")(_pixels())
    assert torch.allclose(first, second)


def test_invalid_layouts_fail_clearly():
    with pytest.raises(ValueError, match="server_narrow takes exactly one client"):
        _model([0, 1], "server_narrow")
    with pytest.raises(ValueError, match="server_modes must be at least"):
        _model([0], "server_narrow", server_modes=6)
    with pytest.raises(ValueError, match="No input-state rule"):
        PhotonicParty(2, "mode_expectations", 1.0, 8, 9)


def test_config_values_reach_the_model():
    cfg = json.loads(
        (PROJECT_DIR / "configs/e1_8clients_merlin_tuning.json").read_text()
    )
    cfg["model"]["client_encoding_scale"] = 0.5
    model = runner._build_model(cfg, torch.float32)
    assert isinstance(model, PhotonicVerticalFederatedModel)
    assert len(model.clients) == 8 and model.client_slices == [0, 1, 2, 3, 4, 5, 6, 7]


def test_missing_merlin_key_is_reported():
    cfg = json.loads(
        (PROJECT_DIR / "configs/e1_8clients_merlin_tuning.json").read_text()
    )
    del cfg["model"]["n_photons"]
    with pytest.raises(KeyError, match="model.n_photons"):
        runner._validate_config(cfg)


pytestmark_data = pytest.mark.skipif(
    not SEMEION_FILE.exists(), reason="Semeion data not downloaded"
)


@pytestmark_data
def test_validation_split_is_carved_from_train_and_test_is_unchanged():
    cfg = json.loads(
        (PROJECT_DIR / "configs/e1_8clients_merlin_tuning.json").read_text()
    )
    cfg["data_root"] = str(REPO_ROOT / "data")
    with_validation = build_splits(cfg)
    cfg["dataset"]["validation_fraction"] = None
    without = build_splits(cfg)
    assert len(with_validation["validation"][1]) == 255
    assert len(with_validation["train"][1]) + 255 == len(without["train"][1]) == 1274
    assert torch.equal(with_validation["test"][1], without["test"][1])


@pytestmark_data
def test_photonic_smoke_run_writes_validated_evidence(tmp_path):
    original = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--config",
                "configs/e1_8clients_merlin_tuning.json",
                "--outdir",
                str(tmp_path),
                "--epochs",
                "1",
                "--max-train-samples",
                "64",
                "--max-test-samples",
                "32",
            ],
        )
    finally:
        os.chdir(original)
    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert (
        status["status"] == "COMPLETED" and status["selection"]["split"] == "validation"
    )
    assert metrics["backend"] == "merlin" and "final_validation_accuracy" in metrics
    assert metrics["trainable_parameters"]["total"] == 8 * 280 + 504


def test_larger_server_chip_with_fixed_grouping():
    """12 modes / 6 photons: 924 outcomes, grouped modulo 10 (924 is not a multiple of 10)."""
    model = _model(list(range(8)), "server_wide", server_modes=12)
    layer = model.server.layer
    assert layer.output_size == 924 and layer.n_photons == 6
    assert isinstance(model.server.group, ml.ModGrouping)
    assert model.server.n_inputs == 64
    log_probabilities = model(_pixels())
    assert torch.allclose(log_probabilities.exp().sum(dim=1), torch.ones(3), atol=1e-4)


def test_eight_mode_server_keeps_lexicographic_grouping():
    assert isinstance(
        _model(list(range(8)), "server_wide").server.group, ml.LexGrouping
    )


def test_trainable_server_head_outputs_log_probabilities_and_is_counted():
    model = _model(list(range(8)), "server_wide", server_head=True)
    log_probabilities = model(_pixels())
    assert torch.allclose(log_probabilities.exp().sum(dim=1), torch.ones(3), atol=1e-5)
    counts = model.trainable_parameter_counts()
    assert counts["server_classical_head"] == 70 * 10 + 10
    assert counts["server"] == 504 + 710 and counts["total"] == 8 * 280 + 504 + 710
    torch.nn.CrossEntropyLoss()(log_probabilities, torch.tensor([0, 1, 2])).backward()
    assert model.server.head.weight.grad is not None


def test_fully_photonic_model_has_no_classical_head():
    counts = _model(list(range(8)), "server_wide").trainable_parameter_counts()
    assert counts["server_classical_head"] == 0


def test_fock_space_keeps_bunched_outcomes_without_post_selection():
    model = _model(list(range(8)), "server_wide", space="fock")
    assert model.server.layer.computation_space == ml.ComputationSpace.FOCK
    assert model.server.layer.output_size == 330  # C(8 + 4 - 1, 4)
    log_probabilities = model(_pixels())
    assert torch.allclose(log_probabilities.exp().sum(dim=1), torch.ones(3), atol=1e-4)


def test_one_photon_per_mode_state_in_fock_space():
    model = _model([0], "server_narrow", space="fock", photons=8)
    layer = model.clients[0].layer
    assert list(layer.input_state) == [1] * 8 and layer.n_photons == 8
    message = model.clients[0]((torch.rand(2, 32) > 0.5).float())
    assert torch.allclose(message.sum(dim=1), torch.full((2,), 8.0), atol=1e-3)


def test_marginal_head_is_small_and_uses_photon_number_marginals():
    model = _model(
        list(range(8)), "server_wide", space="fock", photons=8, server_head=2
    )
    server = model.server
    assert server.layer.output_size == 6435
    assert server.marginals.shape == (6435, 8 * 9)
    # each outcome contributes to exactly one photon count per mode
    assert torch.all(server.marginals.sum(dim=1) == 8)
    assert model.trainable_parameter_counts()["server_classical_head"] == 72 * 10 + 10
    log_probabilities = model(_pixels())
    assert torch.allclose(log_probabilities.exp().sum(dim=1), torch.ones(3), atol=1e-5)


def test_reuploading_doubles_the_stages_of_every_party():
    single = _model(list(range(8)), "server_wide")
    double = _model(list(range(8)), "server_wide", reuploads=2)
    assert double.clients[0].n_inputs == 2 * single.clients[0].n_inputs == 64
    assert double.server.n_inputs == 2 * single.server.n_inputs == 128
    narrow = _model([0], "server_narrow", reuploads=2)
    assert narrow.server.n_inputs == 64 and narrow(_pixels()).shape == (3, 10)


def test_mean_photon_number_message_is_blind_with_one_photon_per_mode():
    """Unitarity: with |1,...,1> every <n_k> is 1 whatever the data (the D2 failure)."""
    party = _model([0], "server_narrow", space="fock", photons=8).clients[0]
    values = (torch.rand(4, 32) > 0.5).float()
    assert torch.allclose(party(values), torch.ones(4, 8), atol=1e-4)


def test_click_probability_message_depends_on_the_data():
    model = _model(
        [0],
        "server_narrow",
        space="fock",
        photons=8,
        client_readout="click_probabilities",
    )
    party = model.clients[0]
    values = (torch.rand(4, 32) > 0.5).float()
    message = party(values)
    assert message.shape == (4, 8)
    assert torch.all((message >= 0) & (message <= 1 + 1e-5))
    assert message.std(dim=0).max() > 1e-3
    assert model(_pixels()).shape == (3, 10)


def _v3_model(circuit, n_modes=12, stages=8, server_modes=8, photons=4):
    return PhotonicVerticalFederatedModel(
        client_slices=list(range(8)),
        server_encoding="server_wide",
        n_modes=n_modes,
        n_photons=photons,
        client_encoding_scale=math.pi / 8,
        server_encoding_scale=math.pi,
        n_client_stages=stages,
        server_modes=server_modes,
        server_head=0,
        computation_space="fock",
        client_readout="click_probabilities",
        circuit=circuit,
    )


def test_superposition_first_layer_replaces_only_the_first_mesh():
    mesh = _model(list(range(8)), "server_wide", space="fock")
    sup = PhotonicVerticalFederatedModel(
        client_slices=list(range(8)),
        server_encoding="server_wide",
        n_modes=8,
        n_photons=4,
        client_encoding_scale=math.pi / 8,
        server_encoding_scale=math.pi,
        n_client_stages=4,
        server_modes=8,
        server_head=0,
        computation_space="fock",
        circuit={"first_layer": "superposition"},
    )
    # One 8-mode mesh has 56 phases; one beam-splitter layer has 7 x (theta, phi).
    assert (
        sum(p.numel() for p in mesh.clients[0].parameters())
        - sum(p.numel() for p in sup.clients[0].parameters())
        == 56 - 14
    )
    assert sup(_pixels()).shape == (3, 10)


def test_feature_maps_use_separate_modes_and_message_width_is_kept():
    model = _v3_model(
        {
            "client_feature_maps": ["identity", "square", "cos"],
            "message_width": 8,
            "server_photons": 4,
        }
    )
    client = model.clients[0]
    assert client.raw_per_stage == 4 and client.n_inputs == 32
    assert client.layer.input_size == 8 * 12
    values = (torch.rand(4, 32) > 0.5).float()
    assert client(values).shape == (4, 8)
    assert model(_pixels()).shape == (3, 10)


def test_feature_maps_change_a_continuous_server_input():
    model = _v3_model(
        {
            "first_layer": "superposition",
            "client_feature_maps": ["identity", "square", "cos"],
            "server_feature_maps": ["identity", "square", "cos"],
            "message_width": 8,
            "server_photons": 4,
        },
        server_modes=12,
    )
    server = model.server
    assert server.raw_per_stage == 4 and server.n_stages == 16
    message = torch.rand(2, 64)
    other = message.clone()
    other[:, 0] += 0.3
    assert not torch.allclose(server(message), server(other))
    model(_pixels()).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_invalid_circuit_options_fail_clearly():
    with pytest.raises(ValueError, match="feature_maps"):
        _v3_model({"client_feature_maps": ["bogus"]})
    with pytest.raises(ValueError, match="first_layer"):
        _v3_model({"first_layer": "bogus"}, n_modes=8, stages=4)
    with pytest.raises(ValueError, match="do not encode"):
        _v3_model({"client_feature_maps": ["identity", "cos"]}, n_modes=12, stages=8)


def test_superposition_mixer_shrinks_the_client_parameter_count():
    maps = ["identity"] * 3
    counts = {}
    for mixer, depth in (("mesh", 1), ("superposition", 1), ("superposition", 2)):
        model = _v3_model(
            {
                "client_feature_maps": maps,
                "message_width": 8,
                "server_photons": 4,
                "client_mixer": mixer,
                "mixer_depth": depth,
            }
        )
        counts[(mixer, depth)] = model.trainable_parameter_counts()["client_0"]
    # 9 mixer layers (8 stages + final) of 11 beam splitters x (angle, phase) per pass.
    assert counts[("superposition", 1)] == 9 * 22
    assert counts[("superposition", 2)] == 2 * 9 * 22
    assert counts[("mesh", 1)] > 2 * counts[("superposition", 2)]
