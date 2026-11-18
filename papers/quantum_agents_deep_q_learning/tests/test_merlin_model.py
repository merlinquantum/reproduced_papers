"""Phase 4 MerLin adaptation: correctness against Perceval, interface and wiring."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import perceval as pcvl
import pytest
import torch
from common import PROJECT_DIR
from lib.models import build_q_function
from lib.qfunction_merlin import (
    MerlinQFunction,
    brickwork_pairs,
    build_circuit,
)


def load_config(name: str) -> dict:
    return json.loads((PROJECT_DIR / "configs" / f"{name}.json").read_text())


def perceval_click_probabilities(model, agent, observation):
    """Independent reference: SLOS on the Perceval circuit with the same phases.

    Threshold readout: P(click on mode k) = sum of P(outcome) with n_k >= 1 over
    the full Fock distribution (no post-selection). Unbunched readout:
    renormalised over collision-free outcomes.
    """
    circuit, _, _ = build_circuit(
        model.n_modes,
        model.n_layers,
        model.n_inputs,
        model.data_reuploading,
        dual_rail_input=model.state_encoding == "dual_rail_input",
    )
    mesh = model.theta[agent].detach().double()
    # Encode with this agent's own input weights (every agent row sees the observation).
    per_agent = observation.expand(model.n_agents, 1, observation.shape[-1])
    encoding = model.encoding_phases(per_agent)[agent, 0].detach().double()
    for parameter in circuit.get_parameters():
        index = int(parameter.name[1:])
        value = mesh[index] if parameter.name.startswith("w") else encoding[index]
        parameter.set_value(float(value) % (2 * math.pi))
    backend = pcvl.SLOSBackend()
    backend.set_circuit(circuit)
    backend.set_input_state(pcvl.BasicState(model.input_state))
    clicks = torch.zeros(model.n_modes, dtype=torch.float64)
    total = 0.0
    for state, probability in backend.prob_distribution().items():
        occupation = list(state)
        if model.readout == "unbunched_click" and max(occupation) > 1:
            continue
        total += probability
        clicks += torch.tensor([float(n > 0) for n in occupation]) * probability
    return clicks / total if model.readout == "unbunched_click" else clicks


@pytest.mark.parametrize(
    "name, readout",
    [
        ("e2e3_cartpole_optimal_full_merlin", "threshold_click"),
        ("e2e3_cartpole_optimal_full_merlin", "unbunched_click"),
        ("e2e3_cartpole_optimal_no_reuploading_merlin", "threshold_click"),
        ("e1_frozenlake_l5_merlin", "threshold_click"),
        ("e1_frozenlake_l5_merlin_reupload", "threshold_click"),
        ("e1_frozenlake_l5_merlin_dualrail", "threshold_click"),
        ("e1_frozenlake_l5_merlin_8modes", "threshold_click"),
    ],
)
def test_click_probabilities_match_perceval_slos(name, readout):
    cfg = load_config(name)
    cfg["model"]["readout"] = readout
    model = MerlinQFunction(
        cfg, [3, 11], n_actions=4 if "frozen" in name else 2, real_dtype=torch.float64
    )
    with torch.no_grad():
        model.input_weights.uniform_(0.5, 1.5)
    if "frozen" in name:
        observations = torch.tensor(
            [[[5.0], [14.0]], [[0.0], [9.0]]], dtype=torch.float64
        )
    else:
        observations = torch.randn(
            2, 2, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
        )
    clicks = model.click_probabilities(observations)
    for agent in range(2):
        for sample in range(2):
            expected = perceval_click_probabilities(
                model, agent, observations[agent, sample]
            )
            torch.testing.assert_close(
                clicks[agent, sample], expected, atol=1e-7, rtol=0
            )


def test_unbunched_readout_postselects_on_all_photons():
    cfg = load_config("e2e3_cartpole_optimal_full_merlin")
    cfg["model"]["readout"] = "unbunched_click"
    model = MerlinQFunction(cfg, [0], 2, torch.float64)
    clicks = model.click_probabilities(torch.randn(1, 8, 4, dtype=torch.float64))
    torch.testing.assert_close(
        clicks.sum(-1), torch.full((1, 8), 3.0, dtype=torch.float64)
    )
    threshold = MerlinQFunction(
        load_config("e2e3_cartpole_optimal_full_merlin"), [0], 2, torch.float64
    )
    total = threshold.click_probabilities(
        torch.randn(1, 8, 4, dtype=torch.float64)
    ).sum(-1)
    assert float(total.max()) < 3.0  # bunching lowers the expected number of clicks


def test_selected_space_and_detector_are_used():
    import merlin as ml

    model = MerlinQFunction(
        load_config("e2e3_cartpole_optimal_full_merlin"), [0], 2, torch.float32
    )
    assert model.readout == "threshold_click"
    assert model.computation_space == ml.ComputationSpace.FOCK
    assert model.layer.output_size == model.click_matrix.shape[0]
    assert set(model.click_matrix.unique().tolist()) == {0.0, 1.0}
    assert (
        len(list(model.layer.parameters())) == 0
    )  # all phases are agent-stacked inputs


@pytest.mark.parametrize(
    "name, expected",
    [
        ("e2e3_cartpole_optimal_full_merlin", 60 + 20 + 2),
        ("e2e3_cartpole_optimal_no_reuploading_merlin", 60 + 4 + 2),
        ("e2e3_cartpole_optimal_no_input_weights_merlin", 60 + 2),
        ("e2e3_cartpole_optimal_fixed_range_90_merlin", 60 + 20),
        ("e1_frozenlake_l5_merlin", 60),
        ("e1_frozenlake_l10_merlin", 110),
        ("e1_frozenlake_l15_merlin", 160),
        ("e1_frozenlake_l10_merlin_reupload", 110),
        ("e1_frozenlake_l5_merlin_dualrail", 84),
        ("e1_frozenlake_l5_merlin_8modes", 84),
    ],
)
def test_parameter_counts(name, expected):
    n_actions = 4 if "frozen" in name else 2
    model = MerlinQFunction(load_config(name), [0], n_actions, torch.float32)
    assert model.count_trainable_parameters() == expected


def test_configs_reach_the_model_and_match_qubit_training_settings():
    for name in (
        "e2e3_cartpole_suboptimal_fixed_range_180_merlin",
        "e1_frozenlake_l10_merlin",
    ):
        cfg = load_config(name)
        qubit = load_config(name.replace("_merlin", "_codefaithful"))
        assert cfg["training"] == qubit["training"]
        assert cfg["environment"] == qubit["environment"]
        assert cfg["agents"] == qubit["agents"]
        assert cfg["evaluation"] == qubit["evaluation"]
        model = build_q_function(cfg, [0], 4 if "frozen" in name else 2, torch.float32)
        assert isinstance(model, MerlinQFunction)
        assert model.n_layers == cfg["model"]["n_layers"]
        assert model.readout_index.tolist() == cfg["model"]["readout_modes"]
        assert model.input_state == cfg["model"]["input_state"]
        assert model.output_factor == cfg["model"]["output_factor"]
        assert (
            model.output_weights.requires_grad == cfg["model"]["train_output_weights"]
        )


def test_output_range_and_shapes():
    model = MerlinQFunction(
        load_config("e2e3_cartpole_optimal_fixed_range_90_merlin"),
        [0, 1],
        2,
        torch.float32,
    )
    q_values = model(torch.randn(2, 16, 4))
    assert q_values.shape == (2, 16, 2)
    assert float(q_values.min()) >= 0.0 and float(q_values.max()) <= 90.0 + 1e-4
    frozen = MerlinQFunction(
        load_config("e1_frozenlake_l5_merlin"), [0], 4, torch.float32
    )
    q_frozen = frozen(torch.arange(16, dtype=torch.float32).view(1, 16, 1))
    assert q_frozen.shape == (1, 16, 4)
    assert float(q_frozen.min()) >= 0.0 and float(q_frozen.max()) <= 1.0 + 1e-6


def test_frozen_lake_bit_order_matches_qubit_basis_state():
    frozen = MerlinQFunction(
        load_config("e1_frozenlake_l5_merlin"), [0], 4, torch.float32
    )
    phases = frozen.encoding_phases(torch.tensor([[[8.0], [1.0], [5.0]]]))[0]
    expected = math.pi * torch.tensor(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]], dtype=torch.float32
    )
    torch.testing.assert_close(phases, expected)


def test_gradients_reach_every_trainable_tensor():
    model = MerlinQFunction(
        load_config("e2e3_cartpole_optimal_full_merlin"), [0, 1], 2, torch.float32
    )
    model(torch.randn(2, 8, 4)).sum().backward()
    for group in model.parameter_groups():
        for tensor in group.tensors:
            assert tensor.grad is not None and float(tensor.grad.abs().sum()) > 0, (
                group.name
            )


def test_agents_are_independent_and_seeded():
    cfg = load_config("e2e3_cartpole_optimal_full_merlin")
    alone = MerlinQFunction(cfg, [7], 2, torch.float64)
    together = MerlinQFunction(cfg, [1, 7], 2, torch.float64)
    assert torch.equal(alone.theta[0], together.theta[1])
    observations = torch.randn(1, 5, 4, dtype=torch.float64)
    torch.testing.assert_close(
        alone(observations)[0],
        together(torch.cat([torch.zeros_like(observations), observations]))[1],
    )
    again = MerlinQFunction(cfg, [7], 2, torch.float64)
    torch.testing.assert_close(alone(observations), again(observations))


def test_invalid_configurations_fail_loudly():
    cfg = load_config("e2e3_cartpole_optimal_full_merlin")
    single = deepcopy(cfg)
    single["model"]["n_photons"], single["model"]["input_state"] = 1, [1, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="at least two photons"):
        MerlinQFunction(single, [0], 2, torch.float32)
    mismatch = deepcopy(cfg)
    mismatch["model"]["input_state"] = [1, 1, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="input_state"):
        MerlinQFunction(mismatch, [0], 2, torch.float32)
    wrong_readout = deepcopy(cfg)
    wrong_readout["model"]["readout_modes"] = [2]
    with pytest.raises(ValueError, match="readout_modes"):
        MerlinQFunction(wrong_readout, [0], 2, torch.float32)
    frozen_weights = load_config("e1_frozenlake_l5_merlin")
    frozen_weights["model"]["train_input_weights"] = True
    with pytest.raises(ValueError, match="no input weights"):
        MerlinQFunction(frozen_weights, [0], 4, torch.float32)
    missing = deepcopy(cfg)
    del missing["model"]["readout"]
    with pytest.raises(KeyError, match="model.readout"):
        MerlinQFunction(missing, [0], 2, torch.float32)


def test_brickwork_layout():
    assert brickwork_pairs(6) == [0, 2, 4, 1, 3]


@pytest.mark.parametrize(
    "name, expected_params",
    [("e2e3_cartpole_optimal_full_merlin", 82), ("e1_frozenlake_l5_merlin", 60)],
)
def test_runner_smoke_with_merlin_model(tmp_path, name, expected_params):
    from common import load_runtime_ready_config
    from lib.runner import train_and_evaluate

    cfg = load_runtime_ready_config()
    for section in (
        "experiment",
        "agents",
        "environment",
        "model",
        "training",
        "evaluation",
    ):
        cfg[section] = load_config(name)[section]
    cfg["agents"]["seeds"] = [0, 1]
    cfg["training"]["max_episodes"] = 2
    cfg["evaluation"]["greedy_episodes"] = 1
    run_dir = tmp_path / "run_merlin"
    train_and_evaluate(cfg, run_dir)
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    assert status["candidate"]["model_kind"] == "merlin"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["trainable_parameters_per_agent"] == expected_params
    assert all(agent["episodes_run"] == 2 for agent in metrics["agents"])


def test_dual_rail_preparation_gives_the_basis_fock_state():
    """Preparation MZIs with phases pi * (1 - bit) map [1,0,1,0,...] to the dual-rail state."""
    model = MerlinQFunction(
        load_config("e1_frozenlake_l5_merlin_dualrail"), [0], 4, torch.float64
    )
    phases = model.encoding_phases(
        torch.arange(16, dtype=torch.float64).view(1, 16, 1)
    )[0]
    for state in range(16):
        circuit = pcvl.Circuit(8)
        for pair in range(4):
            prep = (
                pcvl.Circuit(2)
                // pcvl.BS()
                // (0, pcvl.PS(float(phases[state, pair])))
                // pcvl.BS()
            )
            circuit.add(2 * pair, prep)
        backend = pcvl.SLOSBackend()
        backend.set_circuit(circuit)
        backend.set_input_state(pcvl.BasicState([1, 0] * 4))
        distribution = backend.prob_distribution()
        bits = [(state >> (3 - i)) & 1 for i in range(4)]
        expected = [0] * 8
        for i, bit in enumerate(bits):
            expected[2 * i + bit] = 1
        assert distribution[pcvl.BasicState(expected)] == pytest.approx(1.0, abs=1e-12)


def test_dual_rail_rejects_reuploading_and_wrong_layout():
    cfg = load_config("e1_frozenlake_l5_merlin_dualrail")
    reupload = deepcopy(cfg)
    reupload["model"]["data_reuploading"] = True
    with pytest.raises(ValueError, match="once"):
        MerlinQFunction(reupload, [0], 4, torch.float32)
    layout = deepcopy(load_config("e1_frozenlake_l5_merlin"))
    layout["model"]["state_encoding"] = "dual_rail_input"
    with pytest.raises(ValueError, match="dual_rail_input needs"):
        MerlinQFunction(layout, [0], 4, torch.float32)
