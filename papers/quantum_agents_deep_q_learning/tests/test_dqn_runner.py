"""DQN loop behaviour and an end-to-end tiny run through the runner."""

from __future__ import annotations

import json
from copy import deepcopy

import torch
from common import PROJECT_DIR, load_runtime_ready_config
from lib.dqn import DQNTrainer
from lib.runner import train_and_evaluate


def load_config(name: str) -> dict:
    return json.loads((PROJECT_DIR / "configs" / f"{name}.json").read_text())


def _tiny(cfg: dict, seeds: list[int], episodes: int) -> dict:
    cfg = deepcopy(cfg)
    cfg["agents"]["seeds"] = seeds
    cfg["training"]["max_episodes"] = episodes
    return cfg


def test_agents_are_independent_of_their_batch_mates():
    cfg = _tiny(load_config("e2e3_cartpole_optimal_full_original"), [5], 6)
    alone = DQNTrainer(cfg, torch.device("cpu"), torch.float64).train()
    together = DQNTrainer(
        _tiny(cfg, [5, 9], 6), torch.device("cpu"), torch.float64
    ).train()
    assert alone.scores[0] == together.scores[0]
    assert alone.total_env_steps[0] == together.total_env_steps[0]


def test_epsilon_decays_once_per_episode_and_episode_cap_is_respected():
    cfg = _tiny(load_config("e1_frozenlake_l5_original"), [0, 1], 4)
    result = DQNTrainer(cfg, torch.device("cpu"), torch.float32).train()
    for scores, epsilon in zip(result.scores, result.final_epsilon):
        assert len(scores) == 4
        assert abs(epsilon - 0.99**4) < 1e-12
    assert result.q_star_mae is not None
    assert len(result.q_star_mae["all_64"][0]) == result.total_env_steps[0]


def test_solve_rule_stops_agent():
    cfg = _tiny(load_config("e1_frozenlake_l5_original"), [0], 10)
    trainer = DQNTrainer(cfg, torch.device("cpu"), torch.float32)
    assert not trainer._is_solved([1.0] * 99)
    assert trainer._is_solved([0.0] + [1.0] * 100)
    assert not trainer._is_solved([1.0] * 99 + [0.0])
    cp_trainer = DQNTrainer(
        _tiny(load_config("e2e3_cartpole_optimal_full_original"), [0], 1),
        torch.device("cpu"),
        torch.float32,
    )
    assert not cp_trainer._is_solved(
        [200.0] * 100
    )  # reference code needs > 100 episodes
    assert cp_trainer._is_solved([200.0] * 101)


def test_runner_smoke_writes_contract_artifacts(tmp_path):
    cfg = load_runtime_ready_config()
    cfg["agents"]["seeds"] = [0, 1]
    cfg["training"]["max_episodes"] = 2
    cfg["evaluation"]["greedy_episodes"] = 1
    run_dir = tmp_path / "run_test"
    train_and_evaluate(cfg, run_dir)
    for name in (
        "run.log",
        "run_status.json",
        "config_snapshot.json",
        "metrics.json",
        "scores.npz",
        "score_curve.png",
    ):
        assert (run_dir / name).exists(), name
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["trainable_parameters_per_agent"] == 62
    assert len(metrics["agents"]) == 2
    assert all(agent["episodes_run"] == 2 for agent in metrics["agents"])


def test_phase5_selection_grid_is_matched_and_complete():
    """24 grid configs; only batch, target interval and seeds differ from the source."""
    import itertools

    sources = {
        "mlp_5_5": ("e4_cartpole_mlp_5_5_codefaithful", 67),
        "mlp_14_14": ("e4_cartpole_mlp_14_14_codefaithful", 310),
        "pqc_l5": ("e2e3_cartpole_optimal_full_codefaithful", 62),
        "pqc_l25": ("e4_cartpole_pqc_l25_codefaithful", 302),
    }
    from lib.models import build_q_function

    for key, (source, params) in sources.items():
        base = load_config(source)
        suffix = "_classical" if key.startswith("mlp") else ""
        for batch, target in itertools.product((16, 32, 64), (1, 30)):
            cfg = load_config(f"e5_select_{key}_b{batch}_t{target}{suffix}")
            assert cfg["agents"]["seeds"] == [100, 101, 102, 103, 104]
            assert cfg["training"]["batch_size"] == batch
            assert cfg["training"]["update_target_every"] == target
            for section in ("environment", "model", "evaluation"):
                assert cfg[section] == base[section]
            for name, value in base["training"].items():
                if name not in ("batch_size", "update_target_every"):
                    assert cfg["training"][name] == value, name
            model = build_q_function(cfg, [0], 2, torch.float32)
            assert model.count_trainable_parameters() == params
