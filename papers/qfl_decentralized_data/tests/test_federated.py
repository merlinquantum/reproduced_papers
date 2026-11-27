from __future__ import annotations

import pytest
import torch
from common import PROJECT_DIR  # noqa: F401
from lib.federated import _weighted_average, run_training
from lib.task import Task
from torch import nn


class Scalar(nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))


def _task(client_sizes: list[int], target: float = 1.0) -> Task:
    def loss(model, batch):
        (values,) = batch
        return ((model.weight - values) ** 2).mean()

    data = [(torch.full((n,), target),) for n in client_sizes]
    return Task(
        model=Scalar(),
        client_data=data,
        loss=loss,
        evaluate=lambda model: {"weight": model.weight.item()},
        primary_metric="weight",
        final_report=lambda model: {},
    )


def _cfg(**federation):
    base = {
        "mode": "federated", "rounds": 2, "local_epochs": 1, "n_clients": 2,
        "clients_per_round": "all", "server_lr": 1.0, "partition": "iid",
    }  # fmt: skip
    return {
        "seed": 0,
        "federation": {**base, **federation},
        "training": {"optimizer": "adam", "lr": 0.1, "batch_size": 4},
    }


def test_weighted_average_weights_clients_by_sample_count():
    states = [{"w": torch.tensor(0.0)}, {"w": torch.tensor(4.0)}]
    assert _weighted_average(states, [1.0, 3.0], ["w"])["w"].item() == pytest.approx(
        3.0
    )


class ScalarWithBuffer(Scalar):
    def __init__(self):
        super().__init__()
        self.register_buffer("mask", torch.full((3,), 1.0))


def test_fedavg_averages_parameters_only_and_keeps_buffers_exactly():
    # MerLin layers carry a constant float buffer; with client weights like 1/3 the
    # weighted sum of identical values is not exactly 1.0, so buffers must not be averaged.
    task = _task([7, 11, 13])
    task.model = ScalarWithBuffer()
    run_training(
        _cfg(n_clients=3, rounds=3), task, torch.device("cpu"), lambda entry: None
    )
    assert torch.equal(task.model.mask, torch.full((3,), 1.0))
    assert task.model.weight.item() != 0.0


def test_zero_learning_rate_leaves_the_global_model_unchanged():
    cfg = _cfg()
    cfg["training"]["lr"] = 0.0
    task = _task([8, 8])
    history = run_training(cfg, task, torch.device("cpu"), lambda entry: None)
    assert task.model.weight.item() == 0.0
    assert [entry["round"] for entry in history] == [1, 2]


def test_server_lr_zero_ignores_the_client_updates():
    task = _task([8, 8])
    run_training(_cfg(server_lr=0.0), task, torch.device("cpu"), lambda entry: None)
    assert task.model.weight.item() == 0.0


def test_federated_training_moves_towards_the_clients_optimum():
    task = _task([16, 8], target=1.0)
    cfg = _cfg(rounds=10)
    run_training(cfg, task, torch.device("cpu"), lambda entry: None)
    assert 0.3 < task.model.weight.item() <= 1.2


def test_centralized_mode_uses_only_the_pooled_client():
    task = _task([24])
    cfg = _cfg(mode="centralized", n_clients=1, rounds=5)
    history = run_training(cfg, task, torch.device("cpu"), lambda entry: None)
    assert history[-1]["weight"] > history[0]["weight"]


def test_training_is_deterministic_for_a_fixed_seed():
    first, second = _task([12, 6]), _task([12, 6])
    run_training(_cfg(), first, torch.device("cpu"), lambda entry: None)
    run_training(_cfg(), second, torch.device("cpu"), lambda entry: None)
    assert first.model.weight.item() == second.model.weight.item()


def test_invalid_clients_per_round_fails_clearly():
    with pytest.raises(ValueError, match="clients_per_round"):
        run_training(
            _cfg(clients_per_round=3),
            _task([8, 8]),
            torch.device("cpu"),
            lambda e: None,
        )
    with pytest.raises(ValueError, match="mode"):
        run_training(
            _cfg(mode="sideways"), _task([8, 8]), torch.device("cpu"), lambda e: None
        )
