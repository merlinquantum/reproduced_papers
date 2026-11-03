from __future__ import annotations

import numpy as np
import torch
from power_of_QNN.lib.classical import FeedForwardNetwork
from power_of_QNN.lib.quantum import QubitQNN
from power_of_QNN.lib.training import (
    _gradient,
    initial_parameters,
    per_seed_loss,
    train,
)
from power_of_QNN.lib.training_experiments import randomize_labels

X = torch.tensor(
    [
        [0.9, 0.1, 0.3, 0.2],
        [0.1, 0.8, 0.5, 0.2],
        [0.7, 0.2, 0.1, 0.6],
        [0.2, 0.9, 0.4, 0.1],
    ],
    dtype=torch.float64,
)
Y = torch.tensor([0, 1, 0, 1])
SETTINGS = {
    "iterations": 5,
    "lr": 0.1,
    "finite_difference_step": 1e-10,
    "log_every": 1,
    "gradient": "autograd",
}


def test_seeds_train_independently():
    model = FeedForwardNetwork([4, 2], "leaky_relu", "softmax")
    theta0 = torch.as_tensor(initial_parameters(3, model.d, [0, 1, 2], -1.0, 1.0))
    together = train(model, X, Y, theta0, {**SETTINGS, "optimizer": "torch_adam"})
    alone = train(model, X, Y, theta0[1:2], {**SETTINGS, "optimizer": "torch_adam"})
    assert np.allclose(together["theta"][1], alone["theta"][0])
    assert together["losses"].shape == (6, 3)


def test_torch_adam_matches_torch_optim():
    model = FeedForwardNetwork([4, 1, 1, 1, 2], "leaky_relu", "softmax")
    theta0 = torch.as_tensor(initial_parameters(1, model.d, [7], 0.0, 1.0))
    ours = train(model, X, Y, theta0, {**SETTINGS, "optimizer": "torch_adam"})["theta"]
    parameter = theta0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    for _ in range(5):
        optimizer.zero_grad()
        per_seed_loss(model, X, Y, parameter).sum().backward()
        optimizer.step()
    assert np.allclose(ours, parameter.detach().numpy(), atol=1e-12)


def test_finite_difference_approximates_autograd():
    model = QubitQNN(4, "zz", 2, 1)
    theta = torch.as_tensor(
        initial_parameters(2, model.d, [0, 1], 0.0, 6.283185307179586)
    )
    exact = _gradient(model, X, Y, theta, "autograd", 1e-10)
    numeric = _gradient(model, X, Y, theta, "finite_difference", 1e-7)
    assert torch.allclose(exact, numeric, atol=1e-5)


def test_quantum_training_lowers_loss():
    model = QubitQNN(4, "zz", 2, 1)
    theta0 = torch.as_tensor(
        initial_parameters(2, model.d, [0, 1], 0.0, 6.283185307179586)
    )
    result = train(
        model, X, Y, theta0, {**SETTINGS, "iterations": 10, "optimizer": "qiskit_adam"}
    )
    assert (result["losses"][-1] < result["losses"][0]).all()


def test_randomize_labels_overwrites_only_the_first_fraction():
    labels = np.zeros(100, dtype=np.int64)
    randomized = randomize_labels(labels, 0.4, seed=3)
    assert (randomized[40:] == 0).all()
    assert 0 < randomized[:40].sum() < 40
