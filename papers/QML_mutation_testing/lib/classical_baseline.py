"""Fair classical baselines for the original model's quality.

The paper makes no accuracy claim, so these baselines are context rather than
a test of a paper claim: they say whether the photonic QNN whose faults are
being injected is a reasonable model of Iris at all. They are matched on
parameter count, which is the axis the photonic adaptation controls
(24 trainable phases), and they see exactly the same splits and scaling.
"""

from __future__ import annotations

import logging

import torch
from lib.data import DatasetSplits
from torch import nn

logger = logging.getLogger(__name__)


def linear_parameter_count(n_features: int, n_classes: int) -> int:
    """Return the parameter count of a softmax linear classifier."""

    return n_features * n_classes + n_classes


def mlp_parameter_count(n_features: int, hidden: int, n_classes: int) -> int:
    """Return the parameter count of a one-hidden-layer MLP."""

    return (n_features + 1) * hidden + (hidden + 1) * n_classes


def _train_torch_classifier(model, splits, epochs, lr, seed):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x = splits.x_train
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        outputs = model(x)
        loss = loss_fn(outputs, splits.y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_test = splits.x_test
        if x_test.dim() > 2:
            x_test = x_test.view(x_test.size(0), -1)

        predictions = model(x_test).argmax(dim=1)
        test_acc = (predictions == splits.y_test).float().mean().item()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "test_accuracy": test_acc,
        "n_parameters": n_params,
    }


def run_classical_baselines(
    splits: DatasetSplits,
    *,
    seeds: list[int],
    epochs: int = 200,
    lr: float = 0.05,
) -> dict:
    """Train the classical reference models over several seeds.

    Parameters
    ----------
    splits : IrisSplits
        Prepared data splits.
    seeds : list[int]
        Initialisation seeds.
    epochs : int
        Full-batch steps, matched to the QNN. Default value is 200.
    lr : float
        Adam learning rate, matched to the QNN. Default value is 0.05.

    Returns
    -------
    dict
        Baseline name -> per-seed metrics and their mean/std.
    """

    n_features = splits.x_train.view(splits.x_train.size(0), -1).size(1)
    n_classes = int(splits.y_train.max().item()) + 1

    factories = {
        "linear_softmax": lambda: nn.Linear(n_features, n_classes),
        "mlp_hidden3": lambda: nn.Sequential(
            nn.Linear(n_features, 3), nn.Tanh(), nn.Linear(3, n_classes)
        ),
    }

    results: dict[str, dict] = {}
    for name, factory in factories.items():
        per_seed = [
            _train_torch_classifier(factory(), splits, epochs=epochs, lr=lr, seed=seed)
            for seed in seeds
        ]
        test_accuracies = [row["test_accuracy"] for row in per_seed]
        results[name] = {
            "per_seed": per_seed,
            "n_parameters": per_seed[0]["n_parameters"],
            "test_accuracy_mean": sum(test_accuracies) / len(test_accuracies),
            "test_accuracy_min": min(test_accuracies),
            "test_accuracy_max": max(test_accuracies),
        }
        logger.info(
            "CLASSICAL_BASELINE_COMPLETED | model=%s | params=%d | test_acc_mean=%.4f",
            name,
            results[name]["n_parameters"],
            results[name]["test_accuracy_mean"],
        )

    majority = torch.bincount(splits.y_test).max().item() / len(splits.y_test)
    results["majority_class"] = {
        "n_parameters": 0,
        "test_accuracy_mean": float(majority),
        "test_accuracy_min": float(majority),
        "test_accuracy_max": float(majority),
        "per_seed": [],
    }
    return results
