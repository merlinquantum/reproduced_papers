"""Tests for the model + runner shapes (one forward pass each)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import runner  # noqa: E402
from lib.data import GraphRecord, LadderConcatDataset, collate_pad  # noqa: E402
from lib.model import GCN, GRIT, GRITAttentionLayer, GRITLite  # noqa: E402
from lib.pe_factory import pe_batch  # noqa: E402
from lib.runner import (  # noqa: E402
    _build_learning_rate_scheduler,
    _classification_metrics,
    _enforce_parameter_budget,
    _train_one_epoch,
)


def test_grit_lite_forward_classification():
    ds = LadderConcatDataset(per_class=2, length_range=(3, 3), seed=0)
    batch = collate_pad([ds[i] for i in range(len(ds))])
    K = 4
    PE = pe_batch(batch["A"], batch["mask"], "qirw2", K)
    model = GRITLite(edge_dim=K, node_dim=16, depth=1, num_heads=2, num_classes=2)
    out = model(PE, batch["mask"])
    assert out.shape == (len(ds), 2)


def test_grit_lite_forward_regression():
    ds = LadderConcatDataset(per_class=2, length_range=(3, 3), seed=0)
    batch = collate_pad([ds[i] for i in range(len(ds))])
    K = 4
    PE = pe_batch(batch["A"], batch["mask"], "rrwp", K)
    model = GRITLite(edge_dim=K, node_dim=16, depth=1, num_heads=2, head="graph_reg")
    out = model(PE, batch["mask"])
    assert out.shape == (len(ds),)


def test_gcn_baseline_forward():
    ds = LadderConcatDataset(per_class=2, length_range=(3, 3), seed=0)
    batch = collate_pad([ds[i] for i in range(len(ds))])
    model = GCN(node_in_dim=1, hidden_dim=8, num_classes=2, depth=2)
    x = batch["mask"].float().unsqueeze(-1)
    out = model(x, batch["A"], batch["mask"])
    assert out.shape == (len(ds), 2)


def test_grit_lite_backward():
    ds = LadderConcatDataset(per_class=2, length_range=(3, 3), seed=0)
    batch = collate_pad([ds[i] for i in range(len(ds))])
    K = 4
    PE = pe_batch(batch["A"], batch["mask"], "rrwp", K)
    model = GRITLite(edge_dim=K, node_dim=8, depth=1, num_heads=2, num_classes=2)
    out = model(PE, batch["mask"])
    loss = out.sum()
    loss.backward()
    # At least one parameter should have a non-zero gradient.
    grad_norms = [
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    ]
    assert any(g > 0 for g in grad_norms)


def test_full_grit_encodes_categorical_nodes_and_edges():
    record = GraphRecord(
        num_nodes=3,
        edges=[(0, 1), (1, 0), (1, 2), (2, 1)],
        label=0,
        node_features=torch.tensor([[0], [1], [2]]),
        edge_features=torch.tensor([[0], [1], [2], [3]]),
        directed=True,
        categorical_node_features=True,
        categorical_edge_features=True,
    )
    second_record = GraphRecord(
        num_nodes=3,
        edges=record.edges,
        label=1,
        node_features=torch.tensor([[2], [0], [0]]),
        edge_features=torch.tensor([[3], [2], [1], [0]]),
        directed=True,
        categorical_node_features=True,
        categorical_edge_features=True,
    )
    batch = collate_pad([record, second_record])
    positional_encoding = pe_batch(batch["A"], batch["mask"], "rrwp", 4)
    model = GRIT(
        edge_dim=4,
        node_dim=8,
        depth=1,
        num_heads=2,
        num_classes=2,
        node_feature_type="categorical",
        edge_feature_type="categorical",
        node_vocab_sizes=(3,),
        edge_vocab_sizes=(4,),
        node_in_dim=1,
        edge_in_dim=1,
        attention_dropout=0.25,
    )
    captured_inputs = {}
    model.node_encoder.register_forward_hook(
        lambda module, inputs, output: captured_inputs.update(node=inputs[0].clone())
    )
    model.edge_encoder.register_forward_hook(
        lambda module, inputs, output: captured_inputs.update(edge=inputs[0].clone())
    )
    logits = model(
        positional_encoding,
        batch["mask"],
        node_features=batch["node_features"],
        edge_features=batch["edge_features"],
        edge_mask=batch["edge_mask"],
    )

    assert logits.shape == (2, 2)
    assert torch.equal(captured_inputs["node"], batch["node_features"])
    assert torch.equal(captured_inputs["edge"], batch["edge_features"])
    assert model.node_encoder.embeddings[0].weight.requires_grad
    assert model.edge_encoder.embeddings[0].weight.requires_grad
    assert model.layers[0].attention_dropout.p == 0.25
    assert model.layers[0].dropout.p == 0.0


def test_full_grit_uses_configured_graph_pooling():
    model_sum = GRIT(edge_dim=2, node_dim=4, depth=0, pooling="sum")
    model_mean = GRIT(edge_dim=2, node_dim=4, depth=0, pooling="mean")
    model_mean.load_state_dict(model_sum.state_dict())
    model_sum.output_head = torch.nn.Identity()
    model_mean.output_head = torch.nn.Identity()
    positional_encoding = torch.zeros((1, 2, 2, 2))
    node_mask = torch.ones((1, 2), dtype=torch.bool)
    edge_mask = torch.zeros((1, 2, 2), dtype=torch.bool)
    node_features = torch.ones((1, 2, 1))

    sum_output = model_sum(
        positional_encoding, node_mask, node_features=node_features, edge_mask=edge_mask
    )
    mean_output = model_mean(
        positional_encoding, node_mask, node_features=node_features, edge_mask=edge_mask
    )

    assert torch.allclose(sum_output, 2 * mean_output)


def test_full_grit_node_head_keeps_node_axis():
    model = GRIT(
        edge_dim=3,
        node_dim=8,
        depth=1,
        num_heads=2,
        num_classes=6,
        head="node_class",
        attention_dropout=0.1,
    )
    model.eval()
    output = model(
        torch.zeros((2, 4, 4, 3)),
        torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool),
        node_features=torch.ones((2, 4, 1)),
        edge_mask=torch.zeros((2, 4, 4), dtype=torch.bool),
    )

    assert output.shape == (2, 4, 6)


def test_full_grit_backward_is_finite_with_padded_nodes():
    torch.manual_seed(0)
    model = GRIT(
        edge_dim=4,
        node_dim=16,
        depth=3,
        num_heads=4,
        head="graph_reg",
        attention_dropout=0.2,
    )
    positional_encoding = torch.rand((2, 5, 5, 4))
    node_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    edge_mask = torch.zeros((2, 5, 5), dtype=torch.bool)
    node_features = torch.ones((2, 5, 1))

    output = model(
        positional_encoding,
        node_mask,
        node_features=node_features,
        edge_mask=edge_mask,
    )
    output.sum().backward()

    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_full_grit_directed_pairs_aggregate_source_into_target():
    layer = GRITAttentionLayer(dim=1, num_heads=1, dropout=0.0, attention_dropout=0.0)
    layer.eval()
    with torch.no_grad():
        layer.query.weight.zero_()
        layer.query.bias.zero_()
        layer.key.weight.zero_()
        layer.value.weight.fill_(1.0)
        layer.pair_projection.weight.zero_()
        layer.pair_projection.weight[1, 0] = 1.0
        layer.pair_projection.bias.zero_()
        layer.attention_projection.fill_(1.0)
        layer.edge_value_projection.zero_()
        layer.node_output.weight.fill_(1.0)
        layer.node_output.bias.zero_()
        layer.degree_coefficients.zero_()
        layer.degree_coefficients[..., 0] = 1.0
        layer.pair_output.weight.zero_()
        layer.pair_output.bias.zero_()
        layer.node_feed_forward1.weight.zero_()
        layer.node_feed_forward1.bias.zero_()
        layer.node_feed_forward2.weight.zero_()
        layer.node_feed_forward2.bias.zero_()

    node_states = torch.tensor([[[2.0], [5.0]]])
    pair_states = torch.tensor([[[[0.0], [10.0]], [[0.0], [0.0]]]])
    output, _ = layer(
        node_states,
        pair_states,
        torch.ones((1, 2), dtype=torch.bool),
        torch.zeros((1, 2)),
    )

    assert output[0, 0, 0].item() == pytest.approx(5.5, rel=1e-4)
    expected_target_output = 7.0 + 3.0 / (torch.exp(torch.tensor(5.0)).item() + 1)
    assert output[0, 1, 0].item() == pytest.approx(expected_target_output, rel=1e-4)


def test_full_grit_signed_square_root_has_finite_gradient_at_zero():
    layer = GRITAttentionLayer(dim=4, num_heads=2, dropout=0.0, attention_dropout=0.0)
    layer.eval()
    node_states = torch.zeros((1, 2, 4), requires_grad=True)
    pair_states = torch.zeros((1, 2, 2, 4), requires_grad=True)

    output_nodes, output_pairs = layer(
        node_states,
        pair_states,
        torch.ones((1, 2), dtype=torch.bool),
        torch.zeros((1, 2)),
    )
    (output_nodes.sum() + output_pairs.sum()).backward()

    assert torch.isfinite(node_states.grad).all()
    assert torch.isfinite(pair_states.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )


def test_zinc_full_grit_matches_reported_parameter_count():
    model = GRIT(
        edge_dim=41,
        node_dim=64,
        depth=10,
        num_heads=8,
        head="graph_reg",
        pooling="sum",
        node_in_dim=1,
        edge_in_dim=1,
        node_feature_type="categorical",
        edge_feature_type="categorical",
        node_vocab_sizes=(21,),
        edge_vocab_sizes=(4,),
        attention_dropout=0.2,
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == 476_033
    assert _enforce_parameter_budget(model, 500_000) == 476_033
    with pytest.raises(ValueError, match="exceeding.*paper budget"):
        _enforce_parameter_budget(model, 476_032)


def test_pattern_weighted_accuracy_is_mean_class_recall():
    metrics = _classification_metrics(
        [torch.tensor([0, 0, 0, 1])],
        [torch.tensor([0, 0, 0, 0])],
        num_classes=2,
        report_weighted_accuracy=True,
    )

    assert metrics["acc"] == 0.75
    assert metrics["weighted_acc"] == 0.5


def test_learning_rate_scheduler_warms_up_separately():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = _build_learning_rate_scheduler(
        optimizer, warmup_epochs=5, total_epochs=20, minimum_lr=1e-6
    )

    observed_rates = []
    for _ in range(5):
        observed_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert observed_rates == pytest.approx([2e-4, 4e-4, 6e-4, 8e-4, 1e-3])


def test_grit_lite_node_classification_head_and_training_step():
    batch = collate_pad(
        [
            (3, [(0, 1), (1, 2)], [0, 1, 0]),
            (2, [(0, 1)], [1, 1]),
        ]
    )
    num_features = 3
    positional_encoding = pe_batch(batch["A"], batch["mask"], "rrwp", num_features)
    model = GRITLite(
        edge_dim=num_features,
        node_dim=8,
        depth=1,
        num_heads=2,
        num_classes=2,
        head="node_class",
    )

    logits = model(positional_encoding, batch["mask"])
    assert logits.shape == (2, 3, 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = _train_one_epoch(
        model,
        [batch],
        optimizer,
        torch.nn.CrossEntropyLoss(),
        "rrwp",
        num_features,
        "node_class",
        torch.device("cpu"),
    )

    assert metrics["loss"] > 0
    assert 0 <= metrics["acc"] <= 1


def test_node_classification_runner_writes_node_metrics(monkeypatch, tmp_path):
    records = [(3, [(0, 1), (1, 2)], [index % 2, 1, 0]) for index in range(10)]
    monkeypatch.setattr(
        runner,
        "_build_datasets",
        lambda config: (records[:8], records[8:9], records[9:]),
    )
    config = {
        "description": "node classification runner test",
        "feasibility": {"status": "supported"},
        "dataset": "pattern",
        "dataset_kwargs": {},
        "model": "grit_lite",
        "encoding": "rrwp",
        "pe_dim": 3,
        "node_dim": 8,
        "depth": 1,
        "num_heads": 2,
        "num_classes": 2,
        "head": "node_class",
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "train_frac": 0.8,
        "val_frac": 0.1,
        "device": "cpu",
        "seed": 0,
    }

    output = runner.train_and_evaluate(config, tmp_path)

    assert "weighted_acc" in output["test_metrics"]
    assert output["validation_metric"]["name"] == "weighted_acc"
    assert output["best_epoch"] == 1
    assert output["seed"] == 0
    assert output["num_params"] > 0
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "metrics.json").is_file()


def test_runner_selects_on_validation_and_evaluates_test_last(monkeypatch, tmp_path):
    train_records = [(2, [(0, 1)], 0) for _ in range(4)]
    validation_records = [(2, [(0, 1)], 1)]
    test_records = [(2, [(0, 1)], 0)]
    evaluation_labels = []

    monkeypatch.setattr(
        runner,
        "_build_datasets",
        lambda config: (train_records, validation_records, test_records),
    )

    def fake_train(model, loader, optimizer, *args, **kwargs):
        optimizer.step()
        return {"loss": 0.5, "acc": 0.5}

    monkeypatch.setattr(runner, "_train_one_epoch", fake_train)

    def fake_eval(model, loader, *args, **kwargs):
        label = loader.dataset[0][2]
        evaluation_labels.append(label)
        return {"loss": 0.2, "acc": 0.8 if label == 1 else 0.6}

    monkeypatch.setattr(runner, "_eval", fake_eval)
    config = {
        "description": "official split selection test",
        "feasibility": {"status": "supported"},
        "dataset": "mnist",
        "dataset_kwargs": {},
        "model": "grit_lite",
        "encoding": "rrwp",
        "pe_dim": 2,
        "node_dim": 4,
        "depth": 1,
        "num_heads": 1,
        "num_classes": 2,
        "head": "graph_class",
        "epochs": 2,
        "batch_size": 2,
        "lr": 1e-3,
        "train_frac": 0.8,
        "val_frac": 0.1,
        "device": "cpu",
        "seed": 0,
    }

    output = runner.train_and_evaluate(config, tmp_path)

    assert evaluation_labels == [1, 1, 0]
    assert output["test_metrics"]["acc"] == 0.6
