"""Train + evaluate one experiment from a JSON config.

Public entry: ``train_and_evaluate(cfg: dict, run_dir: str) -> dict``.

The runner is intentionally compact:

- Choose a dataset (synthetic ladder concat, random graph regression, or
  SRG pair).
- Compute positional encodings per-graph (precomputed once, kept on CPU).
- Build a full GRIT, GRITLite, or GCN model with the right head.
- Train, log per-epoch metrics, save a final ``metrics.json``.

Configs are JSON; see ``configs/`` for examples.
"""

from __future__ import annotations

import copy
import json
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import (
    LadderConcatDataset,
    PyGBenchmarkAdapter,
    RandomGraphRegression,
    collate_pad,
    srg_pairs,
)
from .model import GCN, GRIT, GRITLite, SparseGatedGCN
from .pe_factory import pe_batch
from .srg import run_srg_paper_experiment

_DATASET_ARGUMENTS = {
    "ladder_concat": {
        "per_class",
        "length_range",
        "seed",
        "node_encoding",
        "pe_dim",
        "crossing_range",
        "cache_path",
    },
    "graph_reg": {"num_graphs", "n_range", "p", "seed"},
    "srg_pair": {"n_per_class"},
    "zinc": {"split", "limit", "subset"},
    "mnist": {"split", "limit", "subset"},
    "cifar10": {"split", "limit", "subset"},
    "pattern": {"split", "limit", "subset"},
    "cluster": {"split", "limit", "subset"},
}
_KNOWN_DATASET_ARGUMENTS = set().union(*_DATASET_ARGUMENTS.values())
_SUPPORTED_FEASIBILITY_STATUSES = {"supported", "resource_intensive", "infeasible"}


def _resolve_positional_encoding_dimensions(
    cfg: dict,
) -> tuple[int, int | None, int | None]:
    """Resolve total, RRWP, and quantum positional-encoding dimensions."""
    encoding = cfg["encoding"]
    if encoding == "rrwp":
        dimension = cfg.get("rrwp_dim", cfg.get("pe_dim"))
        rrwp_dim, qpe_dim = dimension, None
    elif encoding in {"cqrw1", "qirw2"}:
        dimension = cfg.get("qpe_dim", cfg.get("pe_dim"))
        rrwp_dim, qpe_dim = None, dimension
    elif encoding in {"rrwp+cqrw1", "rrwp+qirw2"}:
        rrwp_dim = cfg.get("rrwp_dim")
        qpe_dim = cfg.get("qpe_dim")
        if rrwp_dim is None or qpe_dim is None:
            raise ValueError(f"{encoding} requires rrwp_dim and qpe_dim")
        dimension = rrwp_dim + qpe_dim
    elif encoding in {"ground_state_corr", "laplacian", "rrwp_node", "rrwp_edge"}:
        dimension = cfg.get("pe_dim")
        rrwp_dim, qpe_dim = None, None
    elif encoding == "none":
        dimension = 1
        rrwp_dim, qpe_dim = None, None
    else:
        raise ValueError(f"unknown encoding: {encoding}")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError(f"{encoding} requires a positive encoding dimension")
    return dimension, rrwp_dim, qpe_dim


def _resolve_cqrw_times(cfg: dict, qpe_dim: int | None) -> list[float] | None:
    """Resolve deterministic Appendix C random evolution times."""
    if "cqrw1" not in cfg["encoding"]:
        return None
    if qpe_dim is None:
        raise ValueError("cqrw1 requires qpe_dim")
    configured_times = cfg.get("times")
    if configured_times is not None:
        if len(configured_times) != qpe_dim:
            raise ValueError("cqrw1 requires exactly qpe_dim evolution times")
        return [float(value) for value in configured_times]
    minimum_time = cfg.get("qpe_min_time")
    maximum_time = cfg.get("qpe_max_time")
    if minimum_time is None or maximum_time is None:
        raise ValueError("paper cqrw1 configs require qpe_min_time and qpe_max_time")
    if not 0 <= minimum_time < maximum_time:
        raise ValueError("qpe_min_time must be non-negative and below qpe_max_time")
    random_generator = np.random.default_rng(cfg.get("seed", 0))
    return random_generator.uniform(minimum_time, maximum_time, qpe_dim).tolist()


def _validate_config(cfg: dict) -> None:
    required_keys = {"dataset", "model", "encoding"}
    missing_keys = required_keys - set(cfg)
    if missing_keys:
        missing_names = ", ".join(sorted(missing_keys))
        raise ValueError(f"missing required config entries: {missing_names}")

    feasibility = cfg.get("feasibility", {"status": "supported"})
    feasibility_status = feasibility.get("status")
    if feasibility_status not in _SUPPORTED_FEASIBILITY_STATUSES:
        raise ValueError(f"unknown feasibility status: {feasibility_status}")
    if feasibility_status == "infeasible":
        reason = feasibility.get("reason")
        if not reason:
            raise ValueError("infeasible configs must provide a reason")
        raise ValueError(f"experiment config is infeasible: {reason}")

    dataset_name = cfg["dataset"]
    if cfg["model"] not in {"grit", "grit_lite", "gcn", "gated_gcn"}:
        raise ValueError(f"unknown model: {cfg['model']}")
    _, _, qpe_dim = _resolve_positional_encoding_dimensions(cfg)
    if "cqrw1" in cfg["encoding"]:
        _resolve_cqrw_times(cfg, qpe_dim)
    expected_initial_distribution = {
        "rrwp+cqrw1": "local",
        "rrwp+qirw2": "adjacency",
    }.get(cfg["encoding"])
    if (
        expected_initial_distribution is not None
        and cfg.get("qpe_initial_distribution") != expected_initial_distribution
    ):
        raise ValueError(
            f"{cfg['encoding']} requires "
            f"qpe_initial_distribution='{expected_initial_distribution}'"
        )
    if "seeds" in cfg:
        seeds = cfg["seeds"]
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(not isinstance(seed, int) for seed in seeds)
        ):
            raise ValueError("seeds must be a non-empty list of integers")
        if len(seeds) != len(set(seeds)):
            raise ValueError("seeds must not contain duplicates")
    if dataset_name in {"zinc", "mnist", "cifar10", "pattern", "cluster"}:
        if "split" in cfg.get("dataset_kwargs", {}):
            raise ValueError(
                "benchmark configs must not select one split; the runner loads "
                "the official train, val, and test splits"
            )
    head_type = cfg.get("head", "graph_class")
    if dataset_name in {"pattern", "cluster"} and head_type != "node_class":
        raise ValueError(f"{dataset_name} requires head='node_class'")
    if head_type == "node_class" and dataset_name not in {"pattern", "cluster"}:
        raise ValueError("head='node_class' requires the pattern or cluster dataset")
    if cfg["model"] == "gcn" and head_type != "graph_class":
        raise ValueError("the GCN baseline supports only head='graph_class'")
    if cfg["model"] == "grit":
        if "attention_dropout" not in cfg:
            raise ValueError("full GRIT configs require attention_dropout")
        if "pooling" not in cfg:
            raise ValueError("full GRIT configs require pooling")
        required_pooling = {"zinc": "sum", "mnist": "mean", "cifar10": "mean"}
        if (
            dataset_name in required_pooling
            and cfg["pooling"] != required_pooling[dataset_name]
        ):
            raise ValueError(
                f"{dataset_name} paper experiments require "
                f"pooling='{required_pooling[dataset_name]}'"
            )
        parameter_budget = cfg.get("parameter_budget")
        if not isinstance(parameter_budget, int) or parameter_budget <= 0:
            raise ValueError("full GRIT configs require a positive parameter_budget")
        warmup_epochs = cfg.get("warmup_epochs")
        if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
            raise ValueError("full GRIT configs require non-negative warmup_epochs")
        if warmup_epochs > cfg.get("epochs", 5):
            raise ValueError("warmup_epochs cannot exceed epochs")
        attention_dropout = cfg["attention_dropout"]
        if not 0 <= attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        minimum_lr = cfg.get("minimum_lr", 1e-6)
        if not 0 <= minimum_lr <= cfg.get("lr", 1e-3):
            raise ValueError("minimum_lr must be between zero and lr")

    train_fraction = cfg.get("train_frac", 0.8)
    validation_fraction = cfg.get("val_frac", 0.1)
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train_frac and val_frac must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_frac + val_frac must be less than 1")

    if dataset_name == "ladder_concat" and cfg["model"] in {"gcn", "gated_gcn"}:
        dataset_arguments = cfg.get("dataset_kwargs", {})
        expected_node_encoding = {
            "ground_state_corr": "quantum",
            "laplacian": "laplacian",
            "rrwp_node": "rrwp",
            "rrwp_edge": "rrwp",
            "none": "none",
        }.get(cfg["encoding"])
        if expected_node_encoding is None:
            raise ValueError(
                "synthetic GCN configs require ground_state_corr, laplacian, "
                "rrwp_node, or none"
            )
        if dataset_arguments.get("node_encoding") != expected_node_encoding:
            raise ValueError(
                f"encoding={cfg['encoding']} requires dataset_kwargs.node_encoding="
                f"'{expected_node_encoding}'"
            )
        if expected_node_encoding != "none" and dataset_arguments.get("pe_dim") != 20:
            raise ValueError("paper synthetic positional encodings require pe_dim=20")
        if "split_seed" not in cfg:
            raise ValueError("synthetic configs require a fixed split_seed")


def _dataset_arguments(cfg: dict, dataset_name: str) -> dict:
    configured_arguments = cfg.get("dataset_kwargs", {})
    unknown_arguments = set(configured_arguments) - _KNOWN_DATASET_ARGUMENTS
    if unknown_arguments:
        unknown_names = ", ".join(sorted(unknown_arguments))
        raise ValueError(f"unknown dataset_kwargs entries: {unknown_names}")
    return {
        name: configured_arguments[name]
        for name in _DATASET_ARGUMENTS[dataset_name]
        if name in configured_arguments
    }


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _build_dataset(cfg: dict):
    name = cfg["dataset"]
    if name == "ladder_concat":
        params = _dataset_arguments(cfg, name)
        return LadderConcatDataset(**params)
    if name == "graph_reg":
        params = _dataset_arguments(cfg, name)
        return RandomGraphRegression(**params)
    if name in ("zinc", "mnist", "cifar10", "pattern", "cluster"):
        params = _dataset_arguments(cfg, name)
        split = params.pop("split", "train")
        return PyGBenchmarkAdapter(
            name=name,
            data_root=cfg["data_root"],
            split=split,
            **params,
        )
    if name == "srg_pair":
        idx = cfg.get("srg_index", 0)
        pair = srg_pairs()[idx]
        # Treat as a binary classification problem on the two SRGs.
        records = []
        for g in (pair.g1, pair.g2):
            n = g.number_of_nodes()
            edges = list(g.edges())
            records.append((n, edges, 0))
        records += [
            (g.number_of_nodes(), list(g.edges()), 1) for g in [pair.g2, pair.g1]
        ]
        # Inflate with random permutations of node labels.

        rng = np.random.default_rng(cfg.get("seed", 0))
        inflated = []
        dataset_arguments = _dataset_arguments(cfg, name)
        n_per_class = dataset_arguments.get("n_per_class", 32)
        for label, g in enumerate([pair.g1, pair.g2]):
            n = g.number_of_nodes()
            base_edges = list(g.edges())
            for _ in range(n_per_class):
                perm = rng.permutation(n)
                edges = [(int(perm[u]), int(perm[v])) for u, v in base_edges]
                inflated.append((n, edges, label))
        rng.shuffle(inflated)

        class _ListDS:
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                return self.items[i]

        return _ListDS(inflated)
    raise ValueError(f"unknown dataset: {name}")


def _build_datasets(cfg: dict):
    """Build train, validation, and test datasets for one experiment.

    Parameters
    ----------
    cfg : dict
        Resolved experiment configuration.

    Returns
    -------
    tuple
        Train, validation, and test datasets. PyG benchmarks retain their
        official splits; generated datasets receive a deterministic random
        split.
    """
    dataset_name = cfg["dataset"]
    benchmark_names = {"zinc", "mnist", "cifar10", "pattern", "cluster"}
    if dataset_name in benchmark_names:
        configured_arguments = _dataset_arguments(cfg, dataset_name)
        if "split" in configured_arguments:
            raise ValueError(
                "benchmark configs must not select one split; the runner loads "
                "the official train, val, and test splits"
            )
        return tuple(
            PyGBenchmarkAdapter(
                name=dataset_name,
                data_root=cfg["data_root"],
                split=split,
                **configured_arguments,
            )
            for split in ("train", "val", "test")
        )

    dataset = _build_dataset(cfg)
    num_records = len(dataset)
    num_train_records = int(num_records * cfg.get("train_frac", 0.8))
    num_validation_records = int(num_records * cfg.get("val_frac", 0.1))
    indices = np.arange(num_records)
    random_generator = np.random.default_rng(cfg.get("split_seed", cfg.get("seed", 0)))
    random_generator.shuffle(indices)
    train_indices = indices[:num_train_records]
    validation_indices = indices[
        num_train_records : num_train_records + num_validation_records
    ]
    test_indices = indices[num_train_records + num_validation_records :]
    return (
        [dataset[index] for index in train_indices],
        [dataset[index] for index in validation_indices],
        [dataset[index] for index in test_indices],
    )


def _build_model(
    cfg: dict, edge_dim: int, feature_schema: dict | None = None
) -> nn.Module:
    arch = cfg["model"]
    feature_schema = feature_schema or {
        "node_feature_dim": cfg.get("node_in_dim", 1),
        "edge_feature_dim": 0,
        "node_feature_type": "continuous",
        "edge_feature_type": "continuous",
        "node_vocab_sizes": (),
        "edge_vocab_sizes": (),
    }
    if arch == "grit_lite":
        return GRITLite(
            edge_dim=edge_dim,
            node_dim=cfg.get("node_dim", 64),
            depth=cfg.get("depth", 2),
            num_heads=cfg.get("num_heads", 4),
            num_classes=cfg.get("num_classes", 2),
            head=cfg.get("head", "graph_class"),
            node_in_dim=feature_schema["node_feature_dim"],
            edge_in_dim=feature_schema["edge_feature_dim"],
            node_feature_type=feature_schema["node_feature_type"],
            edge_feature_type=feature_schema["edge_feature_type"],
            node_vocab_sizes=tuple(feature_schema["node_vocab_sizes"]),
            edge_vocab_sizes=tuple(feature_schema["edge_vocab_sizes"]),
            dropout=cfg.get("dropout", 0.0),
        )
    if arch == "grit":
        return GRIT(
            edge_dim=edge_dim,
            node_dim=cfg.get("node_dim", 64),
            depth=cfg.get("depth", 10),
            num_heads=cfg.get("num_heads", 8),
            num_classes=cfg.get("num_classes", 2),
            head=cfg.get("head", "graph_class"),
            pooling=cfg["pooling"],
            node_in_dim=feature_schema["node_feature_dim"],
            edge_in_dim=feature_schema["edge_feature_dim"],
            node_feature_type=feature_schema["node_feature_type"],
            edge_feature_type=feature_schema["edge_feature_type"],
            node_vocab_sizes=tuple(feature_schema["node_vocab_sizes"]),
            edge_vocab_sizes=tuple(feature_schema["edge_vocab_sizes"]),
            dropout=cfg.get("dropout", 0.0),
            attention_dropout=cfg["attention_dropout"],
        )
    if arch == "gcn":
        return GCN(
            node_in_dim=feature_schema["node_feature_dim"],
            hidden_dim=cfg.get("hidden_dim", 32),
            num_classes=cfg.get("num_classes", 2),
            depth=cfg.get("depth", 2),
        )
    if arch == "gated_gcn":
        return SparseGatedGCN(
            node_in_dim=cfg.get("node_in_dim", 1),
            edge_in_dim=cfg.get("edge_in_dim", edge_dim),
            hidden_dim=cfg.get("hidden_dim", 32),
            num_classes=cfg.get("num_classes", 2),
            depth=cfg.get("depth", 5),
        )
    raise ValueError(f"unknown model: {arch}")


def _enforce_parameter_budget(model: nn.Module, parameter_budget: int | None) -> int:
    """Return the trainable size and reject models above the paper budget."""
    num_parameters = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
    if parameter_budget is not None and num_parameters > parameter_budget:
        raise ValueError(
            f"model has {num_parameters:,} parameters, exceeding the configured "
            f"paper budget of {parameter_budget:,}"
        )
    return num_parameters


def _classification_metrics(
    labels: list[torch.Tensor],
    predictions: list[torch.Tensor],
    num_classes: int,
    report_weighted_accuracy: bool,
) -> dict[str, float]:
    """Compute ordinary and SBM class-weighted accuracy."""
    all_labels = torch.cat(labels)
    all_predictions = torch.cat(predictions)
    metrics = {"acc": float((all_predictions == all_labels).float().mean().item())}
    if report_weighted_accuracy:
        class_recalls = []
        for class_index in range(num_classes):
            class_entries = all_labels == class_index
            if class_entries.any():
                class_recalls.append(
                    (all_predictions[class_entries] == class_index).float().mean()
                )
            else:
                class_recalls.append(torch.tensor(0.0))
        metrics["weighted_acc"] = float(torch.stack(class_recalls).mean().item())
    return metrics


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    use_weighted_loss: bool,
) -> torch.Tensor:
    """Compute the paper's batch-weighted PATTERN cross entropy when needed."""
    if not use_weighted_loss:
        return criterion(logits, labels)
    class_counts = torch.bincount(labels, minlength=logits.shape[-1])
    class_weights = (labels.numel() - class_counts).float() / labels.numel()
    class_weights = class_weights * class_counts.gt(0)
    return nn.functional.cross_entropy(logits, labels, weight=class_weights)


def _train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    encoding,
    K,
    head_type,
    device,
    times=None,
    report_weighted_accuracy=False,
    rrwp_dim=None,
    qpe_dim=None,
    randomize_node_feature_signs=False,
):
    model.train()
    losses = []
    classification_labels: list[torch.Tensor] = []
    classification_predictions: list[torch.Tensor] = []
    num_classes = 0
    abs_errs = []
    for batch_index, batch in enumerate(loader):
        A = None if batch["A"] is None else batch["A"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)
        node_features = batch["node_features"].to(device)
        if randomize_node_feature_signs:
            feature_signs = (
                2
                * torch.randint(
                    0,
                    2,
                    (node_features.shape[0], 1, node_features.shape[2]),
                    device=device,
                )
                - 1
            )
            node_features = node_features * feature_signs
        edge_features = (
            None
            if batch["edge_features"] is None
            else batch["edge_features"].to(device)
        )
        edge_mask = (
            None if batch["edge_mask"] is None else batch["edge_mask"].to(device)
        )
        if isinstance(model, GCN):
            logits = model(
                node_features.float(),
                A,
                mask,
                edge_index=batch["edge_index"].to(device),
            )
        elif isinstance(model, SparseGatedGCN):
            logits = model(
                node_features.float(),
                mask,
                batch["edge_index"].to(device),
                batch["sparse_edge_features"].to(device).float(),
            )
        else:
            PE = pe_batch(
                batch["A"],
                batch["mask"],
                encoding,
                K,
                times=times,
                rrwp_dim=rrwp_dim,
                qpe_dim=qpe_dim,
            ).to(device)
            logits = model(
                PE,
                mask,
                node_features=node_features,
                edge_features=edge_features,
                edge_mask=edge_mask,
            )
        if head_type == "graph_class":
            classification_targets = labels.long()
            loss = _classification_loss(
                logits, classification_targets, criterion, report_weighted_accuracy
            )
            preds = logits.argmax(dim=-1)
        elif head_type == "node_class":
            valid_nodes = mask & labels.ne(-1)
            valid_logits = logits[valid_nodes]
            valid_labels = labels[valid_nodes].long()
            loss = _classification_loss(
                valid_logits, valid_labels, criterion, report_weighted_accuracy
            )
            preds = valid_logits.argmax(dim=-1)
            classification_targets = valid_labels
        else:
            loss = criterion(logits, labels.float())
            abs_errs.extend((logits.detach() - labels.float()).abs().tolist())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss in batch {batch_index}")
        optimizer.zero_grad()
        loss.backward()
        nonfinite_gradient = next(
            (
                parameter_name
                for parameter_name, parameter in model.named_parameters()
                if parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ),
            None,
        )
        if nonfinite_gradient is not None:
            raise FloatingPointError(
                f"non-finite gradient for {nonfinite_gradient} "
                f"in training batch {batch_index}"
            )
        optimizer.step()
        losses.append(loss.item())
        if head_type in {"graph_class", "node_class"}:
            classification_labels.append(classification_targets.detach().cpu())
            classification_predictions.append(preds.detach().cpu())
            num_classes = logits.shape[-1]
    metrics = {"loss": float(np.mean(losses))}
    if head_type in {"graph_class", "node_class"}:
        metrics.update(
            _classification_metrics(
                classification_labels,
                classification_predictions,
                num_classes,
                report_weighted_accuracy,
            )
        )
    else:
        metrics["mae"] = float(np.mean(abs_errs))
    return metrics


@torch.no_grad()
def _eval(
    model,
    loader,
    criterion,
    encoding,
    K,
    head_type,
    device,
    times=None,
    report_weighted_accuracy=False,
    rrwp_dim=None,
    qpe_dim=None,
):
    model.eval()
    losses, abs_errs = [], []
    classification_labels: list[torch.Tensor] = []
    classification_predictions: list[torch.Tensor] = []
    num_classes = 0
    for batch_index, batch in enumerate(loader):
        A = None if batch["A"] is None else batch["A"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)
        node_features = batch["node_features"].to(device)
        edge_features = (
            None
            if batch["edge_features"] is None
            else batch["edge_features"].to(device)
        )
        edge_mask = (
            None if batch["edge_mask"] is None else batch["edge_mask"].to(device)
        )
        if isinstance(model, GCN):
            logits = model(
                node_features.float(),
                A,
                mask,
                edge_index=batch["edge_index"].to(device),
            )
        elif isinstance(model, SparseGatedGCN):
            logits = model(
                node_features.float(),
                mask,
                batch["edge_index"].to(device),
                batch["sparse_edge_features"].to(device).float(),
            )
        else:
            PE = pe_batch(
                batch["A"],
                batch["mask"],
                encoding,
                K,
                times=times,
                rrwp_dim=rrwp_dim,
                qpe_dim=qpe_dim,
            ).to(device)
            logits = model(
                PE,
                mask,
                node_features=node_features,
                edge_features=edge_features,
                edge_mask=edge_mask,
            )
        if head_type == "graph_class":
            classification_targets = labels.long()
            loss = _classification_loss(
                logits, classification_targets, criterion, report_weighted_accuracy
            )
            preds = logits.argmax(dim=-1)
        elif head_type == "node_class":
            valid_nodes = mask & labels.ne(-1)
            valid_logits = logits[valid_nodes]
            valid_labels = labels[valid_nodes].long()
            loss = _classification_loss(
                valid_logits, valid_labels, criterion, report_weighted_accuracy
            )
            preds = valid_logits.argmax(dim=-1)
            classification_targets = valid_labels
        else:
            loss = criterion(logits, labels.float())
            abs_errs.extend((logits - labels.float()).abs().tolist())
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite evaluation loss in batch {batch_index}"
            )
        losses.append(loss.item())
        if head_type in {"graph_class", "node_class"}:
            classification_labels.append(classification_targets.detach().cpu())
            classification_predictions.append(preds.detach().cpu())
            num_classes = logits.shape[-1]
    metrics = {"loss": float(np.mean(losses))}
    if head_type in {"graph_class", "node_class"}:
        metrics.update(
            _classification_metrics(
                classification_labels,
                classification_predictions,
                num_classes,
                report_weighted_accuracy,
            )
        )
    else:
        metrics["mae"] = float(np.mean(abs_errs))
    return metrics


def _build_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    minimum_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build the linear-warmup cosine scheduler used by GRIT."""
    base_lr = optimizer.param_groups[0]["lr"]
    minimum_factor = minimum_lr / base_lr

    def learning_rate_factor(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        decay_epochs = max(1, total_epochs - warmup_epochs)
        decay_progress = min(1.0, (epoch_index - warmup_epochs) / decay_epochs)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)


def train_and_evaluate(cfg: dict, run_dir: str) -> dict:
    if cfg.get("experiment") == "srg_paper":
        return run_srg_paper_experiment(cfg, run_dir)
    _validate_config(cfg)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(cfg, f, indent=2)
    log_lines: list[str] = []

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    _seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cpu"))

    log(f"building dataset: {cfg['dataset']}")
    train_ds, val_ds, test_ds = _build_datasets(cfg)
    bs = cfg.get("batch_size", 16)
    collate_function = (
        partial(collate_pad, include_dense_graph=False)
        if cfg["model"] in {"gcn", "gated_gcn"}
        else collate_pad
    )
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, collate_fn=collate_function
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, collate_fn=collate_function
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, collate_fn=collate_function
    )
    log(f"split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    encoding = cfg["encoding"]
    K, rrwp_dim, qpe_dim = _resolve_positional_encoding_dimensions(cfg)
    times = _resolve_cqrw_times(cfg, qpe_dim)
    edge_dim = K

    feature_schema = getattr(train_ds, "feature_schema", None)
    if feature_schema is not None:
        for split_name, dataset in (("val", val_ds), ("test", test_ds)):
            if dataset.feature_schema != feature_schema:
                raise ValueError(
                    f"{split_name} feature schema differs from the train split"
                )
    model = _build_model(cfg, edge_dim=edge_dim, feature_schema=feature_schema)
    model.to(device)
    num_parameters = _enforce_parameter_budget(model, cfg.get("parameter_budget"))
    log(f"model: {cfg['model']} | num_params={num_parameters}")

    head_type = cfg.get("head", "graph_class")
    if head_type in {"graph_class", "node_class"}:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.L1Loss()
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }.get(cfg.get("optimizer", "adamw"))
    if optimizer_class is None:
        raise ValueError(f"unknown optimizer: {cfg['optimizer']}")
    optimizer = optimizer_class(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    epochs = cfg.get("epochs", 5)
    scheduler = _build_learning_rate_scheduler(
        optimizer,
        warmup_epochs=cfg.get("warmup_epochs", 0),
        total_epochs=epochs,
        minimum_lr=cfg.get("minimum_lr", 1e-6),
    )
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_mae": [],
        "val_mae": [],
        "learning_rate": [],
    }
    classification_task = head_type in {"graph_class", "node_class"}
    report_weighted_accuracy = cfg["dataset"] == "pattern"
    selection_metric_name = (
        "weighted_acc"
        if report_weighted_accuracy
        else "acc"
        if classification_task
        else "mae"
    )
    best_val = -float("inf") if classification_task else float("inf")
    best_state = None
    best_epoch = None
    best_validation_metrics = None
    t0 = time.time()
    for epoch in range(epochs):
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        tr = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            encoding,
            K,
            head_type,
            device,
            times=times,
            report_weighted_accuracy=report_weighted_accuracy,
            rrwp_dim=rrwp_dim,
            qpe_dim=qpe_dim,
            randomize_node_feature_signs=cfg.get(
                "node_feature_sign_augmentation", False
            ),
        )
        va = _eval(
            model,
            val_loader,
            criterion,
            encoding,
            K,
            head_type,
            device,
            times=times,
            report_weighted_accuracy=report_weighted_accuracy,
            rrwp_dim=rrwp_dim,
            qpe_dim=qpe_dim,
        )
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        if classification_task:
            history["train_acc"].append(tr["acc"])
            history["val_acc"].append(va["acc"])
            if report_weighted_accuracy:
                history.setdefault("train_weighted_acc", []).append(tr["weighted_acc"])
                history.setdefault("val_weighted_acc", []).append(va["weighted_acc"])
            metric_str = (
                f"train_{selection_metric_name}={tr[selection_metric_name]:.3f} "
                f"val_{selection_metric_name}={va[selection_metric_name]:.3f}"
            )
            cur_val = va[selection_metric_name]
            improved = cur_val > best_val
        else:
            history["train_mae"].append(tr["mae"])
            history["val_mae"].append(va["mae"])
            metric_str = f"train_mae={tr['mae']:.4f} val_mae={va['mae']:.4f}"
            cur_val = va["mae"]
            improved = cur_val < best_val
        log(
            f"epoch {epoch + 1}/{epochs} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} {metric_str}"
        )
        if improved:
            best_val = cur_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            best_validation_metrics = copy.deepcopy(va)
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_model.pt")
    test_metrics = _eval(
        model,
        test_loader,
        criterion,
        encoding,
        K,
        head_type,
        device,
        times=times,
        report_weighted_accuracy=report_weighted_accuracy,
        rrwp_dim=rrwp_dim,
        qpe_dim=qpe_dim,
    )
    wall_clock = time.time() - t0
    out = {
        "history": history,
        "best_epoch": best_epoch,
        "validation_metric": {
            "name": selection_metric_name,
            "value": best_val,
        },
        "best_validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "test_metric": {
            "name": selection_metric_name,
            "value": test_metrics[selection_metric_name],
        },
        "seed": int(cfg.get("seed", 0)),
        "wall_clock_s": wall_clock,
        "num_params": num_parameters,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(run_dir / "run.log", "w") as f:
        f.write("\n".join(log_lines))
    return out
