"""Train + evaluate one experiment from a JSON config.

Public entry: ``train_and_evaluate(cfg: dict, run_dir: str) -> dict``.

The runner is intentionally compact:

- Choose a dataset (synthetic ladder concat, random graph regression, or
  SRG pair).
- Compute positional encodings per-graph (precomputed once, kept on CPU).
- Build a full GRIT, GRITLite, or GCN model with the right head.
- Train, log per-epoch metrics, save a final ``metrics.json``.

Configs are JSON; see ``configs/`` for examples.

Run artifacts (evidence contract, paths relative to the run directory):

- ``run.log``: written by the shared runtime's logging handler; this module
  logs lifecycle events (``RUN_STARTED``, ``TRAIN_EPOCH_COMPLETED``,
  ``EVALUATION_COMPLETED``, ``RUN_COMPLETED`` / ``RUN_FAILED``) to it.
- ``run_status.json``: run identity, dataset, selection rule, Git state,
  evidence paths and terminal status.
- ``config_snapshot.json``: the resolved configuration.
- ``metrics.json`` (``srg_metrics.json`` for the SRG experiment).
- ``test_outputs.npz``: per-graph test outputs for paired statistical tests
  (absolute errors for regression; labels and predictions for classification).
- ``checkpoint.pt``: resumable training state, rewritten every
  ``checkpoint_every`` epochs; ``best_model.pt``: best-validation weights.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import (
    LadderConcatDataset,
    PyGBenchmarkAdapter,
    RandomGraphRegression,
    build_dataloaders,
    srg_pairs,
)
from .experiment_logging import git_state, sha256, utc_now, write_json
from .model import GCN, GRIT, GRITLite, SparseGatedGCN
from .pe_factory import pe_batch
from .photonic import photonic_walk_covariance
from .srg import run_srg_paper_experiment, run_srg_photonic_experiment

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1
PAPER_NAME = "QPE_GNN"
_SRG_EXPERIMENTS = {
    "srg_paper": run_srg_paper_experiment,
    "srg_photonic": run_srg_photonic_experiment,
}
# Encodings whose quantum channels are indexed by evolution times drawn as in
# Appendix C (the photonic encoding reuses the 1-CQRW times of the same seed).
_TIMED_ENCODINGS = {"cqrw1", "rrwp+cqrw1", "rrwp+photon2"}
# Each run of the photonic encoding recomputes its first training graphs with
# MerLin and fails if the closed-form features deviate by more than this.
_PHOTONIC_CHECK_GRAPHS = 64
_PHOTONIC_CHECK_TOLERANCE = 1e-8
REPO_ROOT = Path(__file__).resolve().parents[3]
# Training state keys that must agree between a checkpoint and the run that
# resumes it; any difference would make the resumed run a different experiment.
_RESUME_IDENTITY_KEYS = (
    "dataset",
    "dataset_kwargs",
    "model",
    "encoding",
    "seed",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "warmup_epochs",
    "minimum_lr",
    "rrwp_dim",
    "qpe_dim",
    "node_dim",
    "depth",
)

_DATASET_ARGUMENTS = {
    "ladder_concat": {
        "per_class",
        "length_range",
        "seed",
        "node_encoding",
        "pe_dim",
        "crossing_range",
        "cache_path",
        "quantum_eigenvectors",
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
    elif encoding in {"rrwp+cqrw1", "rrwp+qirw2", "rrwp+photon2"}:
        rrwp_dim = cfg.get("rrwp_dim")
        qpe_dim = cfg.get("qpe_dim")
        if rrwp_dim is None or qpe_dim is None:
            raise ValueError(f"{encoding} requires rrwp_dim and qpe_dim")
        dimension = rrwp_dim + qpe_dim
    elif encoding in {
        "ground_state_corr",
        "laplacian",
        "degree",
        "rrwp_node",
        "rrwp_edge",
    }:
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
    encoding = cfg["encoding"]
    if encoding not in _TIMED_ENCODINGS:
        return None
    if qpe_dim is None:
        raise ValueError(f"{encoding} requires qpe_dim")
    configured_times = cfg.get("times")
    if configured_times is not None:
        if len(configured_times) != qpe_dim:
            raise ValueError(f"{encoding} requires exactly qpe_dim evolution times")
        return [float(value) for value in configured_times]
    minimum_time = cfg.get("qpe_min_time")
    maximum_time = cfg.get("qpe_max_time")
    if minimum_time is None or maximum_time is None:
        raise ValueError(
            f"paper {encoding} configs require qpe_min_time and qpe_max_time"
        )
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
    if cfg["encoding"] in _TIMED_ENCODINGS:
        _resolve_cqrw_times(cfg, qpe_dim)
    expected_initial_distribution = {
        "rrwp+cqrw1": "local",
        "rrwp+qirw2": "adjacency",
        "rrwp+photon2": "collision_free_pairs",
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
            "degree": "degree",
            "rrwp_node": "rrwp",
            "rrwp_edge": "rrwp",
            "none": "none",
        }.get(cfg["encoding"])
        if expected_node_encoding is None:
            raise ValueError(
                "synthetic GCN configs require ground_state_corr, laplacian, "
                "degree, rrwp_node, or none"
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
        if params.get("cache_path") is not None:
            params["cache_path"] = _resolve_paper_data_path(cfg, params["cache_path"])
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
    return_outputs=False,
):
    """Evaluate ``model`` on ``loader``.

    With ``return_outputs=True`` it also returns per-example outputs in loader
    order: absolute errors for regression, or labels and predictions for
    classification. Paired significance tests on the test split use these.
    """
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
    if not return_outputs:
        return metrics
    if head_type in {"graph_class", "node_class"}:
        outputs = {
            "labels": torch.cat(classification_labels).numpy(),
            "predictions": torch.cat(classification_predictions).numpy(),
        }
    else:
        outputs = {"absolute_errors": np.asarray(abs_errs, dtype=np.float64)}
    return metrics, outputs


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


def _run_id(cfg: dict) -> str:
    """Return the run identity; a sweep names each run "<candidate>_seed<seed>"."""
    candidate = cfg.get("sweep", {}).get("candidate")
    seed = int(cfg.get("seed", 0))
    if candidate:
        return f"{candidate}_seed{seed}"
    if cfg.get("experiment") in _SRG_EXPERIMENTS:
        return f"{cfg['experiment']}_seed{seed}"
    return f"{cfg['dataset']}_{cfg['model']}_{cfg['encoding']}_seed{seed}"


def _resolve_paper_data_path(cfg: dict, configured_path: str | Path) -> Path:
    """Resolve a data path against ``<data_root>/QPE_GNN`` when it is relative.

    Relative data paths in configs (SRG catalogs, synthetic dataset caches)
    name a location inside the paper's shared data folder, so they resolve the
    same way whatever the working directory of the run.
    """
    configured = Path(configured_path).expanduser()
    if configured.is_absolute():
        return configured
    if not cfg.get("data_root"):
        raise ValueError(
            f"relative data path {configured} needs data_root (set by the shared runtime)"
        )
    return Path(cfg["data_root"]).expanduser().resolve() / PAPER_NAME / configured


def _resolve_srg_data_directory(cfg: dict) -> Path:
    """Resolve ``srg_data_directory`` against ``<data_root>/QPE_GNN`` when relative."""
    return _resolve_paper_data_path(cfg, cfg["srg_data_directory"])


_BENCHMARK_DATASETS = {"zinc", "mnist", "cifar10", "pattern", "cluster"}


def _dataset_provenance(cfg: dict) -> dict:
    """Describe the dataset subset and preprocessing recorded in run_status.json."""
    if cfg.get("experiment") == "srg_paper":
        return {
            "subset": "every graph of the srg(25,12,5,6) and srg(26,10,3,4) catalogs",
            "preprocessing": (
                "none; 2-QW and Ising correlations and RRWP features are computed "
                "per graph, compared as sorted multisets"
            ),
        }
    if cfg.get("experiment") == "srg_photonic":
        return {
            "subset": "every graph of the srg(25,12,5,6) and srg(26,10,3,4) catalogs",
            "preprocessing": (
                "none; photonic walk covariances (MerLin or closed form) and the "
                "hard-core 2-QW reference are computed per graph, compared as "
                "sorted multisets"
            ),
        }
    dataset_arguments = cfg.get("dataset_kwargs", {})
    if cfg["dataset"] in _BENCHMARK_DATASETS:
        subset = (
            "official 12k ZINC subset"
            if cfg["dataset"] == "zinc" and dataset_arguments.get("subset")
            else "full benchmark"
        )
        if dataset_arguments.get("limit") is not None:
            subset += f", first {dataset_arguments['limit']} records of each split"
    else:
        subset = "generated in memory from dataset.kwargs"
    return {
        "subset": subset,
        "preprocessing": (
            f"positional encoding {cfg['encoding']!r} computed per batch from the "
            "adjacency matrix (no trainable quantum parameters)"
        ),
    }


def _initial_status(
    cfg: dict, run_dir: Path, metrics_name: str, selection: dict
) -> dict:
    sweep_cfg = cfg.get("sweep", {})
    snapshot = run_dir / "config_snapshot.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": _run_id(cfg),
        "sweep_id": sweep_cfg.get("sweep_id"),
        "candidate": sweep_cfg.get("candidate"),
        "seed": int(cfg.get("seed", 0)),
        "repetition": sweep_cfg.get("repetition"),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": {
            "name": cfg.get("dataset", cfg.get("experiment")),
            "kwargs": cfg.get("dataset_kwargs", {}),
            "splits": None,
            **_dataset_provenance(cfg),
        },
        "selection": selection,
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(snapshot) if snapshot.is_file() else None,
        "metrics_path": metrics_name,
        "metrics_sha256": None,
        "error": None,
    }


def _write_checkpoint(path: Path, state: dict) -> None:
    """Write ``state`` atomically so an interrupted save never corrupts it."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _load_resume_state(cfg: dict, device: torch.device) -> dict:
    """Load and check the checkpoint named by ``cfg["resume_from"]``.

    Raises
    ------
    FileNotFoundError
        If the checkpoint does not exist.
    ValueError
        If the checkpoint was written by a different experiment.
    """
    path = Path(cfg["resume_from"]).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    mismatched = [
        key for key in _RESUME_IDENTITY_KEYS if state["config"].get(key) != cfg.get(key)
    ]
    if mismatched:
        raise ValueError(
            "resume checkpoint belongs to a different experiment; differing keys: "
            + ", ".join(mismatched)
        )
    return state


def _check_photonic_features(train_loader, times: list[float], run_dir: Path) -> dict:
    """Recompute the first training graphs' photonic channels with MerLin.

    The ``rrwp+photon2`` encoding uses the closed-form 2x2-permanent backend
    for speed. This check rebuilds the same features with a MerLin
    ``QuantumLayer`` (``ComputationSpace.UNBUNCHED``) for the first
    ``_PHOTONIC_CHECK_GRAPHS`` training graphs, writes ``photonic_check.json``
    and fails the run if the two disagree beyond ``_PHOTONIC_CHECK_TOLERANCE``.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Training loader; its dataset and collate function supply the graphs.
    times : list[float]
        Evolution times of the photonic channels.
    run_dir : pathlib.Path
        Run directory receiving ``photonic_check.json``.

    Returns
    -------
    dict
        The check record written to ``photonic_check.json``.

    Raises
    ------
    ValueError
        If the loader has no dense adjacency or the backends disagree.
    """
    import merlin

    dataset = train_loader.dataset
    records = [
        dataset[index] for index in range(min(_PHOTONIC_CHECK_GRAPHS, len(dataset)))
    ]
    batch = train_loader.collate_fn(records)
    if batch["A"] is None:
        raise ValueError("the photonic check needs dense adjacency batches")
    deviations, success_rates = [], []
    for graph_index in range(len(records)):
        num_nodes = int(batch["mask"][graph_index].sum().item())
        adjacency = (
            batch["A"][graph_index, :num_nodes, :num_nodes]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        for evolution_time in times:
            closed_form, success = photonic_walk_covariance(
                adjacency,
                evolution_time,
                n_photons=2,
                detection="threshold",
                backend="permanent",
                return_success=True,
            )
            simulated = photonic_walk_covariance(
                adjacency,
                evolution_time,
                n_photons=2,
                detection="threshold",
                backend="merlin",
            )
            deviations.append(float(np.max(np.abs(closed_form - simulated))))
            success_rates.append(success)
    record = {
        "graphs_checked": len(records),
        "times": list(times),
        "backend_checked": "permanent",
        "reference": "merlin QuantumLayer, ComputationSpace.UNBUNCHED, float64",
        "merlin_version": merlin.__version__,
        "max_abs_deviation": max(deviations),
        "tolerance": _PHOTONIC_CHECK_TOLERANCE,
        "mean_postselection_success": float(np.mean(success_rates)),
        "min_postselection_success": float(np.min(success_rates)),
    }
    write_json(run_dir / "photonic_check.json", record)
    logger.info(
        "PHOTONIC_CROSS_CHECK | graphs=%d | times=%d | max_abs_deviation=%.3e "
        "| mean_postselection_success=%.4f",
        record["graphs_checked"],
        len(times),
        record["max_abs_deviation"],
        record["mean_postselection_success"],
    )
    logger.info("ARTIFACT_WRITTEN | path=photonic_check.json")
    if record["max_abs_deviation"] > _PHOTONIC_CHECK_TOLERANCE:
        raise ValueError(
            "closed-form photonic features disagree with MerLin: max deviation "
            f"{record['max_abs_deviation']:.3e} > {_PHOTONIC_CHECK_TOLERANCE:g}"
        )
    return record


def _run_srg_experiment(cfg: dict, run_dir: Path) -> dict:
    """Run the SRG experiment with the evidence contract (evaluation only)."""
    status_path = run_dir / "run_status.json"
    status = _initial_status(
        cfg,
        run_dir,
        "srg_metrics.json",
        {
            "name": "distinguished_pairs",
            "direction": "maximize",
            "split": "all catalog pairs",
            "checkpoint": "n/a (no training)",
        },
    )
    write_json(status_path, status)
    logger.info(
        "RUN_STARTED | run_id=%s | experiment=%s", status["run_id"], cfg["experiment"]
    )
    try:
        resolved = dict(cfg)
        resolved["srg_data_directory"] = str(_resolve_srg_data_directory(cfg))
        logger.info(
            "DATASET_READY | name=srg_catalogs | directory=%s",
            resolved["srg_data_directory"],
        )
        logger.info("EVALUATION_STARTED | split=srg_catalogs")
        output = _SRG_EXPERIMENTS[cfg["experiment"]](resolved, run_dir)
        for family, results in output["families"].items():
            for method, summary in results.items():
                logger.info(
                    "EVALUATION_COMPLETED | family=%s | method=%s | distinguished=%s/%s "
                    "| controls_pass=%s",
                    family,
                    method,
                    summary["distinguished_pairs"],
                    summary["total_pairs"],
                    summary["controls_pass"],
                )
        metrics_path = run_dir / "srg_metrics.json"
        logger.info("METRICS_WRITTEN | path=%s", metrics_path.name)
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(metrics_path),
        )
        write_json(status_path, status)
        logger.info("RUN_COMPLETED | run_id=%s", status["run_id"])
        return output
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise


def train_and_evaluate(cfg: dict, run_dir: str) -> dict:
    """Train and evaluate one experiment, writing the run evidence.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (defaults, config overlay and CLI flags).
    run_dir : str or pathlib.Path
        Run directory created by the shared runtime.

    Returns
    -------
    dict
        The metrics written to ``metrics.json`` (``srg_metrics.json`` for the
        SRG experiment).

    Raises
    ------
    Exception
        Any failure is logged as ``RUN_FAILED``, recorded in
        ``run_status.json`` and re-raised.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("experiment") in _SRG_EXPERIMENTS:
        return _run_srg_experiment(cfg, run_dir)
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    head_type = cfg.get("head", "graph_class")
    classification_task = head_type in {"graph_class", "node_class"}
    report_weighted_accuracy = cfg["dataset"] == "pattern"
    selection_metric_name = (
        "weighted_acc"
        if report_weighted_accuracy
        else "acc"
        if classification_task
        else "mae"
    )
    status_path = run_dir / "run_status.json"
    status = _initial_status(
        cfg,
        run_dir,
        "metrics.json",
        {
            "name": f"val_{selection_metric_name}",
            "direction": "maximize" if classification_task else "minimize",
            "split": "val",
            "checkpoint": "best validation epoch",
        },
    )
    write_json(status_path, status)
    logger.info(
        "RUN_STARTED | run_id=%s | dataset=%s | model=%s | encoding=%s | seed=%s | device=%s",
        status["run_id"],
        cfg["dataset"],
        cfg["model"],
        cfg["encoding"],
        cfg.get("seed", 0),
        cfg.get("device", "cpu"),
    )
    try:
        # Validated inside the guard so a rejected config is recorded as
        # RUN_FAILED in run_status.json like any other failure.
        _validate_config(cfg)
        return _train_and_evaluate(
            cfg,
            run_dir,
            status,
            status_path,
            head_type,
            classification_task,
            report_weighted_accuracy,
            selection_metric_name,
        )
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise


def _train_and_evaluate(
    cfg: dict,
    run_dir: Path,
    status: dict,
    status_path: Path,
    head_type: str,
    classification_task: bool,
    report_weighted_accuracy: bool,
    selection_metric_name: str,
) -> dict:
    _seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cpu"))

    loaders = build_dataloaders(cfg)
    train_loader, val_loader, test_loader = (
        loaders["train"],
        loaders["val"],
        loaders["test"],
    )
    train_ds, val_ds, test_ds = (
        train_loader.dataset,
        val_loader.dataset,
        test_loader.dataset,
    )
    status["dataset"]["splits"] = {
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
    }
    write_json(status_path, status)
    logger.info(
        "DATASET_READY | name=%s | train=%d | val=%d | test=%d",
        cfg["dataset"],
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    encoding = cfg["encoding"]
    K, rrwp_dim, qpe_dim = _resolve_positional_encoding_dimensions(cfg)
    times = _resolve_cqrw_times(cfg, qpe_dim)
    photonic_check = (
        _check_photonic_features(train_loader, times, run_dir)
        if encoding == "rrwp+photon2"
        else None
    )
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
    logger.info(
        "MODEL_READY | model=%s | encoding=%s | pe_channels=%d | num_params=%d",
        cfg["model"],
        encoding,
        K,
        num_parameters,
    )

    if classification_task:
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
    best_val = -float("inf") if classification_task else float("inf")
    best_state = None
    best_epoch = None
    best_validation_metrics = None
    start_epoch = 0
    resumed_from = None
    elapsed_before_resume = 0.0
    if cfg.get("resume_from"):
        state = _load_resume_state(cfg, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_val = state["best_val"]
        best_state = state["best_state"]
        best_epoch = state["best_epoch"]
        best_validation_metrics = state["best_validation_metrics"]
        start_epoch = state["epochs_completed"]
        elapsed_before_resume = state["elapsed_seconds"]
        torch.set_rng_state(state["torch_rng_state"])
        if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        resumed_from = str(Path(cfg["resume_from"]).expanduser())
        logger.info(
            "RESUMED_FROM | path=%s | epochs_completed=%d", resumed_from, start_epoch
        )

    checkpoint_every = int(cfg.get("checkpoint_every", 0))
    checkpoint_path = run_dir / "checkpoint.pt"
    logger.info(
        "TRAINING_STARTED | epochs=%d | start_epoch=%d | batch_size=%d | lr=%g | "
        "checkpoint_every=%d",
        epochs,
        start_epoch + 1,
        train_loader.batch_size,
        cfg.get("lr", 1e-3),
        checkpoint_every,
    )
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_started = time.time()
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
            cur_val = va[selection_metric_name]
            improved = cur_val > best_val
        else:
            history["train_mae"].append(tr["mae"])
            history["val_mae"].append(va["mae"])
            cur_val = va["mae"]
            improved = cur_val < best_val
        if improved:
            best_val = cur_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            best_validation_metrics = copy.deepcopy(va)
        scheduler.step()
        logger.info(
            "TRAIN_EPOCH_COMPLETED | epoch=%d/%d | train_loss=%.6f | val_loss=%.6f | "
            "train_%s=%.6f | val_%s=%.6f | lr=%.3g | improved=%s | best_epoch=%s | "
            "seconds=%.1f",
            epoch + 1,
            epochs,
            tr["loss"],
            va["loss"],
            selection_metric_name,
            tr[selection_metric_name],
            selection_metric_name,
            va[selection_metric_name],
            history["learning_rate"][-1],
            improved,
            best_epoch,
            time.time() - epoch_started,
        )
        if checkpoint_every and (
            (epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs
        ):
            _write_checkpoint(
                checkpoint_path,
                {
                    "config": cfg,
                    "epochs_completed": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                    "best_val": best_val,
                    "best_state": best_state,
                    "best_epoch": best_epoch,
                    "best_validation_metrics": best_validation_metrics,
                    "elapsed_seconds": elapsed_before_resume + time.time() - t0,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                    "numpy_rng_state": np.random.get_state(),
                },
            )
            logger.info(
                "CHECKPOINT_WRITTEN | path=%s | epochs_completed=%d",
                checkpoint_path.name,
                epoch + 1,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_model.pt")
        logger.info(
            "CHECKPOINT_WRITTEN | path=best_model.pt | best_epoch=%s", best_epoch
        )
    logger.info("EVALUATION_STARTED | split=test | checkpoint=best_validation")
    test_metrics, test_outputs = _eval(
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
        return_outputs=True,
    )
    logger.info(
        "EVALUATION_COMPLETED | split=test | %s=%.6f | loss=%.6f",
        selection_metric_name,
        test_metrics[selection_metric_name],
        test_metrics["loss"],
    )
    outputs_path = run_dir / "test_outputs.npz"
    np.savez_compressed(outputs_path, **test_outputs)
    logger.info("ARTIFACT_WRITTEN | path=%s", outputs_path.name)
    wall_clock = elapsed_before_resume + time.time() - t0
    out = {
        "schema_version": SCHEMA_VERSION,
        "run_id": status["run_id"],
        "history": history,
        "best_epoch": best_epoch,
        "epochs_completed": epochs,
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
        "test_outputs_path": outputs_path.name,
        "seed": int(cfg.get("seed", 0)),
        "wall_clock_s": wall_clock,
        "num_params": num_parameters,
        "resumed_from": resumed_from,
    }
    if photonic_check is not None:
        out["photonic_check"] = photonic_check
    metrics_path = run_dir / "metrics.json"
    write_json(metrics_path, out)
    logger.info("METRICS_WRITTEN | path=%s", metrics_path.name)
    status.update(
        status="COMPLETED",
        completed_at=utc_now(),
        metrics_sha256=sha256(metrics_path),
    )
    write_json(status_path, status)
    logger.info(
        "RUN_COMPLETED | run_id=%s | test_%s=%.6f",
        status["run_id"],
        selection_metric_name,
        test_metrics[selection_metric_name],
    )
    return out
