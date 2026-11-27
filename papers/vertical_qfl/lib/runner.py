"""Runtime entry point: train and evaluate one VQFL experiment.

One call trains one model (one experiment, one feature distribution, one seed) and
writes ``run.log``, ``run_status.json``, ``config_snapshot.json`` (the runtime
writes this one) and ``metrics.json`` into ``run_dir``.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from torch import nn

from runtime_lib.dtypes import coerce_dtype_spec

from .data import build_dataloader, build_splits
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .model import VerticalQuantumFederatedModel
from .model_classical import ClassicalCentralModel, ClassicalVerticalFederatedModel
from .model_merlin import PhotonicVerticalFederatedModel

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_KEYS = {
    "dataset": [
        "name",
        "subdir",
        "file",
        "permutation_seed",
        "partition_seed",
        "batch_size",
    ],
    "training": ["epochs", "optimizer", "lr"],
}
# Model keys depend on the backend: the original gate-model reproduction or the
# MerLin photonic adaptation (Phase 4).
MODEL_KEYS = {
    "gate": [
        "client_slices",
        "server_encoding",
        "softmax_output",
        "n_blocks",
        "depth_per_block",
    ],
    "merlin": [
        "client_slices",
        "server_encoding",
        "n_modes",
        "n_photons",
        "client_encoding_scale",
        "server_encoding_scale",
        "n_client_stages",
        "server_modes",
        "server_head",
        "computation_space",
        "reuploads",
        "client_readout",
    ],
    # Phase 5 classical baselines; the remaining keys depend on model.architecture.
    "classical": ["architecture"],
}


def _backend(cfg: dict[str, Any]) -> str:
    # Configs written before the photonic adaptation have no backend key and mean "gate".
    backend = cfg["model"].get("backend", "gate")
    if backend not in MODEL_KEYS:
        raise ValueError(
            f"Unknown model.backend {backend!r}; expected {list(MODEL_KEYS)}"
        )
    return backend


CLASSICAL_KEYS = {
    "vertical": ["client_slices", "server_encoding", "client_hidden", "server_hidden"],
    "central": ["central_hidden"],
}


def _validate_config(cfg: dict[str, Any]) -> None:
    for section, keys in REQUIRED_KEYS.items():
        for key in keys:
            if key not in cfg.get(section, {}):
                raise KeyError(f"Missing required config key {section}.{key}")
    for key in MODEL_KEYS[_backend(cfg)]:
        if key not in cfg["model"]:
            raise KeyError(f"Missing required config key model.{key}")
    if _backend(cfg) == "classical":
        architecture = cfg["model"]["architecture"]
        if architecture not in CLASSICAL_KEYS:
            raise ValueError(f"Unknown model.architecture {architecture!r}")
        for key in CLASSICAL_KEYS[architecture]:
            if key not in cfg["model"]:
                raise KeyError(f"Missing required config key model.{key}")
    if cfg["training"]["optimizer"] != "adam":
        raise ValueError("Only the paper's Adam optimizer is implemented")


def _torch_dtype(cfg: dict[str, Any]) -> torch.dtype:
    spec = coerce_dtype_spec(cfg.get("dtype") or "float32")
    if spec is None or spec.torch is None:
        raise ValueError(f"Cannot resolve dtype from {cfg.get('dtype')!r}")
    return spec.torch


def _build_model(cfg: dict[str, Any], dtype: torch.dtype):
    model_cfg = cfg["model"]
    if _backend(cfg) == "classical":
        if model_cfg["architecture"] == "central":
            return ClassicalCentralModel(int(model_cfg["central_hidden"]), dtype=dtype)
        return ClassicalVerticalFederatedModel(
            client_slices=model_cfg["client_slices"],
            server_encoding=model_cfg["server_encoding"],
            client_hidden=int(model_cfg["client_hidden"]),
            server_hidden=int(model_cfg["server_hidden"]),
            dtype=dtype,
        )
    if _backend(cfg) == "gate":
        return VerticalQuantumFederatedModel(
            client_slices=model_cfg["client_slices"],
            server_encoding=model_cfg["server_encoding"],
            softmax_output=model_cfg["softmax_output"],
            n_blocks=model_cfg["n_blocks"],
            depth_per_block=model_cfg["depth_per_block"],
            dtype=dtype,
        )
    return PhotonicVerticalFederatedModel(
        client_slices=model_cfg["client_slices"],
        server_encoding=model_cfg["server_encoding"],
        n_modes=model_cfg["n_modes"],
        n_photons=model_cfg["n_photons"],
        client_encoding_scale=model_cfg["client_encoding_scale"],
        server_encoding_scale=model_cfg["server_encoding_scale"],
        n_client_stages=model_cfg["n_client_stages"],
        server_modes=model_cfg["server_modes"],
        server_head=int(model_cfg["server_head"]),
        computation_space=model_cfg["computation_space"],
        reuploads=int(model_cfg["reuploads"]),
        client_readout=model_cfg["client_readout"],
        circuit=model_cfg.get("circuit"),
        dtype=dtype,
    )


def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Mean cross-entropy loss and accuracy over a loader."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for pixels, labels in loader:
            pixels, labels = pixels.to(device), labels.to(device)
            scores = model(pixels)
            total_loss += criterion(scores, labels).item()
            correct += (scores.argmax(dim=1) == labels).sum().item()
            count += len(labels)
    return total_loss / count, correct / count


def _train(
    cfg: dict[str, Any],
    model: nn.Module,
    splits: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    training = cfg["training"]
    train_loader = build_dataloader(cfg, "train", splits)
    test_loader = build_dataloader(cfg, "test", splits)
    # An optional held-out validation split (hyperparameter selection runs) is evaluated
    # every epoch next to the test split; selection must use the validation split only.
    validation_loader = (
        build_dataloader(cfg, "validation", splits) if "validation" in splits else None
    )
    # Adam updates every parameter independently, so one optimiser over all parties
    # equals the authors' separate per-party Adam optimisers.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training["lr"],
        weight_decay=training.get("weight_decay", 0.0),
    )
    criterion = nn.CrossEntropyLoss()
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "epoch_seconds": [],
    }
    if validation_loader is not None:
        history["validation_loss"] = []
        history["validation_accuracy"] = []
    epochs = training["epochs"]
    logger.info(
        "TRAINING_STARTED | epochs=%d | optimizer=adam | lr=%g | batch_size=%d",
        epochs,
        training["lr"],
        cfg["dataset"]["batch_size"],
    )
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for step, (pixels, labels) in enumerate(train_loader, start=1):
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad()
            scores = model(pixels)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (scores.argmax(dim=1) == labels).sum().item()
            count += len(labels)
            logger.debug(
                "TRAIN_STEP | epoch=%d | step=%d | loss=%.6f", epoch, step, loss.item()
            )
        train_loss, train_accuracy = loss_sum / count, correct / count
        logger.info(
            "TRAIN_EPOCH_COMPLETED | epoch=%d/%d | loss=%.6f | accuracy=%.4f | lr=%g",
            epoch,
            epochs,
            train_loss,
            train_accuracy,
            training["lr"],
        )
        logger.info("EVALUATION_STARTED | split=test | epoch=%d", epoch)
        test_loss, test_accuracy = _evaluate(model, test_loader, device)
        logger.info(
            "EVALUATION_COMPLETED | split=test | epoch=%d | loss=%.6f | accuracy=%.4f",
            epoch,
            test_loss,
            test_accuracy,
        )
        if validation_loader is not None:
            logger.info("EVALUATION_STARTED | split=validation | epoch=%d", epoch)
            validation_loss, validation_accuracy = _evaluate(
                model, validation_loader, device
            )
            logger.info(
                "EVALUATION_COMPLETED | split=validation | epoch=%d | loss=%.6f | accuracy=%.4f",
                epoch,
                validation_loss,
                validation_accuracy,
            )
            history["validation_loss"].append(validation_loss)
            history["validation_accuracy"].append(validation_accuracy)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        history["epoch_seconds"].append(time.perf_counter() - started)
    return history


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """Train one VQFL model and write the run evidence into ``run_dir``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (see ``configs/defaults.json``).
    run_dir : pathlib.Path
        Output directory created by the shared runtime.

    Raises
    ------
    KeyError
        If a required configuration key is missing.
    """
    run_dir = Path(run_dir)
    configure_logging(
        run_dir / "run.log",
        level=logging.DEBUG
        if cfg.get("logging", {}).get("level") == "debug"
        else logging.INFO,
    )
    # A coordinator may fix the ID so its sweep status can name the run before it ends.
    run_id = cfg.get("run_id") or f"{run_dir.name}-{uuid.uuid4().hex[:8]}"
    config_path = run_dir / "config_snapshot.json"
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": {
            "experiment": cfg.get("experiment"),
            "permutation_seed": cfg.get("dataset", {}).get("permutation_seed"),
            # Hyperparameter values varied by a selection sweep (empty otherwise).
            **cfg.get("candidate_overrides", {}),
        },
        "seed": cfg.get("seed"),
        "repetition": cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": {
            "name": cfg.get("dataset", {}).get("name"),
            "splits": None,
            "preprocessing": "pixels/255*pi/4 (binary pixels -> {0, pi/4}); seeded column permutation; "
            "stratified 80/20 split"
            + (
                "; photonic and classical backends use binary pixels {0,1}"
                if cfg.get("model", {}).get("backend") in ("merlin", "classical")
                else ""
            )
            + (
                f"; {cfg['dataset']['validation_fraction']:.0%} of the training split held out for validation"
                if cfg.get("dataset", {}).get("validation_fraction")
                else ""
            ),
            "subset": {
                "max_train_samples": cfg.get("dataset", {}).get("max_train_samples"),
                "max_test_samples": cfg.get("dataset", {}).get("max_test_samples"),
            },
        },
        "selection": {
            "name": "final_epoch_validation_accuracy"
            if cfg.get("dataset", {}).get("validation_fraction")
            else "final_epoch_test_accuracy",
            "direction": "maximize",
            "split": "validation"
            if cfg.get("dataset", {}).get("validation_fraction")
            else "test",
            "checkpoint": "last",
        },
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(config_path) if config_path.exists() else None,
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "RUN_STARTED | run_id=%s | seed=%s | experiment=%s",
        run_id,
        cfg.get("seed"),
        cfg.get("experiment"),
    )
    try:
        _validate_config(cfg)
        device = torch.device(cfg.get("device", "cpu"))
        dtype = _torch_dtype(cfg)
        torch.manual_seed(int(cfg["seed"]))

        splits = build_splits(cfg)
        status["dataset"]["splits"] = {
            "train": len(splits["train"][1]),
            "test": len(splits["test"][1]),
            **(
                {"validation": len(splits["validation"][1])}
                if "validation" in splits
                else {}
            ),
            "permutation_seed": cfg["dataset"]["permutation_seed"],
            "partition_seed": cfg["dataset"]["partition_seed"],
        }
        write_json(run_dir / "run_status.json", status)
        logger.info("DATASET_READY | %s", status["dataset"]["splits"])

        model = _build_model(cfg, dtype).to(device)
        parameter_counts = model.trainable_parameter_counts()
        logger.info(
            "MODEL_READY | clients=%d | trainable_parameters=%s",
            len(model.clients),
            parameter_counts,
        )

        started = time.perf_counter()
        history = _train(cfg, model, splits, device)
        wall_seconds = time.perf_counter() - started

        test_accuracy = history["test_accuracy"]
        best_epoch = max(range(len(test_accuracy)), key=test_accuracy.__getitem__) + 1
        metrics = {
            "experiment": cfg.get("experiment"),
            "label": cfg.get("label"),
            "history": history,
            "final_epoch": len(test_accuracy),
            "final_test_accuracy": test_accuracy[-1],
            "final_test_loss": history["test_loss"][-1],
            # The paper plots test accuracy per epoch with no validation split, so the
            # best-epoch value is selected on the test set and is reported only as such.
            "best_epoch_test_accuracy_selected_on_test": test_accuracy[best_epoch - 1],
            "best_epoch": best_epoch,
            "trainable_parameters": parameter_counts,
            "wall_clock_seconds": wall_seconds,
            "backend": _backend(cfg),
        }
        if "validation_accuracy" in history:
            metrics["final_validation_accuracy"] = history["validation_accuracy"][-1]
        write_json(run_dir / "metrics.json", metrics)
        logger.info(
            "METRICS_WRITTEN | path=metrics.json | final_test_accuracy=%.4f",
            metrics["final_test_accuracy"],
        )
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(run_dir / "metrics.json"),
        )
        write_json(run_dir / "run_status.json", status)
        logger.info(
            "RUN_COMPLETED | run_id=%s | wall_clock_seconds=%.1f", run_id, wall_seconds
        )
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.exception("RUN_FAILED | run_id=%s", run_id)
        raise
