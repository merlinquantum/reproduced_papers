"""Shared-runtime entry point for the PQKD reproduction.

One run trains one model:

- ``task = "teacher"``: dense teacher, cross-entropy; writes ``teacher.pt``.
- ``task = "student"``: KD student distilled from ``teacher.checkpoint``. The
  student variant and conditioning source select PQKD or a baseline
  (LOG.md, Fair Baseline Plan).

Every run writes ``run.log``, ``run_status.json``, ``config_snapshot.json`` and
``metrics.json`` (the experiment-logging contract), plus the model checkpoint
and, for feature-conditioned students, ``feature_trace.npz`` with the raw and
used conditioning vector of every epoch (paper Supp. Fig. S1).
"""

from __future__ import annotations

import json
import logging
import signal
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset

from runtime_lib.dtypes import dtype_torch

from .conditioning import build_conditioning_source
from .data import DatasetSplits, dataset_info, load_splits, make_loader
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .models import (
    ThreeBlockCNN,
    conv_parameter_count,
    iso_parameter_widths,
    trainable_parameter_count,
)
from .training import (
    evaluate,
    load_checkpoint,
    teacher_logits_for,
    train_student,
    train_teacher,
)

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "checkpoint.pt"
# Keys that identify a run rather than the experiment; they may differ between
# a stopped run and its resumption.
RUN_IDENTITY_KEYS = ("outdir", "sweep", "resume_from", "data_root", "logging")


class StopRequestedError(Exception):
    """Raised by the SIGTERM handler so a stopped run ends as FAILED with its checkpoint kept."""


def _raise_stop(signal_number: int, _frame: Any) -> None:
    raise StopRequestedError(
        f"stopped by signal {signal_number}; resume with --resume-from"
    )


def _experiment_fingerprint(cfg: dict[str, Any]) -> str:
    experiment = {
        key: value for key, value in cfg.items() if key not in RUN_IDENTITY_KEYS
    }
    return json.dumps(experiment, sort_keys=True, default=_json_default)


def _load_resume_state(previous_run: Path, run_dir: Path) -> dict[str, Any]:
    """Load the checkpoint of a stopped run after checking it is the same experiment.

    Both runs' ``config_snapshot.json`` files (written the same way by the
    runtime) must agree on everything except :data:`RUN_IDENTITY_KEYS`.
    """
    checkpoint_path = previous_run / CHECKPOINT_NAME
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No {CHECKPOINT_NAME} in {previous_run}; nothing to resume"
        )
    previous_cfg = json.loads((previous_run / "config_snapshot.json").read_text())
    current_cfg = json.loads((run_dir / "config_snapshot.json").read_text())
    if _experiment_fingerprint(previous_cfg) != _experiment_fingerprint(current_cfg):
        raise ValueError(
            f"{previous_run} was a different experiment configuration; refusing to resume"
        )
    state = load_checkpoint(checkpoint_path)
    logger.info(
        "RESUMED_FROM | run=%s | completed_epochs=%d", previous_run, state["epoch"]
    )
    return state


REPO_ROOT = Path(__file__).resolve().parents[3]
STATUS_SCHEMA_VERSION = 1
TEACHER_CHECKPOINT_NAME = "teacher.pt"
STUDENT_VARIANTS = ("dictionary", "dictionary_trainable", "dense")


def _required(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise KeyError(f"Missing required config key {where}.{key}")
    return section[key]


def _json_default(value: Any) -> Any:
    return str(value)


def _model_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _sampling_generator(seed: int) -> torch.Generator:
    # Separate stream for shots, random features and SPSA directions, so the
    # same seed gives the same student initialisation across conditioning sources.
    return torch.Generator().manual_seed(seed + 1_000_003)


def _resolve_teacher_checkpoint(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_dir():
        path = path / TEACHER_CHECKPOINT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"teacher.checkpoint={path_value!r} does not point to {TEACHER_CHECKPOINT_NAME} "
            "or a run directory containing it; run a task=teacher config first"
        )
    return path


def _build_student(
    cfg: dict[str, Any], in_channels: int, teacher_widths: tuple[int, int, int]
) -> tuple[ThreeBlockCNN, dict[str, Any]]:
    student_cfg = _required(cfg, "student", "cfg")
    conditioning_cfg = _required(cfg, "conditioning", "cfg")
    variant = _required(student_cfg, "variant", "student")
    if variant not in STUDENT_VARIANTS:
        raise ValueError(
            f"Unknown student.variant {variant!r}; expected one of {STUDENT_VARIANTS}"
        )
    compressed = tuple(
        bool(flag) for flag in _required(student_cfg, "compressed_layers", "student")
    )
    ranks = tuple(int(rank) for rank in _required(student_cfg, "ranks", "student"))
    feature_dim = int(_required(conditioning_cfg, "feature_dim", "conditioning"))
    dropout = float(_required(cfg["teacher"], "dropout", "teacher"))
    seed = int(cfg["seed"])

    def dictionary_student(mixing: str) -> ThreeBlockCNN:
        return ThreeBlockCNN(
            in_channels,
            teacher_widths,
            dropout,
            compressed=compressed,
            ranks=ranks,
            feature_dim=feature_dim,
            mixing=mixing,
            generator=_model_generator(seed),
        )

    if variant == "dictionary":
        return dictionary_student("feature"), {"widths": list(teacher_widths)}
    if variant == "dictionary_trainable":
        return dictionary_student("trainable"), {"widths": list(teacher_widths)}

    # B1: dense KD-only student matched to the PQKD student's trainable budget
    # (dictionary layers + dim(theta)).
    reference = dictionary_student("feature")
    target = trainable_parameter_count(reference) + int(conditioning_cfg["theta_dim"])
    widths = iso_parameter_widths(target, in_channels, teacher_widths[2])
    model = ThreeBlockCNN(in_channels, widths, dropout)
    return model, {"widths": list(widths), "iso_parameter_target": target}


def _teacher_task(
    cfg: dict[str, Any], splits: DatasetSplits, run_dir: Path, device: torch.device
) -> dict[str, Any]:
    info = dataset_info(splits.name)
    teacher_cfg = _required(cfg, "teacher", "cfg")
    widths = tuple(int(width) for width in _required(teacher_cfg, "widths", "teacher"))
    model = ThreeBlockCNN(
        info["channels"], widths, float(_required(teacher_cfg, "dropout", "teacher"))
    ).to(device)
    dataset_cfg = cfg["dataset"]
    train_loader = make_loader(
        splits.train,
        int(dataset_cfg["batch_size"]),
        shuffle=True,
        seed=int(cfg["seed"]),
    )
    eval_batch = int(dataset_cfg["eval_batch_size"])
    val_loader = make_loader(splits.val, eval_batch, shuffle=False)
    test_loader = make_loader(splits.test, eval_batch, shuffle=False)

    result = train_teacher(model, train_loader, val_loader, cfg["training"], device)
    final_epoch = result.history[-1]
    model.load_state_dict(result.best_state)
    logger.info(
        "EVALUATION_STARTED | model=teacher | checkpoint=best_val_epoch_%d | splits=val,test",
        result.best_epoch,
    )
    validation = evaluate(model, val_loader, device)
    test = evaluate(model, test_loader, device)
    logger.info(
        "EVALUATION_COMPLETED | model=teacher | val_acc=%.3f | val_ce=%.5f | test_acc=%.3f | test_ce=%.5f",
        validation["accuracy"],
        validation["cross_entropy"],
        test["accuracy"],
        test["cross_entropy"],
    )
    checkpoint_path = run_dir / TEACHER_CHECKPOINT_NAME
    torch.save(
        {
            "state_dict": result.best_state,
            "widths": list(widths),
            "in_channels": info["channels"],
            "dropout": float(teacher_cfg["dropout"]),
            "dataset": splits.name,
            "split_seed": int(dataset_cfg["split_seed"]),
            "seed": int(cfg["seed"]),
            "best_epoch": result.best_epoch,
        },
        checkpoint_path,
    )
    logger.info("CHECKPOINT_WRITTEN | path=%s", checkpoint_path.name)
    return {
        "task": "teacher",
        "dataset": splits.name,
        "widths": list(widths),
        "trainable_parameters": trainable_parameter_count(model),
        "conv_parameters": conv_parameter_count(model),
        "final_epoch": final_epoch,
        "best_checkpoint": {
            "epoch": result.best_epoch,
            "val_accuracy": validation["accuracy"],
            "val_cross_entropy": validation["cross_entropy"],
            "test_accuracy": test["accuracy"],
            "test_cross_entropy": test["cross_entropy"],
        },
        "checkpoint_path": checkpoint_path.name,
        "history": result.history,
    }


def _student_task(
    cfg: dict[str, Any], splits: DatasetSplits, run_dir: Path, device: torch.device
) -> dict[str, Any]:
    info = dataset_info(splits.name)
    dataset_cfg = cfg["dataset"]
    teacher_path = _resolve_teacher_checkpoint(
        _required(cfg["teacher"], "checkpoint", "teacher")
    )
    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    expected_widths = [int(width) for width in cfg["teacher"]["widths"]]
    mismatches = {
        "dataset": (checkpoint["dataset"], splits.name),
        "split_seed": (checkpoint["split_seed"], int(dataset_cfg["split_seed"])),
        "widths": (checkpoint["widths"], expected_widths),
    }
    mismatches = {
        key: value for key, value in mismatches.items() if value[0] != value[1]
    }
    if mismatches:
        raise ValueError(
            f"Teacher checkpoint {teacher_path} does not match this run: {mismatches}"
        )
    teacher = ThreeBlockCNN(
        checkpoint["in_channels"], checkpoint["widths"], checkpoint["dropout"]
    ).to(device)
    teacher.load_state_dict(checkpoint["state_dict"])
    logger.info(
        "TEACHER_LOADED | path=%s | widths=%s | best_epoch=%s",
        teacher_path,
        checkpoint["widths"],
        checkpoint["best_epoch"],
    )

    eval_batch = int(dataset_cfg["eval_batch_size"])
    val_loader = make_loader(splits.val, eval_batch, shuffle=False)
    test_loader = make_loader(splits.test, eval_batch, shuffle=False)
    teacher_val = evaluate(teacher, val_loader, device)
    teacher_test = evaluate(teacher, test_loader, device)
    train_logits = teacher_logits_for(teacher, splits.train, device, eval_batch)
    train_with_logits = TensorDataset(*splits.train.tensors, train_logits)
    train_loader = make_loader(
        train_with_logits,
        int(dataset_cfg["batch_size"]),
        shuffle=True,
        seed=int(cfg["seed"]),
    )

    spsa_cfg = cfg["spsa"]
    proxy_size = int(spsa_cfg["val_batches"]) * int(spsa_cfg["val_batch_size"])
    proxy_images, proxy_labels = (tensor[:proxy_size] for tensor in splits.val.tensors)
    proxy_logits = teacher_logits_for(
        teacher, TensorDataset(proxy_images, proxy_labels), device, eval_batch
    )
    batch = int(spsa_cfg["val_batch_size"])
    val_proxy_batches = [
        (
            proxy_images[start : start + batch],
            proxy_labels[start : start + batch],
            proxy_logits[start : start + batch],
        )
        for start in range(0, proxy_images.shape[0], batch)
    ]

    model, student_meta = _build_student(
        cfg, info["channels"], tuple(checkpoint["widths"])
    )
    model = model.to(device)
    generator = _sampling_generator(int(cfg["seed"]))
    source = build_conditioning_source(cfg, generator) if model.uses_feature else None

    resume_state = None
    if cfg.get("resume_from"):
        resume_state = _load_resume_state(Path(cfg["resume_from"]), run_dir)
    result = train_student(
        model,
        source,
        train_loader,
        val_loader,
        val_proxy_batches,
        cfg,
        device,
        generator,
        checkpoint_path=run_dir / CHECKPOINT_NAME,
        resume_state=resume_state,
    )

    logger.info(
        "EVALUATION_STARTED | model=student | checkpoint=final_epoch | splits=val,test"
    )
    validation = evaluate(model, val_loader, device)
    test = evaluate(model, test_loader, device)
    conditioning_checks: dict[str, Any] = {}
    if model.uses_feature:
        # Paper Fig. 4 control: the same trained student evaluated with z = 0.
        model.set_feature(source.zero_feature().to(device))
        test_zero = evaluate(model, test_loader, device)
        # One fresh shot-limited draw without EMA, to expose test-time shot noise.
        model.set_feature(source.feature(generator).to(device))
        test_fresh = evaluate(model, test_loader, device)
        model.set_feature(result.feature_used.to(device))
        conditioning_checks = {
            "test_accuracy_zero_feature": test_zero["accuracy"],
            "delta_test_accuracy_vs_zero_feature": test["accuracy"]
            - test_zero["accuracy"],
            "test_accuracy_fresh_feature": test_fresh["accuracy"],
        }
    logger.info(
        "EVALUATION_COMPLETED | model=student | val_acc=%.3f | val_ce=%.5f | test_acc=%.3f | test_ce=%.5f | teacher_test_acc=%.3f%s",
        validation["accuracy"],
        validation["cross_entropy"],
        test["accuracy"],
        test["cross_entropy"],
        teacher_test["accuracy"],
        "".join(f" | {key}={value:.3f}" for key, value in conditioning_checks.items()),
    )

    student_trainable = trainable_parameter_count(model)
    conditioning_parameters = source.parameter_count if source is not None else 0
    teacher_parameters = trainable_parameter_count(teacher)
    teacher_conv = conv_parameter_count(teacher)
    student_conv = conv_parameter_count(model)
    total_student = student_trainable + conditioning_parameters
    history = result.history
    proxy_delta = history[0]["proxy_objective"] - history[-1]["proxy_objective"]

    student_path = run_dir / "student.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_used": result.feature_used,
            "conditioning_vector": source.vector() if source is not None else None,
        },
        student_path,
    )
    logger.info("CHECKPOINT_WRITTEN | path=%s", student_path.name)
    trace_path = None
    if result.feature_trace_raw and bool(cfg["logging"]["save_feature_trace"]):
        trace_path = run_dir / "feature_trace.npz"
        np.savez_compressed(
            trace_path,
            raw=torch.stack(result.feature_trace_raw).numpy(),
            used=torch.stack(result.feature_trace_used).numpy(),
        )
        logger.info("ARTIFACT_WRITTEN | path=%s", trace_path.name)

    return {
        "task": "student",
        "dataset": splits.name,
        "variant": cfg["student"]["variant"],
        "conditioning_source": source.name if source is not None else "none",
        "student": student_meta,
        "teacher_checkpoint": str(teacher_path),
        "teacher": {
            "widths": checkpoint["widths"],
            "val_accuracy": teacher_val["accuracy"],
            "val_cross_entropy": teacher_val["cross_entropy"],
            "test_accuracy": teacher_test["accuracy"],
            "test_cross_entropy": teacher_test["cross_entropy"],
            "trainable_parameters": teacher_parameters,
            "conv_parameters": teacher_conv,
        },
        "final_epoch": history[-1],
        "final_evaluation": {
            "val_accuracy": validation["accuracy"],
            "val_cross_entropy": validation["cross_entropy"],
            "test_accuracy": test["accuracy"],
            "test_cross_entropy": test["cross_entropy"],
            "val_accuracy_drop_vs_teacher": teacher_val["accuracy"]
            - validation["accuracy"],
            "test_accuracy_drop_vs_teacher": teacher_test["accuracy"]
            - test["accuracy"],
            **conditioning_checks,
        },
        "parameters": {
            "student_trainable": student_trainable,
            "conditioning_trainable": conditioning_parameters,
            "student_total_trainable": total_student,
            "student_conv": student_conv,
            "compression_overall": teacher_parameters / total_student,
            "compression_conv_only": teacher_conv
            / (student_conv + conditioning_parameters),
        },
        "proxy_objective_delta_first_minus_final": proxy_delta,
        "feature_trace_path": trace_path.name if trace_path is not None else None,
        "checkpoint_path": student_path.name,
        "history": history,
    }


def _selection(task: str) -> dict[str, str]:
    if task == "teacher":
        return {
            "name": "val_accuracy",
            "direction": "maximize",
            "split": "val",
            "checkpoint": "best_val_epoch",
        }
    return {
        "name": "test_accuracy",
        "direction": "maximize",
        "split": "test",
        "checkpoint": "final_epoch",
    }


def _candidate(cfg: dict[str, Any]) -> dict[str, Any]:
    task = cfg["task"]
    candidate = {
        "task": task,
        "dataset": cfg["dataset"]["name"],
        "teacher_widths": cfg["teacher"]["widths"],
    }
    if task == "student":
        candidate.update(
            {
                "variant": cfg["student"]["variant"],
                "compressed_layers": cfg["student"]["compressed_layers"],
                "ranks": cfg["student"]["ranks"],
                "conditioning_source": cfg["conditioning"]["source"],
                "theta_dim": cfg["conditioning"]["theta_dim"],
                "shots": cfg["conditioning"]["shots"],
                "ema": cfg["conditioning"]["ema"]["enabled"],
                "theta_init_mode": cfg["conditioning"]["theta_init_mode"],
            }
        )
    return candidate


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """Run one PQKD teacher or student experiment and write its evidence.

    Parameters
    ----------
    cfg : dict
        Resolved configuration from the shared runtime.
    run_dir : pathlib.Path
        Timestamped run directory.

    Raises
    ------
    ValueError
        If ``task``, the student variant or the dtype is unsupported.
    """
    run_dir = Path(run_dir)
    level = getattr(logging, str(cfg.get("logging", {}).get("level", "info")).upper())
    configure_logging(run_dir / "run.log", level=level)
    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(cfg, indent=2, default=_json_default), encoding="utf-8"
        )
    task = _required(cfg, "task", "cfg")
    seed = int(_required(cfg, "seed", "cfg"))
    # utils/run_sweep.py passes its own run identity so the sweep validator can
    # match this run to its planned (candidate, seed, repetition) entry.
    sweep_context = cfg.get("sweep") or {}
    run_id = sweep_context.get("run_id") or run_dir.name
    status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_id": run_id,
        "sweep_id": sweep_context.get("sweep_id"),
        "candidate": sweep_context.get("candidate"),
        "candidate_details": None,
        "seed": seed,
        "repetition": sweep_context.get("repetition"),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": {
            "name": cfg.get("dataset", {}).get("name"),
            "splits": None,
            "preprocessing": None,
            "subset": None,
        },
        "selection": _selection(task),
        "log_path": "run.log",
        "config_path": config_path.name,
        "config_sha256": sha256(config_path),
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
        "resumed_from": cfg.get("resume_from"),
    }
    status_path = run_dir / "run_status.json"
    write_json(status_path, status)
    logger.info("RUN_STARTED | run_id=%s | task=%s | seed=%d", run_id, task, seed)
    # A SIGTERM (utils/stop_sweep.py, or the sweep coordinator stopping) ends
    # the run as FAILED with a traceback; the last epoch checkpoint survives.
    previous_handler = signal.signal(signal.SIGTERM, _raise_stop)
    try:
        if task not in ("teacher", "student"):
            raise ValueError(f"Unknown task {task!r}; expected 'teacher' or 'student'")
        dtype = dtype_torch(cfg.get("dtype"))
        if dtype not in (None, torch.float32):
            raise ValueError(f"Only float32 is supported (paper setting), got {dtype}")
        status["candidate_details"] = _candidate(cfg)
        if status["candidate"] is None:
            status["candidate"] = status["candidate_details"]
        device = torch.device(_required(cfg, "device", "cfg"))
        torch.manual_seed(seed)

        splits = load_splits(cfg)
        status["dataset"] = {
            "name": splits.description["name"],
            "splits": splits.description["splits"],
            "preprocessing": splits.description["preprocessing"],
            "subset": splits.description["subset"],
        }
        write_json(status_path, status)
        logger.info(
            "DATASET_READY | name=%s | train=%d | val=%d | test=%d | subset=%s",
            splits.name,
            len(splits.train),
            len(splits.val),
            len(splits.test),
            splits.description["subset"],
        )

        if task == "teacher":
            metrics = _teacher_task(cfg, splits, run_dir, device)
        else:
            metrics = _student_task(cfg, splits, run_dir, device)

        metrics_path = run_dir / "metrics.json"
        write_json(metrics_path, json.loads(json.dumps(metrics, default=_json_default)))
        logger.info("METRICS_WRITTEN | path=%s", metrics_path.name)
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(metrics_path),
        )
        write_json(status_path, status)
        logger.info("RUN_COMPLETED | run_id=%s", run_id)
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("RUN_FAILED | run_id=%s", run_id)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
