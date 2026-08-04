"""Runtime entry point for the distributed_qml_cc reproduction.

This module exposes :func:`train_and_evaluate`, the function expected by
the shared repository runtime. It dispatches to one of three pipelines:

* ``quantum``     - the DQML circuit reproduction (the paper's core
  experiment).
* ``classical``   - a fair iso-parameter classical MLP baseline.
* ``merlin``      - the photonic MerLin extension of the CC scheme.

The CLI surface keeps the four DQML schemes selectable through
``model.scheme`` and the convolutional depth through ``model.n_layers``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import SyntheticDatasetConfig, build_synthetic_dataset
from .model import DQMLConfig, DQMLModel
from .training import train_dqml

LOGGER = logging.getLogger(__name__)


def _tensors(x_train, y_train, x_val, y_val, dtype):
    return (
        torch.as_tensor(x_train, dtype=dtype),
        torch.as_tensor(y_train, dtype=dtype),
        torch.as_tensor(x_val, dtype=dtype),
        torch.as_tensor(y_val, dtype=dtype),
    )


def _resolve_dtype(cfg: dict) -> torch.dtype:
    raw = cfg.get("dtype", "float32")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return raw[1]
    if isinstance(raw, torch.dtype):
        return raw
    return {"float32": torch.float32, "float64": torch.float64}.get(str(raw), torch.float32)


def _run_quantum(cfg: dict, run_dir: Path, dtype: torch.dtype) -> dict[str, Any]:
    """Train one or more DQML configurations and serialise the metrics."""
    dataset_cfg = SyntheticDatasetConfig(**cfg.get("dataset", {}).get("params", {}))
    model_cfg = cfg.get("model", {}).get("params", {})
    scheme = str(model_cfg.get("scheme", "cc")).lower()
    n_layers = int(model_cfg.get("n_layers", 9))
    qubits_per_qpu = int(model_cfg.get("qubits_per_qpu", 4))

    training_cfg = cfg.get("training", {})
    n_iterations = int(training_cfg.get("n_iterations", 1000))
    lr = float(training_cfg.get("lr", 0.05))
    batch_size = int(training_cfg.get("batch_size", 512))
    eval_every = int(training_cfg.get("eval_every", 50))

    seeds = list(cfg.get("seeds") or [int(cfg.get("seed", 42))])

    x_tr, y_tr, x_va, y_va, info = build_synthetic_dataset(dataset_cfg)
    x_tr_t, y_tr_t, x_va_t, y_va_t = _tensors(x_tr, y_tr, x_va, y_va, dtype)

    summary: dict[str, Any] = {
        "scheme": scheme,
        "n_layers": n_layers,
        "qubits_per_qpu": qubits_per_qpu,
        "n_iterations": n_iterations,
        "lr": lr,
        "batch_size": batch_size,
        "dataset_info": {
            "n_train": int(x_tr.shape[0]),
            "n_val": int(x_va.shape[0]),
            "pearson_max_abs": info["pearson_max_abs"],
            "seed": dataset_cfg.seed,
        },
        "seeds": seeds,
        "runs": [],
    }

    for seed in seeds:
        LOGGER.info("Training %s scheme L=%d seed=%d", scheme, n_layers, seed)
        torch.manual_seed(int(seed))
        model = DQMLModel(
            DQMLConfig(scheme=scheme, n_layers=n_layers, qubits_per_qpu=qubits_per_qpu),
            dtype=dtype,
        )
        n_params = model.num_parameters()
        t0 = time.time()
        result = train_dqml(
            model,
            x_tr_t,
            y_tr_t,
            x_va_t,
            y_va_t,
            n_iterations=n_iterations,
            lr=lr,
            batch_size=batch_size,
            eval_every=eval_every,
            seed=int(seed),
        )
        elapsed = time.time() - t0
        run_entry = {
            "seed": int(seed),
            "n_params": int(n_params),
            "wall_clock_seconds": elapsed,
            "final_val_acc": result.final_val_acc,
            "final_train_acc": result.final_train_acc,
            "final_loss": result.final_loss,
            "history": {
                "iteration": result.iterations,
                "train_loss": result.train_loss,
                "train_acc": result.train_acc,
                "val_acc": result.val_acc,
            },
        }
        summary["runs"].append(run_entry)
        LOGGER.info(
            "  seed %d val_acc=%.4f train_acc=%.4f loss=%.4f time=%.1fs",
            seed, result.final_val_acc, result.final_train_acc, result.final_loss, elapsed,
        )

    val_accs = np.array([r["final_val_acc"] for r in summary["runs"]])
    summary["mean_val_acc"] = float(val_accs.mean())
    summary["std_val_acc"] = float(val_accs.std(ddof=0))

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Wrote metrics to %s", metrics_path)
    return summary


def _run_classical(cfg: dict, run_dir: Path, dtype: torch.dtype) -> dict[str, Any]:
    """Train an iso-parameter MLP baseline."""
    from .classical_model import train_classical_baseline

    return train_classical_baseline(cfg, run_dir, dtype)


def _run_merlin(cfg: dict, run_dir: Path, dtype: torch.dtype) -> dict[str, Any]:
    """Train the photonic MerLin single-chip baseline."""
    from .merlin_model import train_merlin_dqml

    return train_merlin_dqml(cfg, run_dir, dtype)


def _run_merlin_distributed(cfg: dict, run_dir: Path, dtype: torch.dtype) -> dict[str, Any]:
    """Train the photonic distributed (two-chip) NC/CC variant."""
    from .merlin_distributed import train_merlin_distributed

    return train_merlin_distributed(cfg, run_dir, dtype)


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline = str(cfg.get("pipeline", "quantum")).lower()
    dtype = _resolve_dtype(cfg)
    LOGGER.info("Running pipeline=%s dtype=%s -> %s", pipeline, dtype, run_dir)

    if pipeline == "quantum":
        _run_quantum(cfg, run_dir, dtype)
    elif pipeline == "classical":
        _run_classical(cfg, run_dir, dtype)
    elif pipeline == "merlin":
        _run_merlin(cfg, run_dir, dtype)
    elif pipeline == "merlin_distributed":
        _run_merlin_distributed(cfg, run_dir, dtype)
    else:
        raise ValueError(f"unknown pipeline '{pipeline}'")
