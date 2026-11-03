"""Runtime entry point: train one (model, task, hyperparameter, seed) run.

Reproduces one cell of the benchmark grid from arXiv:2504.12416.  Writes
structured artifacts (metrics.json, losses.csv, curve plot, model state dict)
that downstream sweep / plotting utilities aggregate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import DataHandling
from .models import build_model, count_parameters
from .trainer import Trainer

logger = logging.getLogger(__name__)


def _metrics(model: nn.Module, x, y) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(x)
        mse = nn.MSELoss()(out, y).item()
        mae = nn.L1Loss()(out, y).item()
        stacked = torch.reshape(torch.stack((out, y)), (2, -1))
        corr = torch.corrcoef(stacked)[0][1].item()
    return {"mse": mse, "mae": mae, "corr": corr}


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    ds = cfg["dataset"]
    mc = cfg["model"]
    tc = cfg["training"]
    params = mc.get("params", {})

    data_label = ds["name"]
    seq_length = int(ds["sequence_length"])
    prediction_step = int(ds["prediction_step"])
    batch_size = int(ds.get("batch_size", 64))
    # The shared runtime chdirs into the run directory, so resolve a relative
    # data root against the repository root (parents[3] of this file). The
    # datasets live at <repo>/data/variational_qml_ts_benchmark/.
    data_root = Path(ds.get("root", "data/variational_qml_ts_benchmark"))
    if not data_root.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        data_root = repo_root / data_root

    model_name = mc["name"]
    ansatz = params.get("ansatz", "relu_16")
    num_qubits = params.get("num_qubits")
    hidden_size = params.get("hidden_size")
    bugfix = bool(params.get("bugfix", False))

    torch.manual_seed(seed)
    model = build_model(
        model_name,
        data_label,
        seq_length,
        ansatz,
        num_qubits,
        hidden_size,
        seed,
        bugfix=bugfix,
    )
    n_params = count_parameters(model)
    logger.info(
        "model=%s ansatz=%s qubits=%s hidden=%s params=%d seed=%d",
        model_name,
        ansatz,
        num_qubits,
        hidden_size,
        n_params,
        seed,
    )

    data = DataHandling(data_label, seq_length, prediction_step, data_root=data_root)
    xtr, ytr, xval, yval, xte, yte = data.get_training_and_test_data()
    logger.info(
        "shapes: train=%s val=%s test=%s",
        tuple(xtr.shape),
        tuple(xval.shape),
        tuple(xte.shape),
    )

    trainer = Trainer(
        model,
        random_id=seed,
        learning_rate=float(tc.get("lr", 1e-3)),
        batch_size=batch_size,
        max_epochs=tc.get("epochs"),
        min_epochs=int(tc.get("min_epochs", 400)),
        window=int(tc.get("window", 200)),
        use_convergence=bool(tc.get("use_convergence", True)),
    )
    result = trainer.train(xtr, ytr, xval, yval, xte, yte)

    losses = pd.DataFrame(
        {
            "epoch": np.arange(1, result["epochs"] + 1),
            "train_mse": result["cost_training"],
            "val_mse": result["cost_validation"],
            "test_mse": result["cost_testing"],
        }
    )
    losses.to_csv(run_dir / "losses.csv", index=False)

    # Report metrics using the best-validation model (the paper's reported metric).
    model.load_state_dict(result["model_best_validation"])
    best = {f"{k}_test": v for k, v in _metrics(model, xte, yte).items()}
    best.update({f"{k}_val": v for k, v in _metrics(model, xval, yval).items()})
    torch.save(result["model_best_validation"], run_dir / "best_validation_model.pt")

    metrics = {
        "model_name": model_name,
        "data_label": data_label,
        "ansatz": ansatz,
        "num_qubits": num_qubits,
        "hidden_size": hidden_size,
        "seq_length": seq_length,
        "prediction_step": prediction_step,
        "batch_size": batch_size,
        "seed": seed,
        "bugfix": bugfix,
        "num_parameters": n_params,
        "epochs": result["epochs"],
        "total_time_s": result["total_time"],
        "mse_test": best["mse_test"],
        "mse_val": best["mse_val"],
        "mae_test": best["mae_test"],
        "corr_test": best["corr_test"],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    logger.info(
        "DONE mse_test=%.6g mse_val=%.6g epochs=%d time=%.1fs",
        best["mse_test"],
        best["mse_val"],
        result["epochs"],
        result["total_time"],
    )

    _plot_curves(losses, run_dir, metrics)


def _plot_curves(losses: pd.DataFrame, run_dir: Path, metrics: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return
    plt.figure(figsize=(6, 4))
    plt.plot(losses["epoch"], losses["train_mse"], label="train")
    plt.plot(losses["epoch"], losses["val_mse"], label="validation")
    plt.plot(losses["epoch"], losses["test_mse"], label="test")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(
        f"{metrics['model_name']} | {metrics['data_label']} "
        f"pred={metrics['prediction_step']} seq={metrics['seq_length']}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close()
