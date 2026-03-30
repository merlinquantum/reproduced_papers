"""Runtime entrypoints for the VQC classification reproduction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lib.training import ExperimentArgs, summarize_results, train_model_multiple_runs

from .data import prepare_datasets

LOGGER = logging.getLogger(__name__)

MODEL_TYPE_PRESETS: dict[str, tuple[str, list[int]]] = {
    "vqc_100": ("vqc", [1, 0, 0]),
    "vqc_111": ("vqc", [1, 1, 1]),
}


def _normalize_model_type(label: str) -> tuple[str, list[int] | None]:
    base, preset = MODEL_TYPE_PRESETS.get(label, (label, None))
    return base, preset.copy() if preset is not None else None


def build_args(config: dict[str, Any]) -> ExperimentArgs:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    logging_cfg = config.get("logging", {})
    betas = training_cfg.get("betas", [0.9, 0.999])

    args = ExperimentArgs(
        m=model_cfg.get("num_modes", 3),
        input_size=model_cfg.get("input_size", 2),
        initial_state=model_cfg.get("initial_state", [1, 1, 1]),
        activation=model_cfg.get("activation", "none"),
        no_bunching=model_cfg.get("no_bunching", False),
        num_runs=training_cfg.get("num_runs", 5),
        n_epochs=training_cfg.get("epochs", 150),
        batch_size=training_cfg.get("batch_size", 30),
        lr=training_cfg.get("learning_rate", 0.02),
        alpha=training_cfg.get("alpha", 0.0),
        betas=(betas[0], betas[1]),
        circuit=model_cfg.get("circuit", "bs_mesh"),
        scale_type=model_cfg.get("scale_type", "learned"),
        regu_on=model_cfg.get("regularization_target"),
        log_wandb=logging_cfg.get("log_wandb", False),
        wandb_project=logging_cfg.get("wandb_project", "vqc_reproduction"),
        wandb_entity=logging_cfg.get("wandb_entity"),
        device=config.get("device", "cpu"),
    )

    requested_model = config.get("experiment", {}).get("model_type", "vqc")
    base_model, preset_state = _normalize_model_type(requested_model)
    if preset_state is not None:
        args.initial_state = preset_state
    args.requested_model_type = requested_model
    args.set_model_type(base_model)
    return args


def _serialize_training_metrics(results: dict[str, dict]) -> dict[str, Any]:
    return {
        dataset: {
            "runs": _serialize_runs_data(data["runs"]),
            "final_test_accs": [float(run["final_test_acc"]) for run in data["runs"]],
            "avg_final_test_acc": float(data["avg_final_test_acc"]),
        }
        for dataset, data in results.items()
    }


def _serialize_runs_data(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized_runs: list[dict[str, Any]] = []
    for run in runs:
        serialized_run: dict[str, Any] = {}
        for key, value in run.items():
            if isinstance(value, np.floating):
                serialized_run[key] = float(value)
            else:
                serialized_run[key] = value
        serialized_runs.append(serialized_run)
    return serialized_runs


def _serialize_decision_boundaries(
    best_models: list[dict[str, Any]],
    resolution: int = 100,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for entry in best_models:
        x_train = entry["x_train"].cpu().numpy()
        y_train = entry["y_train"].cpu().numpy()
        x_test = entry["x_test"].cpu().numpy()
        y_test = entry["y_test"].cpu().numpy()

        combined = np.vstack([x_train, x_test])
        x_min, x_max = combined[:, 0].min() - 1, combined[:, 0].max() + 1
        y_min, y_max = combined[:, 1].min() - 1, combined[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        model_type = entry["model_type"]
        activation = entry.get("activation", "none")
        model = entry["model"]
        if model_type.startswith("svm"):
            preds = model.predict(grid_points).astype(float)
        else:
            model.eval()
            with torch.no_grad():
                outputs = model(torch.tensor(grid_points, dtype=torch.float32))
            if activation == "softmax":
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                preds = torch.round(outputs).squeeze().cpu().numpy()

        preds = (preds > 0.5).astype(int)
        class_map = preds.reshape(xx.shape).astype(float)

        payloads.append(
            {
                "dataset": entry["dataset"],
                "model_type": model_type,
                "requested_model_type": entry.get("requested_model_type", model_type),
                "activation": activation,
                "initial_state": entry.get("initial_state"),
                "best_acc": float(entry.get("best_acc", 0.0)),
                "x_train": x_train.tolist(),
                "y_train": y_train.tolist(),
                "x_test": x_test.tolist(),
                "y_test": y_test.tolist(),
                "grid_x": xx.tolist(),
                "grid_y": yy.tolist(),
                "class_map": class_map.tolist(),
            }
        )
    return payloads


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    data_cfg = cfg.get("data", {})
    datasets = prepare_datasets(data_cfg)

    args = build_args(cfg)
    results, best_models = train_model_multiple_runs(args.model_type, args, datasets)

    summary = summarize_results(results, args)
    (run_dir / "summary.txt").write_text(summary)

    metrics = _serialize_training_metrics(results)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    boundary_dir = run_dir / "decision_boundaries"
    boundary_dir.mkdir(parents=True, exist_ok=True)
    boundary_payload = _serialize_decision_boundaries(best_models)
    (boundary_dir / "boundary_data.json").write_text(
        json.dumps(boundary_payload, indent=2)
    )

    LOGGER.info("Artifacts saved to %s", run_dir.resolve())


__all__ = ["train_and_evaluate", "build_args"]
