"""Runtime entrypoints for the VQC Fourier-series expressivity reproduction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from lib.training import summarize_results, train_models_multiple_runs
from lib.vqc import VQCFactory

from .data import generate_dataset

LOGGER = logging.getLogger(__name__)


def _serialize_metrics(results: dict[str, Any]) -> dict[str, Any]:
    return {
        label: {
            "final_mses": [run["train_mses"][-1] for run in data["runs"]],
            "loss_curves": data["runs"],
            "initial_state": data["initial_state"],
        }
        for label, data in results.items()
    }


def _serialize_learned_functions(
    best_models: list[dict[str, Any]],
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    x_values = inputs.squeeze(-1).cpu().tolist()
    target_values = targets.cpu().tolist()
    entries = []
    for entry in best_models:
        model = entry["model"]
        model.eval()
        with torch.no_grad():
            preds = model(inputs).view(-1).cpu().tolist()
        initial_state = entry.get("initial_state")
        entries.append(
            {
                "label": entry["label"],
                "initial_state": list(initial_state)
                if initial_state is not None
                else [],
                "color": entry.get("color"),
                "predictions": preds,
            }
        )
    return {"x": x_values, "target": target_values, "entries": entries}


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    device = torch.device(cfg.get("device", "cpu"))

    dataset = generate_dataset(cfg.get("data", {}))
    x_tensor = dataset["x"]
    y_tensor = dataset["y"]

    model_factory = VQCFactory(cfg.get("model", {}))
    training_cfg = cfg.get("training", {})
    initial_states = training_cfg.get("initial_states", [])
    plotting_cfg = cfg.get("plotting", {})
    color_palette = plotting_cfg.get("colors")

    results, best_models = train_models_multiple_runs(
        model_factory,
        initial_states,
        x_tensor,
        y_tensor,
        training_cfg,
        device=device,
        colors=color_palette,
    )

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_text = summarize_results(results)
    (run_dir / "summary.txt").write_text(summary_text)
    (run_dir / "metrics.json").write_text(
        json.dumps(_serialize_metrics(results), indent=2)
    )

    learned_dir = run_dir / "learned_function"
    learned_dir.mkdir(parents=True, exist_ok=True)
    (learned_dir / "predictions.json").write_text(
        json.dumps(
            _serialize_learned_functions(best_models, x_tensor, y_tensor), indent=2
        )
    )

    LOGGER.info("Artifacts saved to %s", run_dir.resolve())


__all__ = ["train_and_evaluate"]
