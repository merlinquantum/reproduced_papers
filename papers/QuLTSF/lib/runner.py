from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from runtime_lib.dtypes import describe_dtype

from .data import build_dataloaders
from .model import build_model
from .training import evaluate_model, fit_model


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Using dtype: %s", describe_dtype(cfg.get("dtype")))

    train_loader, val_loader, test_loader, metadata = build_dataloaders(cfg)
    model = build_model(cfg, metadata)

    metrics = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        run_dir=run_dir,
    )

    model_name = str(cfg.get("model", {}).get("name", "qultsf_reference")).strip()
    model_path = run_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model checkpoint to %s", model_path)

    predictions = evaluate_model(
        model=model,
        loaders={
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        },
        metadata=metadata,
        cfg=cfg,
    )

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions_path = run_dir / "predictions.json"
    predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    metadata["predictions_path"] = str(predictions_path)
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    done_path = run_dir / "done.txt"
    done_path.write_text("ok\n", encoding="utf-8")
    logger.info("Saved completion marker to %s", done_path)
