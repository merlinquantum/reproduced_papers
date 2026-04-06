from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _device_from_cfg(cfg: dict) -> torch.device:
    return torch.device(cfg.get("device", "cpu"))


def _mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).abs().mean()


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_mse = 0.0
    total_mae = 0.0
    total_items = 0
    for features, target in loader:
        features = features.to(device)
        target = target.to(device)

        if train_mode:
            optimizer.zero_grad()

        prediction = model(features)
        mse = F.mse_loss(prediction, target)
        mae = _mae(prediction, target)

        if train_mode:
            mse.backward()
            optimizer.step()

        batch_size = int(features.shape[0])
        total_mse += float(mse.detach()) * batch_size
        total_mae += float(mae.detach()) * batch_size
        total_items += batch_size

    if total_items == 0:
        return {"mse": 0.0, "mae": 0.0}
    return {
        "mse": total_mse / total_items,
        "mae": total_mae / total_items,
    }


def fit_model(*, model, train_loader, val_loader, test_loader, cfg: dict, run_dir: Path):
    device = _device_from_cfg(cfg)
    model = model.to(device)
    training_cfg = cfg.get("training", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(training_cfg.get("epochs", 1))
    history = []
    for epoch in range(epochs):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

    test_metrics = _run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        device=device,
    )

    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"history": history, "test": test_metrics}


def evaluate_model(*, model, loaders: dict, metadata: dict, cfg: dict):
    device = _device_from_cfg(cfg)
    model = model.to(device)
    model.eval()

    target_shift = torch.tensor(metadata["target_shift"], device=device)
    target_scale = torch.tensor(metadata["target_scale"], device=device)
    outputs = {}
    with torch.no_grad():
        for split_name, loader in loaders.items():
            rows = []
            for features, target in loader:
                features = features.to(device)
                target = target.to(device)
                prediction = model(features)

                prediction_denorm = prediction * target_scale + target_shift
                target_denorm = target * target_scale + target_shift
                for pred_row, tgt_row in zip(prediction_denorm, target_denorm):
                    rows.append(
                        {
                            "prediction": pred_row.detach().cpu().tolist(),
                            "target": tgt_row.detach().cpu().tolist(),
                        }
                    )
            outputs[split_name] = rows
    return outputs
