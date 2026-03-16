from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from runtime_lib.dtypes import describe_dtype, dtype_torch

from .data import build_dataloaders
from .model import RNNRegressor
from .training import fit


def _build_model(cfg: dict, input_size: int) -> nn.Module:
    logger = logging.getLogger(__name__)
    model_cfg = cfg.get("model", {})
    name = str(model_cfg.get("name", "rnn_baseline")).strip().lower()
    params = model_cfg.get("params", {})

    if name in {"gate_pqrnn", "pqrnn_gate", "pqrnn"}:
        try:
            from .gate_pqrnn import GatePQRNNConfig, GatePQRNNRegressor
        except Exception as exc:
            raise RuntimeError(
                "gate_pqrnn model requires PennyLane. Install it (pip install pennylane), "
                "or use model.name='rnn_baseline' / 'photonic_qrnn'."
            ) from exc

        # Parameter naming convention: align with the photonic QRNN where possible.
        # - Prefer kd/kh (logical data/hidden size)
        # - Keep n_data/n_hidden as backwards-compatible aliases
        kd = params.get("kd", params.get("n_data", 2))
        kh = params.get("kh", params.get("n_hidden", 1))

        if "shots" in params and params.get("shots") not in (0, None):
            logger.warning(
                "Ignoring model.params.shots=%r for gate_pqrnn (analytic mode only).",
                params.get("shots"),
            )

        gate_cfg = GatePQRNNConfig(
            n_data=int(kd),
            n_hidden=int(kh),
            depth=int(params.get("depth", 1)),
            entangling=str(params.get("entangling", "cb")),
            entangling_wrap=bool(params.get("entangling_wrap", True)),
            dtype=dtype_torch(cfg.get("dtype")),
            input_encoding=str(params.get("input_encoding", "identity")),
        )
        return GatePQRNNRegressor(input_size=input_size, config=gate_cfg)

    if name == "photonic_qrnn":
        try:
            from .photonic_qrnn import PhotonicQRNNConfig, PhotonicQRNNRegressor
        except Exception as exc:
            raise RuntimeError(
                "photonic_qrnn model requires Merlin/Perceval features that are not available in this environment. "
                "Install a compatible merlinquantum/merlin version, or use model.name='rnn_baseline'."
            ) from exc

        if "branch_selection" in params:
            logger.warning(
                "Ignoring deprecated model.params.branch_selection=%r; photonic QRNN always propagates all branches.",
                params.get("branch_selection"),
            )
        if "y_dim" in params:
            logger.warning(
                "Ignoring deprecated model.params.y_dim=%r; photonic QRNN output dimension is derived from the dataset (currently scalar).",
                params.get("y_dim"),
            )
        photonic_cfg = PhotonicQRNNConfig(
            kd=int(params.get("kd", 1)),
            kh=int(params.get("kh", 1)),
            depth=int(params.get("depth", 1)),
            shots=int(params.get("shots", 0)),
            dtype=dtype_torch(cfg.get("dtype")),
            measurement_space=str(params.get("measurement_space", "dual_rail")),
            input_encoding=str(params.get("input_encoding", "identity")),
        )
        return PhotonicQRNNRegressor(input_size=input_size, config=photonic_cfg)

    hidden_dim = int(params.get("hidden_dim", 64))
    layers = int(params.get("layers", 1))
    dropout = float(params.get("dropout", 0.0))
    cell_type = str(params.get("cell_type", "rnn"))
    bidirectional = bool(params.get("bidirectional", False))
    input_dropout = float(params.get("input_dropout", 0.0))
    model = RNNRegressor(
        input_size=input_size,
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=dropout,
        cell_type=cell_type,
        bidirectional=bidirectional,
        input_dropout=input_dropout,
    )
    return model


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    dtype_label = describe_dtype(cfg.get("dtype"))
    logger.info("Using dtype: %s", dtype_label)

    train_loader, val_loader, test_loader, metadata = build_dataloaders(cfg)
    model = _build_model(cfg, metadata["input_size"])

    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "rnn_baseline")).strip().lower()
    model_params = model_cfg.get("params", {})

    dataset_cfg = cfg.get("dataset", {})
    logger.info(
        "Config summary | dataset=%s path=%s preprocess=%s seq_len=%d pred_horizon=%d batch=%d "
        "splits(train/val/test)=%d/%d/%d",
        dataset_cfg.get("name"),
        dataset_cfg.get("path"),
        dataset_cfg.get("preprocess"),
        dataset_cfg.get("sequence_length"),
        dataset_cfg.get("prediction_horizon"),
        dataset_cfg.get("batch_size"),
        metadata.get("splits", {}).get("train", 0),
        metadata.get("splits", {}).get("val", 0),
        metadata.get("splits", {}).get("test", 0),
    )

    if model_name == "photonic_qrnn":
        logger.info(
            "Model summary | model=%s kd=%s kh=%s depth=%s shots=%s measurement_space=%s input_encoding=%s | "
            "epochs=%d lr=%g opt=%s",
            model_name,
            model_params.get("kd"),
            model_params.get("kh"),
            model_params.get("depth"),
            model_params.get("shots"),
            model_params.get("measurement_space"),
            model_params.get("input_encoding", "identity"),
            cfg.get("training", {}).get("epochs"),
            cfg.get("training", {}).get("lr"),
            cfg.get("training", {}).get("optimizer", "adam"),
        )
    elif model_name in {"gate_pqrnn", "pqrnn_gate", "pqrnn"}:
        logger.info(
            "Model summary | model=%s kd=%s kh=%s depth=%s entangling=%s input_encoding=%s | epochs=%d lr=%g opt=%s",
            model_name,
            model_params.get("kd", model_params.get("n_data")),
            model_params.get("kh", model_params.get("n_hidden")),
            model_params.get("depth"),
            model_params.get("entangling", "cb"),
            model_params.get("input_encoding", "identity"),
            cfg.get("training", {}).get("epochs"),
            cfg.get("training", {}).get("lr"),
            cfg.get("training", {}).get("optimizer", "adam"),
        )
    else:
        logger.info(
            "Model summary | model=%s cell=%s hidden=%s layers=%s dropout=%s bidi=%s | epochs=%d lr=%g opt=%s",
            model_name,
            model_params.get("cell_type", "rnn"),
            model_params.get("hidden_dim"),
            model_params.get("layers"),
            model_params.get("dropout"),
            model_params.get("bidirectional", False),
            cfg.get("training", {}).get("epochs"),
            cfg.get("training", {}).get("lr"),
            cfg.get("training", {}).get("optimizer", "adam"),
        )

    dtype = dtype_torch(cfg.get("dtype"))
    is_quantum = model_name in {"photonic_qrnn", "gate_pqrnn", "pqrnn_gate", "pqrnn"}
    if dtype is not None and not is_quantum:
        model = model.to(dtype=dtype)

    if model_name == "photonic_qrnn":
        try:
            from .perceval_export import export_qrb_circuit_svg

            svg_path = export_qrb_circuit_svg(
                run_dir=run_dir,
                kd=int(model_cfg.get("params", {}).get("kd", 1)),
                kh=int(model_cfg.get("params", {}).get("kh", 1)),
            )
            if svg_path is not None:
                logger.info("Saved QRB circuit diagram to %s", svg_path)
        except Exception:
            logger.warning("Failed to export QRB circuit diagram", exc_info=True)

    device = torch.device(cfg.get("device", "cpu"))
    cfg_with_metadata = dict(cfg)
    cfg_with_metadata["metadata"] = metadata
    metrics = fit(
        model, train_loader, val_loader, test_loader, cfg_with_metadata, run_dir
    )

    model_path = run_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model checkpoint to %s", model_path)

    accuracies = _write_predictions_csv(
        model, cfg_with_metadata, train_loader, val_loader, test_loader, device, run_dir
    )
    if accuracies:
        metrics["accuracy"] = accuracies
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metadata["predictions_csv"] = str(run_dir / "predictions.csv")
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved metadata to %s", metadata_path)

    done_marker = run_dir / "done.txt"
    done_marker.write_text("ok", encoding="utf-8")
    logger.info("Saved completion marker to %s", done_marker)


def _build_prediction_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    batch_size: int,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for name, loader in (
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ):
        if loader is None or len(loader.dataset) == 0:  # type: ignore[arg-type]
            continue
        subset = loader.dataset
        loaders[name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
    return loaders


def _write_predictions_csv(
    model: torch.nn.Module,
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
) -> dict[str, float | None]:
    model.eval()
    batch_size = int(cfg.get("dataset", {}).get("batch_size", 16))
    metadata = cfg.get("metadata", {}) or {}
    target_scale = metadata.get("target_abs_mean")
    target_mean = metadata.get("target_mean", 0.0)
    target_std = metadata.get("target_std", 1.0)
    loaders = _build_prediction_loaders(
        train_loader, val_loader, test_loader, batch_size
    )

    rows = []
    with torch.no_grad():
        for split, loader in loaders.items():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                preds_norm = model(batch_x).cpu().view(-1)
                targets_norm = batch_y.view(-1)
                preds = (preds_norm * target_std) + target_mean
                targets = (targets_norm * target_std) + target_mean
                for pred, target in zip(preds.tolist(), targets.tolist()):
                    rows.append({"split": split, "prediction": pred, "target": target})

    pred_path = run_dir / "predictions.csv"
    df = pd.DataFrame(rows)
    df.to_csv(pred_path, index=False)
    logging.getLogger(__name__).info("Saved predictions to %s", pred_path)

    accuracies: dict[str, float | None] = {}
    for split, split_df in df.groupby("split"):
        valid = split_df[split_df["target"] != 0]
        if valid.empty:
            accuracies[split] = None
            continue
        scale = target_scale if target_scale is not None else 0.0
        denom = np.clip(np.abs(valid["target"]), max(1e-6, scale), None)
        rel_error = (valid["target"] - valid["prediction"]) / denom
        rmse_rel = float(np.sqrt(np.mean(np.square(rel_error))))
        accuracies[split] = float((1 - rmse_rel) * 100)
    return accuracies


__all__ = ["train_and_evaluate"]
