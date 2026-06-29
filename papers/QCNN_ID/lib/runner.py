"""Orchestrator for QCNN-ID reproductions.

Exposes :func:`train_and_evaluate` as required by the shared runtime in
``implementation.py``. The function trains every requested model variant on the
preprocessed dataset, dumps structured artifacts under ``run_dir``, and renders
the loss / accuracy / precision / recall / time per-epoch figure that mirrors
Fig. 3 of the paper.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import load_and_prepare
from .models import (
    CNNClassifier,
    PhotonicClassifier,
    QCNNClassifier,
)
from .training import EpochMetrics, TrainHistory, train_and_evaluate_model

# All implemented models
MODEL_REGISTRY = (
    "cnn_classifier",
    "qcnn_classifier",
    "photonic_classifier",
)

QUANTUM_MODEL_REGISTRY = (
    "qcnn_classifier",
    "photonic_classifier",
)


def _public_config_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a portable config snapshot without local runtime paths."""

    snapshot = dict(cfg)
    snapshot.pop("data_root", None)
    return snapshot


def _build_model(
    name: str,
    cfg: dict[str, Any],
    input_dim: int,
    device: torch.device,
):
    if name == "cnn_classifier":
        return CNNClassifier(
            input_dim=input_dim,
            num_classes=1,
            device=device,
        )
    if name == "qcnn_classifier":
        return QCNNClassifier(
            n_qubits=int(cfg["n_qubits"]),
            reps=int(cfg["ansatz_reps"]),
            num_classes=1,
            device=device,
        )
    if name == "photonic_classifier":
        return PhotonicClassifier(
            n_modes=int(cfg["n_components"]),
            n_photons=int(cfg["n_components"]) // 2,
            num_classes=1,
            device=device,
        )
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_REGISTRY}")


def _features_for(name: str, prepared):
    if name not in QUANTUM_MODEL_REGISTRY:
        return prepared.X_train_classical, prepared.X_test_classical
    return prepared.X_train_quantum, prepared.X_test_quantum


def _learning_rate_for(name: str, cfg: dict[str, Any]) -> float:
    if name == "cnn_classifier":
        return float(cfg["lr_cnn"])
    if name == "photonic_classifier":
        return float(cfg.get("lr_photonic", cfg["lr_qcnn"]))
    return float(cfg["lr_qcnn"])


def _save_curves(
    histories: list[TrainHistory],
    outdir: Path,
    *,
    roc_records: list[tuple[str, int, TrainHistory]] | None = None,
    roc_zoom: dict[str, list[float]] | None = None,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    metric_specs = [
        ("loss", "Train loss"),
        ("accuracy", "Test accuracy"),
        ("precision", "Macro precision"),
        ("recall", "Macro recall"),
        ("time_s", "Epoch wall-clock (s)"),
    ]
    for ax, (key, title) in zip(axes, metric_specs):
        for h in histories:
            xs = list(range(1, len(h.epochs) + 1))
            ys = [getattr(e, key) for e in h.epochs]
            ax.plot(xs, ys, marker="o", label=h.name)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    roc_axis = axes[-1]
    plotted = False
    for model_name, seed, h in roc_records or []:
        if not h.roc_curve:
            continue
        multi_seed_suffix = (
            f" seed={seed}" if len({s for _, s, _ in roc_records or []}) > 1 else ""
        )
        roc_axis.plot(
            h.roc_curve["fpr"],
            h.roc_curve["tpr"],
            marker=".",
            label=f"{model_name}{multi_seed_suffix} (AUC={h.roc_auc:.4f})"
            if h.roc_auc is not None
            else f"{model_name}{multi_seed_suffix}",
        )
        plotted = True
    if plotted:
        if roc_zoom:
            xlim = tuple(float(v) for v in roc_zoom.get("xlim", [0.0, 1.0]))
            ylim = tuple(float(v) for v in roc_zoom.get("ylim", [0.0, 1.0]))
            roc_title = "ROC curve (zoom)"
        else:
            xlim = (0.0, 1.0)
            ylim = (0.0, 1.0)
            roc_title = "ROC curve"
        roc_axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.5", linewidth=1)
        roc_axis.set_title(roc_title)
        roc_axis.set_xlabel("False positive rate")
        roc_axis.set_ylabel("True positive rate")
        roc_axis.set_xlim(*xlim)
        roc_axis.set_ylim(*ylim)
        roc_axis.grid(True, alpha=0.3)
        roc_axis.legend(fontsize=8)
    else:
        roc_axis.axis("off")
    fig.suptitle("QCNN-ID training dynamics (test split metrics)", y=1.02)
    fig.tight_layout()
    path = outdir / "training_curves.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def _save_confusion(histories: list[TrainHistory], outdir: Path) -> Path:
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, h in zip(axes, histories):
        cm = np.array(h.confusion_matrix or [[0, 0], [0, 0]])
        im = ax.imshow(cm, cmap="YlGnBu", vmin=0)
        ax.set_title(h.name)
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        max_value = float(cm.max()) if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = (
                    "white" if max_value and cm[i, j] > 0.55 * max_value else "#1f2933"
                )
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                )
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = outdir / "confusion_matrices.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def _sum_confusion_matrices(histories: list[TrainHistory]) -> list[list[int]]:
    matrices = [
        np.array(h.confusion_matrix or [[0, 0], [0, 0]], dtype=np.int64)
        for h in histories
    ]
    return np.sum(matrices, axis=0).astype(int).tolist()


def _mean_history(name: str, histories: list[TrainHistory]) -> TrainHistory:
    n_epochs = min(len(h.epochs) for h in histories)
    epochs = []
    for idx in range(n_epochs):
        epoch_metrics = [h.epochs[idx] for h in histories]
        epochs.append(
            EpochMetrics(
                loss=float(np.mean([e.loss for e in epoch_metrics])),
                accuracy=float(np.mean([e.accuracy for e in epoch_metrics])),
                precision=float(np.mean([e.precision for e in epoch_metrics])),
                recall=float(np.mean([e.recall for e in epoch_metrics])),
                time_s=float(np.mean([e.time_s for e in epoch_metrics])),
            )
        )
    return TrainHistory(
        name=name,
        epochs=epochs,
        final_accuracy=float(np.mean([h.final_accuracy for h in histories])),
        final_precision=float(np.mean([h.final_precision for h in histories])),
        final_recall=float(np.mean([h.final_recall for h in histories])),
        confusion_matrix=_sum_confusion_matrices(histories),
        param_count=histories[0].param_count,
        train_time_s=float(np.mean([h.train_time_s for h in histories])),
        roc_auc=float(np.mean([h.roc_auc for h in histories if h.roc_auc is not None]))
        if any(h.roc_auc is not None for h in histories)
        else None,
        roc_curve=histories[0].roc_curve if len(histories) == 1 else None,
    )


def _aggregate_histories(
    per_seed: dict[str, list[TrainHistory]],
) -> list[TrainHistory]:
    return [
        _mean_history(name, histories)
        for name, histories in per_seed.items()
        if histories
    ]


def _roc_records(
    per_seed: dict[str, list[TrainHistory]],
    seeds: list[int],
) -> list[tuple[str, int, TrainHistory]]:
    records = []
    for model_name, histories in per_seed.items():
        for seed, history in zip(seeds, histories):
            records.append((model_name, seed, history))
    return records


def _save_predictions(
    per_seed: dict[str, list[TrainHistory]],
    seeds: list[int],
    outdir: Path,
    *,
    split: str,
) -> Path | None:
    rows = []
    for model_name, histories in per_seed.items():
        for seed, history in zip(seeds, histories):
            preds = (
                history.train_predictions
                if split == "train"
                else history.test_predictions
            )
            if not preds:
                continue
            for idx in range(len(preds["y_true"])):
                rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "sample_index": preds["sample_index"][idx],
                        "y_true": preds["y_true"][idx],
                        "y_pred": preds["y_pred"][idx],
                        "logit": preds["logit"][idx],
                        "probability": preds["probability"][idx],
                    }
                )
    if not rows:
        return None
    path = outdir / f"{split}_predictions.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "seed",
                "sample_index",
                "y_true",
                "y_pred",
                "logit",
                "probability",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _aggregate_seeds(per_seed: dict[str, list[TrainHistory]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, histories in per_seed.items():
        accs = [h.final_accuracy for h in histories]
        precs = [h.final_precision for h in histories]
        recs = [h.final_recall for h in histories]
        times = [h.train_time_s for h in histories]
        params = histories[0].param_count
        roc_aucs = [h.roc_auc for h in histories if h.roc_auc is not None]
        entry = {
            "seeds": len(histories),
            "param_count": params,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(statistics.pstdev(accs) if len(accs) > 1 else 0.0),
            "precision_mean": float(np.mean(precs)),
            "recall_mean": float(np.mean(recs)),
            "train_time_mean_s": float(np.mean(times)),
            "confusion_matrix": _sum_confusion_matrices(histories),
        }
        if roc_aucs:
            entry["roc_auc_mean"] = float(np.mean(roc_aucs))
        summary[name] = entry
    return summary


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("QCNN_ID")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    log_file = Path(run_dir) / "run.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)

    with (run_dir / "config_snapshot.json").open("w") as f:
        json.dump(_public_config_snapshot(cfg), f, indent=2, default=str)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device") != "cpu" else "cpu"
    )
    logger.info("Using device: %s", device)

    models_to_run: list[str] = list(cfg["models"])
    seeds: list[int] = [int(s) for s in cfg["seeds"]]

    per_seed_histories: dict[str, list[TrainHistory]] = {m: [] for m in models_to_run}
    for seed in seeds:
        logger.info("=== seed %d ===", seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        prepared = load_and_prepare(cfg, seed=seed)
        logger.info(
            "Loaded data: train=%d test=%d full_dim=%d pca_dim=%d balance=%s",
            prepared.n_train,
            prepared.n_test,
            prepared.n_features_full,
            prepared.n_components,
            prepared.class_balance,
        )
        for model_name in models_to_run:
            torch.manual_seed(seed)
            np.random.seed(seed)
            X_tr, X_te = _features_for(model_name, prepared)
            model = _build_model(
                model_name,
                cfg,
                input_dim=X_tr.shape[1],
                device=device,
            )
            lr = _learning_rate_for(model_name, cfg)
            history = train_and_evaluate_model(
                name=model_name,
                model=model,
                X_train=X_tr,
                y_train=prepared.y_train,
                X_test=X_te,
                y_test=prepared.y_test,
                epochs=int(cfg["epochs"]),
                batch_size=int(cfg["batch_size"]),
                lr=lr,
                device=device,
                logger=logger,
                num_classes=2,
            )
            per_seed_histories[model_name].append(history)
    aggregate_histories = _aggregate_histories(per_seed_histories)
    if aggregate_histories:
        _save_curves(
            aggregate_histories,
            run_dir,
            roc_records=_roc_records(per_seed_histories, seeds),
            roc_zoom=cfg.get("roc_zoom"),
        )
        _save_confusion(aggregate_histories, run_dir)
    train_predictions_path = _save_predictions(
        per_seed_histories, seeds, run_dir, split="train"
    )
    if train_predictions_path is not None:
        logger.info("Wrote train predictions to %s", train_predictions_path)
    test_predictions_path = _save_predictions(
        per_seed_histories, seeds, run_dir, split="test"
    )
    if test_predictions_path is not None:
        logger.info("Wrote test predictions to %s", test_predictions_path)

    metrics_path = run_dir / "metrics.json"
    summary = _aggregate_seeds(per_seed_histories)
    payload = {
        "summary": summary,
        "per_seed": {
            m: [h.to_dict() for h in hs] for m, hs in per_seed_histories.items()
        },
        "data": {
            "subset_size": cfg.get("subset_size"),
            "n_components": cfg["n_components"],
            "test_size": cfg["test_size"],
        },
    }
    with metrics_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote metrics to %s", metrics_path)
    logger.removeHandler(file_handler)
    file_handler.close()
    return payload
