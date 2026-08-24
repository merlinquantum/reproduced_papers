"""Shared-runtime entry point for the Sodar qSVM/qKNN reproduction."""

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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

from .data import PreparedData, load_and_prepare
from .models import MODEL_REGISTRY, make_model
from .training import ModelResult, data_for_model, run_model_variant

ROC_FULL_XLIM = (0.0, 1.0)
ROC_FULL_YLIM = (0.0, 1.02)
ROC_ZOOM_XLIM = (0.0, 0.2)
ROC_ZOOM_YLIM = (0.8, 1.02)
RADAR_METRIC_KEYS = ["accuracy_mean", "precision_mean", "recall_mean", "f1_score_mean"]
RADAR_METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1-score"]
RADAR_SCORE_TICKS = (0.8, 0.9, 0.95, 0.99, 0.999)
RADAR_SCORE_EPS = 1e-4
DECISION_BOUNDARY_CLASS_COLORS = ("#3b6f8f", "#c83355")
DEFAULT_DECISION_BOUNDARY_MODELS = (
    "svm_classical",
    "state_svm_angle",
    "photonic_hybrid_svm_angle",
)


def _portable_source(source: str) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return str(source).replace(str(repo_root) + "/", "")


def _seeds_from_config(cfg: dict[str, Any]) -> list[int]:
    if cfg.get("seeds"):
        return [int(seed) for seed in cfg["seeds"]]
    return [int(cfg.get("seed", 1337))]


def _class_counts(labels: np.ndarray) -> dict[int, int]:
    classes, counts = np.unique(labels, return_counts=True)
    return {int(cls): int(count) for cls, count in zip(classes, counts)}


def _class_balance_from_counts(counts: dict[int, int]) -> dict[int, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {int(cls): float(count / total) for cls, count in counts.items()}


def _format_class_distribution(counts: dict[int, int]) -> str:
    balance = _class_balance_from_counts(counts)
    return ", ".join(
        f"{cls}: {count} ({balance[cls]:.3f})" for cls, count in sorted(counts.items())
    )


def _data_summary(prepared: PreparedData) -> dict[str, Any]:
    train_counts = _class_counts(prepared.y_train)
    test_counts = _class_counts(prepared.y_test)
    all_classes = sorted(set(train_counts) | set(test_counts))
    return {
        "dataset": prepared.dataset,
        "source": _portable_source(prepared.source),
        "n_classes": len(all_classes),
        "classes": all_classes,
        "n_rows_raw": prepared.n_rows_raw,
        "n_rows_clean": prepared.n_rows_clean,
        "n_train": int(len(prepared.y_train)),
        "n_test": int(len(prepared.y_test)),
        "n_features_raw": prepared.n_features_raw,
        "n_features_used": len(prepared.feature_names),
        "feature_names": prepared.feature_names,
        "class_counts_clean": prepared.class_counts_clean,
        "class_counts_train": train_counts,
        "class_counts_test": test_counts,
        "photonic_n_modes": prepared.photonic_preprocessing["n_modes"],
        "photonic_feature_names": prepared.photonic_feature_names,
        "photonic_preprocessing": prepared.photonic_preprocessing,
        "class_balance_clean": prepared.class_balance_clean,
        "class_balance_train": _class_balance_from_counts(train_counts),
        "class_balance_test": _class_balance_from_counts(test_counts),
    }


def _log_data_summary(logger: logging.Logger, summary: dict[str, Any]) -> None:
    logger.info(
        (
            "Loaded %s from %s: rows raw=%d clean=%d train=%d test=%d; "
            "classes=%s; features raw=%d used=%d; photonic_modes=%d"
        ),
        summary["dataset"],
        summary["source"],
        summary["n_rows_raw"],
        summary["n_rows_clean"],
        summary["n_train"],
        summary["n_test"],
        summary["classes"],
        summary["n_features_raw"],
        summary["n_features_used"],
        summary["photonic_n_modes"],
    )
    logger.info(
        "Class distribution clean/subset={%s}; train={%s}; test={%s}",
        _format_class_distribution(summary["class_counts_clean"]),
        _format_class_distribution(summary["class_counts_train"]),
        _format_class_distribution(summary["class_counts_test"]),
    )


def _aggregate_summary(
    per_seed: dict[str, list[ModelResult]],
) -> dict[str, dict[str, Any]]:
    summary = {}

    def _std(values: list[float]) -> float:
        return float(statistics.pstdev(values) if len(values) > 1 else 0.0)

    for name, results in per_seed.items():
        if not results:
            continue
        acc = [r.metrics["accuracy"] for r in results]
        precision = [r.metrics["precision"] for r in results]
        recall = [r.metrics["recall"] for r in results]
        f1 = [r.metrics["f1_score"] for r in results]
        train_times = [r.train_time_s for r in results]
        predict_times = [r.predict_time_s for r in results]
        roc_auc_values = [
            r.metrics["roc_auc"] for r in results if "roc_auc" in r.metrics
        ]
        summary[name] = {
            "seeds": len(results),
            "accuracy_mean": float(np.mean(acc)),
            "accuracy_std": _std(acc),
            "precision_mean": float(np.mean(precision)),
            "precision_std": _std(precision),
            "recall_mean": float(np.mean(recall)),
            "recall_std": _std(recall),
            "f1_score_mean": float(np.mean(f1)),
            "f1_score_std": _std(f1),
            "train_time_mean_s": float(np.mean(train_times)),
            "train_time_std_s": _std(train_times),
            "predict_time_mean_s": float(np.mean(predict_times)),
            "predict_time_std_s": _std(predict_times),
            "n_train": results[0].n_train,
            "n_test": results[0].n_test,
            "n_features": results[0].n_features,
            "n_qubits": results[0].n_qubits,
        }
        if roc_auc_values:
            summary[name]["roc_auc_mean"] = float(np.mean(roc_auc_values))
            summary[name]["roc_auc_std"] = _std(roc_auc_values)
    return summary


def _write_summary_csv(summary: dict[str, dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "model",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_score_mean",
        "f1_score_std",
        "roc_auc_mean",
        "roc_auc_std",
        "train_time_mean_s",
        "train_time_std_s",
        "predict_time_mean_s",
        "predict_time_std_s",
        "n_train",
        "n_test",
        "n_features",
        "n_qubits",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for model, values in summary.items():
            row = {"model": model}
            row.update({key: values.get(key) for key in fieldnames if key != "model"})
            writer.writerow(row)


def _stage_profile_keys(per_seed: dict[str, list[ModelResult]]) -> list[str]:
    keys = set()
    for results in per_seed.values():
        for result in results:
            keys.update(
                key
                for key, value in result.metadata.items()
                if key.startswith("profile_") and isinstance(value, int | float | bool)
            )
    return sorted(keys)


def _write_stage_profile_csv(
    per_seed: dict[str, list[ModelResult]],
    seeds: list[int],
    path: Path,
) -> None:
    """Write per-seed model stage timings extracted from result metadata."""

    profile_keys = _stage_profile_keys(per_seed)
    fieldnames = [
        "model",
        "seed",
        "n_train",
        "n_test",
        "n_features",
        "n_qubits",
        "train_time_s",
        "predict_time_s",
        *profile_keys,
    ]
    rows = []
    for model, results in per_seed.items():
        for seed, result in zip(seeds, results):
            row = {
                "model": model,
                "seed": seed,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "n_features": result.n_features,
                "n_qubits": result.n_qubits,
                "train_time_s": result.train_time_s,
                "predict_time_s": result.predict_time_s,
            }
            row.update({key: result.metadata.get(key) for key in profile_keys})
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(
    per_seed: dict[str, list[ModelResult]],
    seeds: list[int],
    outdir: Path,
    *,
    split: str,
) -> Path:
    fieldnames = ["model", "seed", "sample_index", "y_true", "y_pred", "y_score"]
    rows = []
    for model, results in per_seed.items():
        for seed, result in zip(seeds, results):
            if split == "train":
                indices = result.train_indices
                y_true = result.y_train_true
                y_pred = result.y_train_pred
                y_score = result.y_train_score
            elif split == "test":
                indices = result.test_indices
                y_true = result.y_test_true
                y_pred = result.y_test_pred
                y_score = result.y_test_score
            else:
                raise ValueError("split must be 'train' or 'test'.")
            scores = y_score if y_score is not None else [None] * len(y_true)
            for sample_index, true_label, pred_label, score in zip(
                indices,
                y_true,
                y_pred,
                scores,
            ):
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "sample_index": sample_index,
                        "y_true": true_label,
                        "y_pred": pred_label,
                        "y_score": "" if score is None else score,
                    }
                )
    path = outdir / f"{split}_predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _roc_family(name: str) -> str | None:
    if "svm" in name:
        return "svm"
    if "knn" in name:
        return "knn"
    return None


def _roc_curves(
    per_seed: dict[str, list[ModelResult]],
) -> list[dict[str, Any]]:
    curves = []
    mean_fpr = np.linspace(0.0, 1.0, 101)
    for model, results in per_seed.items():
        family = _roc_family(model)
        if family is None:
            continue
        tpr_parts = []
        auc_values = []
        for result in results:
            if result.y_test_score is None:
                continue
            y_true = np.asarray(result.y_test_true, dtype=int)
            y_score = np.asarray(result.y_test_score, dtype=float)
            if y_score.shape[0] != y_true.shape[0] or len(np.unique(y_true)) != 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tpr[-1] = 1.0
            tpr_parts.append(interp_tpr)
            auc_values.append(float(auc(fpr, tpr)))
        if not tpr_parts:
            continue
        tpr_matrix = np.vstack(tpr_parts)
        mean_tpr = np.mean(tpr_matrix, axis=0)
        std_tpr = np.std(tpr_matrix, axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        curves.append(
            {
                "model": model,
                "family": family,
                "fpr": mean_fpr,
                "tpr": mean_tpr,
                "tpr_std": std_tpr,
                "roc_auc": float(np.mean(auc_values)),
                "roc_auc_std": float(
                    np.std(auc_values) if len(auc_values) > 1 else 0.0
                ),
                "n_seeds": len(tpr_parts),
                "n_test": results[0].n_test,
            }
        )
    return curves


def _write_roc_csv(curves: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "family",
        "model",
        "n_seeds",
        "n_test",
        "roc_auc",
        "roc_auc_std",
        "fpr",
        "tpr",
        "tpr_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for curve in curves:
            for fpr, tpr, tpr_std in zip(
                curve["fpr"],
                curve["tpr"],
                curve["tpr_std"],
            ):
                writer.writerow(
                    {
                        "family": curve["family"],
                        "model": curve["model"],
                        "n_seeds": curve["n_seeds"],
                        "n_test": curve.get("n_test"),
                        "roc_auc": curve["roc_auc"],
                        "roc_auc_std": curve["roc_auc_std"],
                        "fpr": float(fpr),
                        "tpr": float(tpr),
                        "tpr_std": float(tpr_std),
                    }
                )


def _roc_curve_label(curve: dict[str, Any]) -> str:
    auc_std = float(curve.get("roc_auc_std", 0.0))
    if auc_std > 0.0:
        return f"{curve['model']} (AUC={curve['roc_auc']:.3f}+/-{auc_std:.3f})"
    return f"{curve['model']} (AUC={curve['roc_auc']:.3f})"


def _n_test_label_from_values(values: list[Any]) -> str | None:
    n_test_values = sorted({int(value) for value in values if value not in (None, "")})
    if not n_test_values:
        return None
    if len(n_test_values) == 1:
        return f"n_test={n_test_values[0]}"
    return f"n_test={n_test_values[0]}-{n_test_values[-1]}"


def _append_n_test_to_title(title: str, n_test_label: str | None) -> str:
    if not n_test_label:
        return title
    return f"{title}\n({n_test_label})"


def _n_test_label_from_curves(curves: list[dict[str, Any]]) -> str | None:
    return _n_test_label_from_values([curve.get("n_test") for curve in curves])


def _n_test_label_from_summary(
    summary: dict[str, dict[str, Any]],
    *,
    family: str | None = None,
) -> str | None:
    values = []
    for model, model_summary in summary.items():
        if family is not None and _summary_family(model) != family:
            continue
        values.append(model_summary.get("n_test"))
    return _n_test_label_from_values(values)


def _plot_roc_axis(
    ax: plt.Axes,
    curves: list[dict[str, Any]],
    *,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_ylabel: bool,
) -> None:
    for curve in curves:
        fpr = np.asarray(curve["fpr"], dtype=float)
        tpr = np.asarray(curve["tpr"], dtype=float)
        ax.plot(fpr, tpr, linewidth=2.0, label=_roc_curve_label(curve))
        tpr_std = np.asarray(curve.get("tpr_std", []), dtype=float)
        if tpr_std.size == tpr.size and np.any(tpr_std > 0.0):
            lower = np.clip(tpr - tpr_std, 0.0, 1.0)
            upper = np.clip(tpr + tpr_std, 0.0, 1.0)
            ax.fill_between(fpr, lower, upper, alpha=0.12)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("False positive rate")
    if show_ylabel:
        ax.set_ylabel("True positive rate")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)


def _save_roc_plot(
    curves: list[dict[str, Any]],
    path: Path,
    *,
    title: str,
    family: str | None = None,
) -> bool:
    selected = [
        curve for curve in curves if family is None or curve["family"] == family
    ]
    if not selected:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.5))
    _plot_roc_axis(
        axes[0],
        selected,
        title="Full view",
        xlim=ROC_FULL_XLIM,
        ylim=ROC_FULL_YLIM,
        show_ylabel=True,
    )
    _plot_roc_axis(
        axes[1],
        selected,
        title="Zoomed view (FPR <= 0.2, TPR >= 0.8)",
        xlim=ROC_ZOOM_XLIM,
        ylim=ROC_ZOOM_YLIM,
        show_ylabel=False,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(3, len(labels)),
            fontsize=7,
        )
    fig.suptitle(
        _append_n_test_to_title(title, _n_test_label_from_curves(selected)),
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.96))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def _pretty_model_name(name: str) -> str:
    return name.replace("_", "\n")


def _summary_family(name: str) -> str | None:
    if "svm" in name:
        return "svm"
    if "knn" in name:
        return "knn"
    return None


def _radar_score_to_radius(score: float) -> float:
    clipped = min(max(float(score), 0.0), 1.0)
    return float(-np.log10(max(1.0 - clipped, RADAR_SCORE_EPS)))


def _radar_max_radius_from_summary(
    summary: dict[str, dict[str, Any]],
    *,
    family: str | None = None,
) -> float:
    scores = [
        float(values[key])
        for model, values in summary.items()
        if family is None or _summary_family(model) == family
        for key in RADAR_METRIC_KEYS
        if values.get(key) not in (None, "")
    ]
    if not scores:
        return _radar_score_to_radius(1.0)
    max_radius = _radar_score_to_radius(max(scores))
    return min(_radar_score_to_radius(1.0), max_radius * 1.06 + 0.04)


def _style_log_error_radar_axis(
    ax: plt.Axes,
    *,
    max_radius: float | None = None,
) -> None:
    if max_radius is None:
        max_radius = _radar_score_to_radius(1.0)
    tick_scores = [
        score
        for score in RADAR_SCORE_TICKS
        if _radar_score_to_radius(score) <= max_radius
    ]
    if not tick_scores:
        tick_scores = [RADAR_SCORE_TICKS[0]]
    tick_radii = [_radar_score_to_radius(score) for score in tick_scores]
    tick_labels = [f"{score:g}" for score in tick_scores]
    ax.set_ylim(0.0, max_radius)
    ax.set_yticks(tick_radii)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_rlabel_position(140)
    ax.grid(alpha=0.3)


def _save_radar_plot(
    summary: dict[str, dict[str, Any]],
    path: Path,
    *,
    title: str,
    family: str | None = None,
) -> bool:
    selected = [
        (model, values)
        for model, values in summary.items()
        if family is None or _summary_family(model) == family
    ]
    if not selected:
        return False

    angles = np.linspace(0, 2 * np.pi, len(RADAR_METRIC_KEYS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.2, 6.4), subplot_kw={"polar": True})
    for model, values in selected:
        scores = [
            _radar_score_to_radius(float(values[key])) for key in RADAR_METRIC_KEYS
        ]
        scores += scores[:1]
        ax.plot(angles, scores, linewidth=1.8, label=model)
        ax.fill(angles, scores, alpha=0.07)

    ax.set_title(
        _append_n_test_to_title(
            title,
            _n_test_label_from_summary(summary, family=family),
        ),
        pad=18,
    )
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_METRIC_LABELS)
    _style_log_error_radar_axis(
        ax,
        max_radius=_radar_max_radius_from_summary(summary, family=family),
    )
    ax.text(
        0.5,
        -0.11,
        "Radial scale: -log10(1 - score); outer rings are better.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.03), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def _comparison_group_label(model: str) -> str:
    if model in {"svm_classical", "knn_classical"}:
        return "Classical"
    if model.startswith(("state_svm_", "state_knn_")):
        encoding = model.rsplit("_", maxsplit=1)[1]
        return f"State\n{encoding}"
    if model.startswith(("qsvm_", "qknn_")):
        encoding = model.split("_", maxsplit=1)[1]
        return f"Legacy kernel\n{encoding}"
    if model.startswith(("hybrid_svm_", "hybrid_knn_")):
        encoding = model.rsplit("_", maxsplit=1)[1]
        return f"Hybrid\n{encoding}"
    if model.startswith(("photonic_hybrid_svm_", "photonic_hybrid_knn_")):
        encoding = model.rsplit("_", maxsplit=1)[1]
        return f"Photonic hybrid\n{encoding}"
    if model.startswith(("photonic_state_svm_", "photonic_state_knn_")):
        encoding = model.rsplit("_", maxsplit=1)[1]
        return f"Photonic state\n{encoding}"
    if model.startswith(("photonic_fidelity_svm_", "photonic_fidelity_knn_")):
        encoding = model.rsplit("_", maxsplit=1)[1]
        return f"Photonic kernel\n{encoding}"
    return _pretty_model_name(model)


def _dynamic_accuracy_ylim(values: np.ndarray) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    score_range = max_value - min_value
    padding = max(0.01, 0.15 * score_range)
    lower = max(0.0, np.floor((min_value - padding) / 0.02) * 0.02)
    upper = 1.0
    if upper - lower < 0.08:
        lower = max(0.0, upper - 0.08)
    return float(lower), float(upper)


def _save_family_plot(summary: dict[str, dict[str, Any]], path: Path) -> None:
    if not summary:
        return

    grouped: dict[str, dict[str, float]] = {}
    for model, values in summary.items():
        family = _summary_family(model)
        if family is None:
            continue
        label = _comparison_group_label(model)
        grouped.setdefault(label, {})[family] = float(values["accuracy_mean"])
    if not grouped:
        return

    labels = list(grouped)
    x = np.arange(len(labels))
    width = 0.36
    svm_values = np.asarray(
        [grouped[label].get("svm", np.nan) for label in labels],
        dtype=float,
    )
    knn_values = np.asarray(
        [grouped[label].get("knn", np.nan) for label in labels],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(max(8.5, 0.9 * len(labels) + 3.0), 5.2))
    svm_mask = ~np.isnan(svm_values)
    knn_mask = ~np.isnan(knn_values)
    has_svm = bool(svm_mask.any())
    has_knn = bool(knn_mask.any())
    svm_x = x[svm_mask] - width / 2 if has_svm and has_knn else x[svm_mask]
    knn_x = x[knn_mask] + width / 2 if has_svm and has_knn else x[knn_mask]
    if has_svm:
        ax.bar(
            svm_x,
            svm_values[svm_mask],
            width,
            label="SVM",
            color="#2f6f73",
        )
    if has_knn:
        ax.bar(
            knn_x,
            knn_values[knn_mask],
            width,
            label="KNN",
            color="#b36b24",
        )
    title = (
        "SVM vs KNN accuracy by model family"
        if has_svm and has_knn
        else "Model accuracy by family"
    )
    ax.set_title(_append_n_test_to_title(title, _n_test_label_from_summary(summary)))
    ax.set_ylabel("Accuracy")
    ax.set_ylim(*_dynamic_accuracy_ylim(np.concatenate([svm_values, knn_values])))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(axis="y", alpha=0.25)
    if has_svm or has_knn:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _decision_boundary_grid(
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_axis = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(grid_axis, grid_axis)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid


def _stratified_indices(labels: np.ndarray, limit: int, *, seed: int) -> np.ndarray:
    if limit <= 0 or limit >= len(labels):
        return np.arange(len(labels))
    selected, _ = train_test_split(
        np.arange(len(labels)),
        train_size=limit,
        random_state=seed,
        stratify=labels,
    )
    return np.asarray(selected, dtype=int)


def _pca_projection(
    prepared: PreparedData,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_all = np.vstack([prepared.X_train_classical, prepared.X_test_classical])
    reducer = PCA(n_components=2, random_state=seed)
    embedding = np.asarray(
        reducer.fit_transform(X_all),
        dtype=float,
    )
    embedding = MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(embedding)
    embedding = np.clip(embedding, 0.0, 1.0)
    split = len(prepared.y_train)
    return embedding[:split], embedding[split:]


def _probability_like_scores(
    predictions: np.ndarray,
    scores: np.ndarray | None,
) -> np.ndarray:
    """Convert model scores to a [0, 1] positive-class surface for plotting."""

    predictions = np.asarray(predictions, dtype=float)
    if scores is None:
        return predictions

    raw_scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(raw_scores)
    if not finite.any():
        return predictions

    output = raw_scores.copy()
    if np.nanmin(raw_scores[finite]) < 0.0 or np.nanmax(raw_scores[finite]) > 1.0:
        clipped = np.clip(raw_scores[finite], -50.0, 50.0)
        output[finite] = 1.0 / (1.0 + np.exp(-clipped))
    output[~finite] = predictions[~finite]
    return np.clip(output, 0.0, 1.0)


def _model_predictions_on_prepared_data(
    model_name: str,
    prepared: PreparedData,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, _y_test, _train_indices, _test_indices = data_for_model(
        model_name,
        prepared,
    )
    model = make_model(model_name, cfg)
    model.fit(X_train, y_train)
    train_pred, train_score = model.predict_train_with_scores()
    test_pred, test_score = model.predict_with_scores(X_test)
    train_pred = np.asarray(train_pred, dtype=int)
    test_pred = np.asarray(test_pred, dtype=int)
    return (
        train_pred,
        _probability_like_scores(train_pred, train_score),
        test_pred,
        _probability_like_scores(test_pred, test_score),
    )


def _decision_boundary_plot_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return configured PCA prediction-region plots.

    The legacy ``decision_boundary_models`` key keeps producing
    ``pca_prediction_regions.png``. ``extra_decision_boundary_plots`` can add
    focused comparisons without hard-coding filenames in the runner.
    """

    specs: list[dict[str, Any]] = []
    default_models = list(cfg.get("decision_boundary_models") or [])
    if default_models:
        specs.append(
            {
                "filename": "pca_prediction_regions.png",
                "models": default_models,
                "title": "PCA projection of full-model predictions",
            }
        )

    for raw_spec in cfg.get("extra_decision_boundary_plots") or []:
        if not isinstance(raw_spec, dict):
            raise ValueError("Each extra decision-boundary plot must be an object.")
        filename = str(raw_spec.get("filename", "")).strip()
        if not filename:
            raise ValueError("Extra decision-boundary plots require a filename.")
        path = Path(filename)
        if path.is_absolute() or path.name != filename:
            raise ValueError(
                "Extra decision-boundary plot filenames must be simple filenames."
            )
        models = list(raw_spec.get("models") or [])
        if not models:
            raise ValueError(
                f"Extra decision-boundary plot {filename!r} requires models."
            )
        specs.append(
            {
                "filename": filename,
                "models": models,
                "title": raw_spec.get("title"),
            }
        )
    return specs


def _save_decision_boundary_plot(
    prepared: PreparedData,
    cfg: dict[str, Any],
    path: Path,
    *,
    seed: int,
    logger: logging.Logger,
    models: list[str] | None = None,
    title: str | None = None,
) -> bool:
    selected_models = (
        models if models is not None else cfg.get("decision_boundary_models") or []
    )
    models = list(selected_models)
    if not models:
        return False
    unknown = [model for model in models if model not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown decision-boundary model(s) {unknown}; expected {MODEL_REGISTRY}."
        )

    train_embedding, test_embedding = _pca_projection(
        prepared,
        seed=seed,
    )
    test_limit = int(cfg.get("decision_boundary_test_limit", 800))
    test_indices = _stratified_indices(
        prepared.y_test,
        test_limit,
        seed=seed + 1,
    )
    X_test_plot = test_embedding[test_indices]
    y_test_plot = prepared.y_test[test_indices]
    grid_size = int(cfg.get("decision_boundary_grid_size", 70))
    xx, yy, grid = _decision_boundary_grid(grid_size)
    interpolation_neighbors = min(
        int(cfg.get("decision_boundary_neighbors", 75)),
        len(prepared.y_train) + len(prepared.y_test),
    )
    n_jobs = cfg.get("parallel_n_jobs")
    n_jobs = None if n_jobs is None else int(n_jobs)
    background_cmap = ListedColormap(DECISION_BOUNDARY_CLASS_COLORS)
    point_cmap = ListedColormap(DECISION_BOUNDARY_CLASS_COLORS)

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(max(5.2 * len(models), 8.0), 4.8),
        sharex=True,
        sharey=True,
    )
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        logger.info(
            "Fitting PCA prediction-region interpolation for %s: "
            "train=%d test=%d display_test=%d grid=%dx%d k=%d",
            model_name,
            len(prepared.y_train),
            len(prepared.y_test),
            len(y_test_plot),
            grid_size,
            grid_size,
            interpolation_neighbors,
        )
        train_pred, train_score, test_pred, test_score = (
            _model_predictions_on_prepared_data(
                model_name,
                prepared,
                cfg,
            )
        )
        predictor_embedding = np.vstack([train_embedding, test_embedding])
        predictor_scores = np.concatenate([train_score, test_score])
        boundary_model = KNeighborsRegressor(
            n_neighbors=interpolation_neighbors,
            weights=str(cfg.get("decision_boundary_weights", "distance")),
            n_jobs=n_jobs,
        )
        boundary_model.fit(predictor_embedding, predictor_scores)
        zz = np.clip(
            np.asarray(boundary_model.predict(grid), dtype=float).reshape(xx.shape),
            0.0,
            1.0,
        )
        zz_class = (zz >= 0.5).astype(float)
        ax.contourf(
            xx,
            yy,
            zz_class,
            levels=[-0.5, 0.5, 1.5],
            alpha=0.24,
            cmap=background_cmap,
        )
        ax.scatter(
            X_test_plot[:, 0],
            X_test_plot[:, 1],
            c=y_test_plot,
            cmap=point_cmap,
            vmin=0,
            vmax=1,
            s=12,
            alpha=0.72,
            edgecolors="none",
        )
        ax.set_title(model_name.replace("_", "\n"))
        ax.set_xlabel("PCA component 1, MinMax scaled")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("PCA component 2, MinMax scaled")
    legend_handles = [
        Patch(facecolor=DECISION_BOUNDARY_CLASS_COLORS[0], label="Benign"),
        Patch(facecolor=DECISION_BOUNDARY_CLASS_COLORS[1], label="Malicious"),
    ]
    axes[-1].legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        title="Test points",
        title_fontsize=8,
    )
    plot_title = title or "PCA projection of full-model predictions"
    fig.suptitle(
        f"{plot_title} (n_train={len(prepared.y_train)}, "
        f"n_test_display={len(y_test_plot)})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run configured qSVM/qKNN experiments.

    Parameters
    ----------
    cfg : dict
        Resolved shared-runtime configuration.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Returns
    -------
    dict
        Metrics payload written to ``metrics.json``.
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("qSVM_qKNN")
    logger.info("Starting qSVM_qKNN run")

    models_to_run = list(cfg.get("models", []))
    unknown = [name for name in models_to_run if name not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; expected {MODEL_REGISTRY}.")
    seeds = _seeds_from_config(cfg)
    per_seed: dict[str, list[ModelResult]] = {name: [] for name in models_to_run}
    data_runs = []
    decision_boundary_written = False
    decision_boundary_specs = _decision_boundary_plot_specs(cfg)

    for seed in seeds:
        logger.info("=== seed %d ===", seed)
        prepared = load_and_prepare(cfg, seed=seed)
        data_summary = _data_summary(prepared)
        data_runs.append(data_summary)
        _log_data_summary(logger, data_summary)
        if not decision_boundary_written and decision_boundary_specs:
            for plot_spec in decision_boundary_specs:
                _save_decision_boundary_plot(
                    prepared,
                    cfg,
                    run_dir / str(plot_spec["filename"]),
                    seed=seed,
                    logger=logger,
                    models=list(plot_spec["models"]),
                    title=plot_spec.get("title"),
                )
            decision_boundary_written = True
        feature_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        for model_name in models_to_run:
            result = run_model_variant(
                model_name,
                prepared,
                cfg,
                seed=seed,
                logger=logger,
                feature_cache=feature_cache,
            )
            per_seed[model_name].append(result)
            logger.info(
                "%s accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
                model_name,
                result.metrics["accuracy"],
                result.metrics["precision"],
                result.metrics["recall"],
                result.metrics["f1_score"],
            )

    summary = _aggregate_summary(per_seed)
    payload = {
        "summary": summary,
        "per_seed": {
            model: [result.to_dict() for result in results]
            for model, results in per_seed.items()
        },
        "data": data_runs,
        "config": {
            "dataset": cfg.get("dataset"),
            "subset_size": cfg.get("subset_size"),
            "max_test_size": cfg.get("max_test_size"),
            "feature_limit": cfg.get("feature_limit"),
            "photonic_n_modes": cfg.get("photonic_n_modes"),
            "photonic_n_photons": cfg.get("photonic_n_photons"),
            "photonic_computation_space": cfg.get("photonic_computation_space"),
            "zz_reps": cfg.get("zz_reps"),
            "encoder_batch_size": cfg.get("encoder_batch_size"),
            "parallel_n_jobs": cfg.get("parallel_n_jobs"),
            "parallel_chunk_size": cfg.get("parallel_chunk_size"),
            "parallel_min_rows": cfg.get("parallel_min_rows"),
            "decision_boundary_models": cfg.get("decision_boundary_models"),
            "extra_decision_boundary_plots": cfg.get("extra_decision_boundary_plots"),
            "decision_boundary_test_limit": cfg.get("decision_boundary_test_limit"),
            "decision_boundary_grid_size": cfg.get("decision_boundary_grid_size"),
            "decision_boundary_neighbors": cfg.get("decision_boundary_neighbors"),
            "decision_boundary_weights": cfg.get("decision_boundary_weights"),
            "models": models_to_run,
            "seeds": seeds,
        },
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    _write_summary_csv(summary, run_dir / "summary.csv")
    _write_stage_profile_csv(per_seed, seeds, run_dir / "stage_profile.csv")
    _write_predictions(per_seed, seeds, run_dir, split="train")
    _write_predictions(per_seed, seeds, run_dir, split="test")
    _save_family_plot(summary, run_dir / "svm_knn_comparison.png")
    _save_radar_plot(
        summary,
        run_dir / "svm_radar_chart.png",
        title="SVM metric radar chart",
        family="svm",
    )
    _save_radar_plot(
        summary,
        run_dir / "knn_radar_chart.png",
        title="KNN metric radar chart",
        family="knn",
    )
    curves = _roc_curves(per_seed)
    if curves:
        _write_roc_csv(curves, run_dir / "roc_curves.csv")
        _save_roc_plot(
            curves,
            run_dir / "svm_roc_curves.png",
            title="SVM ROC curves",
            family="svm",
        )
        _save_roc_plot(
            curves,
            run_dir / "knn_roc_curves.png",
            title="KNN ROC curves",
            family="knn",
        )
    logger.info("Wrote qSVM_qKNN artifacts to %s", run_dir)
    return payload
