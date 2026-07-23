"""Curate qSVM runs into README- and notebook-ready artifacts."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PAPER_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = PAPER_ROOT / "outdir"
RESULTS = PAPER_ROOT / "results"
sys.path.insert(0, str(PAPER_ROOT))
from lib.runner import (  # noqa: E402
    RADAR_METRIC_KEYS,
    RADAR_METRIC_LABELS,
    _comparison_group_label,
    _dynamic_accuracy_ylim,
    _radar_max_radius_from_summary,
    _radar_score_to_radius,
    _save_family_plot,
    _save_radar_plot,
    _save_roc_plot,
    _style_log_error_radar_axis,
    _summary_family,
)

ROC_FULL_XLIM = (0.0, 1.0)
ROC_FULL_YLIM = (0.0, 1.02)
ROC_ZOOM_XLIM = (0.0, 0.2)
ROC_ZOOM_YLIM = (0.8, 1.02)
COMPARISON_EXPERIMENTS = ("all_models", "fast_models", "merlin")
EXTRA_EXPERIMENTS = ("original", "notebook", "kernel_vs_state_fidelity")
EXTRA_RADAR_EXPERIMENTS = ("original",)
EXTRA_FAMILY_GRID_EXPERIMENTS = ("original",)


def _family_from_models(models: list[str], *, experiment: str | None = None) -> str:
    if all("svm" in model for model in models):
        return "svm"
    if all("knn" in model for model in models):
        return "knn"
    if experiment == "all_models":
        return "all_models"
    if experiment == "fast_models":
        return "fast_models"
    if experiment == "merlin":
        return "photonic"
    if experiment == "original":
        return "gate_based"
    if experiment == "notebook":
        return "explicit_features"
    if experiment == "kernel_vs_state_fidelity":
        return "kernel_validation"
    return "explicit_features"


def _experiment_from_config(cfg: dict[str, Any]) -> str | None:
    description = str(cfg.get("description", "")).lower()
    if description.startswith("all-model"):
        return "all_models"
    if description.startswith("fast-model"):
        return "fast_models"
    if description.startswith("merlin"):
        return "merlin"
    if description.startswith("original-style"):
        return "original"
    if description.startswith("notebook pedagogical"):
        return "notebook"
    if "validation run comparing fidelity-kernel" in description:
        return "kernel_vs_state_fidelity"
    return None


def _artifact_name(run_dir: Path, cfg: dict[str, Any]) -> str | None:
    experiment = _experiment_from_config(cfg)
    if experiment is None:
        return None
    dataset = str(cfg["dataset"]).lower()
    family = _family_from_models(list(cfg.get("models", [])), experiment=experiment)
    if experiment == "original":
        return f"original_{dataset}_gate_based"
    if experiment == "notebook":
        return f"notebook_{dataset}"
    if experiment == "kernel_vs_state_fidelity":
        return f"kernel_vs_state_fidelity_{dataset}"
    if experiment in {"all_models", "fast_models", "merlin"}:
        return f"{experiment}_{dataset}"
    return f"{experiment}_{dataset}_{family}"


def _experiment_from_artifact(artifact: str) -> str | None:
    if artifact.startswith("all_models_"):
        return "all_models"
    if artifact.startswith("fast_models_"):
        return "fast_models"
    if artifact.startswith("merlin_"):
        return "merlin"
    if artifact.startswith("original_"):
        return "original"
    if artifact.startswith("notebook_"):
        return "notebook"
    if artifact.startswith("kernel_vs_state_fidelity_"):
        return "kernel_vs_state_fidelity"
    return None


def _run_payload(
    artifact: str,
    run_dir: Path,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    experiment: str,
    source: str,
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "run_dir": run_dir,
        "config": cfg,
        "metrics": metrics,
        "experiment": experiment,
        "dataset": str(cfg["dataset"]).lower(),
        "family": _family_from_models(
            list(cfg.get("models", [])),
            experiment=experiment,
        ),
        "source": source,
    }


def _load_runs() -> dict[str, dict[str, Any]]:
    runs = {}
    for run_dir in sorted(OUTDIR.glob("run_*")):
        cfg_path = run_dir / "config_snapshot.json"
        metrics_path = run_dir / "metrics.json"
        if not cfg_path.exists() or not metrics_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        experiment = _experiment_from_config(cfg)
        artifact = _artifact_name(run_dir, cfg)
        if artifact is None:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs[artifact] = _run_payload(
            artifact,
            run_dir,
            cfg,
            metrics,
            str(experiment),
            "outdir",
        )
    return dict(sorted(runs.items()))


def _load_curated_runs() -> dict[str, dict[str, Any]]:
    runs = {}
    for artifact_dir in sorted(RESULTS.iterdir() if RESULTS.exists() else []):
        if not artifact_dir.is_dir():
            continue
        experiment = _experiment_from_artifact(artifact_dir.name)
        if experiment is None:
            continue
        cfg_path = artifact_dir / "config_snapshot.json"
        metrics_path = artifact_dir / "metrics.json"
        if not cfg_path.exists() or not metrics_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs[artifact_dir.name] = _run_payload(
            artifact_dir.name,
            artifact_dir,
            cfg,
            metrics,
            experiment,
            "results",
        )
    return dict(sorted(runs.items()))


def _copy_runs(runs: dict[str, dict[str, Any]]) -> None:
    for artifact, run in runs.items():
        if run.get("source") != "outdir":
            continue
        target = RESULTS / artifact
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(run["run_dir"], target)


def _clean_managed_results() -> None:
    for experiment in COMPARISON_EXPERIMENTS:
        for suffix in (
            "results_table.csv",
            "radar_chart.png",
            "roc_curves.png",
            "runtime_profile.png",
        ):
            path = RESULTS / f"{experiment}_{suffix}"
            if path.exists():
                path.unlink()
    for experiment in EXTRA_EXPERIMENTS:
        path = RESULTS / f"{experiment}_results_table.csv"
        if path.exists():
            path.unlink()
    for experiment in EXTRA_RADAR_EXPERIMENTS:
        path = RESULTS / f"{experiment}_radar_chart.png"
        if path.exists():
            path.unlink()
    for experiment in EXTRA_FAMILY_GRID_EXPERIMENTS:
        path = RESULTS / f"{experiment}_svm_knn_comparison.png"
        if path.exists():
            path.unlink()
    for path in (
        RESULTS / "kernel_vs_state_fidelity_runtime_table.csv",
        RESULTS / "kernel_vs_state_fidelity_speedup_table.csv",
    ):
        if path.exists():
            path.unlink()
    for path in (RESULTS / "results_table.csv", RESULTS / "latest_summary.json"):
        if path.exists():
            path.unlink()


def _metric_rows(
    runs: dict[str, dict[str, Any]], experiment: str
) -> list[dict[str, Any]]:
    rows = []
    for artifact, run in runs.items():
        if run["experiment"] != experiment:
            continue
        summary = run["metrics"]["summary"]
        for model, values in summary.items():
            rows.append(
                {
                    "experiment": experiment.upper(),
                    "artifact": artifact,
                    "dataset": run["dataset"],
                    "family": "svm" if "svm" in model else "knn",
                    "model": model,
                    "seeds": values.get("seeds"),
                    "accuracy": values.get("accuracy_mean"),
                    "accuracy_std": values.get("accuracy_std"),
                    "precision": values.get("precision_mean"),
                    "precision_std": values.get("precision_std"),
                    "recall": values.get("recall_mean"),
                    "recall_std": values.get("recall_std"),
                    "f1_score": values.get("f1_score_mean"),
                    "f1_score_std": values.get("f1_score_std"),
                    "roc_auc": values.get("roc_auc_mean"),
                    "roc_auc_std": values.get("roc_auc_std"),
                    "train_time_s": values.get("train_time_mean_s"),
                    "train_time_std_s": values.get("train_time_std_s"),
                    "predict_time_s": values.get("predict_time_mean_s"),
                    "predict_time_std_s": values.get("predict_time_std_s"),
                    "n_train": values.get("n_train"),
                    "n_test": values.get("n_test"),
                    "n_features": values.get("n_features"),
                    "n_qubits": values.get("n_qubits"),
                }
            )
    return rows


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary_by_panel(
    runs: dict[str, dict[str, Any]],
    experiment: str,
    dataset: str,
    family: str,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for run in runs.values():
        if run["experiment"] != experiment or run["dataset"] != dataset:
            continue
        for model, values in run["metrics"]["summary"].items():
            if family in model:
                selected[model] = values
    return selected


def _n_test_label_from_summary(summary: dict[str, dict[str, Any]]) -> str | None:
    n_test_values = sorted(
        {
            int(values["n_test"])
            for values in summary.values()
            if values.get("n_test") not in (None, "")
        }
    )
    if not n_test_values:
        return None
    if len(n_test_values) == 1:
        return f"n_test={n_test_values[0]}"
    return f"n_test={n_test_values[0]}-{n_test_values[-1]}"


def _append_n_test_to_title(title: str, summary: dict[str, dict[str, Any]]) -> str:
    n_test_label = _n_test_label_from_summary(summary)
    if not n_test_label:
        return title
    return f"{title}\n({n_test_label})"


def _float_or_default(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _save_radar_grid(
    runs: dict[str, dict[str, Any]],
    experiment: str,
    path: Path,
) -> None:
    panels = [
        ("url2016", "svm", "URL2016 SVM"),
        ("url2016", "knn", "URL2016 KNN"),
        ("ids2012", "svm", "IDS2012 SVM"),
        ("ids2012", "knn", "IDS2012 KNN"),
    ]
    angles = np.linspace(0, 2 * np.pi, len(RADAR_METRIC_KEYS), endpoint=False).tolist()
    angles += angles[:1]
    legend_labels: list[str] = []
    for dataset, family, _title in panels:
        summary = _summary_by_panel(runs, experiment, dataset, family)
        for model in summary:
            label = _comparison_group_label(model).replace("\n", " ")
            if label not in legend_labels:
                legend_labels.append(label)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_by_label = {
        label: colors[index % len(colors)] for index, label in enumerate(legend_labels)
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), subplot_kw={"polar": True})
    for ax, (dataset, family, title) in zip(axes.ravel(), panels):
        summary = _summary_by_panel(runs, experiment, dataset, family)
        if not summary:
            ax.axis("off")
            continue
        for model, values in summary.items():
            label = _comparison_group_label(model).replace("\n", " ")
            scores = [
                _radar_score_to_radius(float(values[key])) for key in RADAR_METRIC_KEYS
            ]
            scores += scores[:1]
            ax.plot(
                angles,
                scores,
                linewidth=1.5,
                label=label,
                color=color_by_label[label],
            )
            ax.fill(angles, scores, alpha=0.035, color=color_by_label[label])
        ax.set_title(_append_n_test_to_title(title, summary), pad=15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_METRIC_LABELS)
        _style_log_error_radar_axis(
            ax,
            max_radius=_radar_max_radius_from_summary(summary),
        )
    fig.text(
        0.5,
        0.025,
        "Radial scale: -log10(1 - score); outer rings are better.",
        ha="center",
        fontsize=9,
    )
    if legend_labels:
        handles = [
            Line2D([0], [0], color=color_by_label[label], linewidth=1.8)
            for label in legend_labels
        ]
        fig.legend(
            handles,
            legend_labels,
            title="Model family",
            loc="center left",
            bbox_to_anchor=(1.0, 0.54),
            fontsize=7,
            title_fontsize=8,
        )
    fig.suptitle(f"{experiment.upper()} metric radar charts", y=0.99)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.98))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _family_groups_for_dataset(
    runs: dict[str, dict[str, Any]],
    *,
    experiment: str,
    dataset: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, float]] = {}
    selected: dict[str, dict[str, Any]] = {}
    for run in runs.values():
        if run["experiment"] != experiment or run["dataset"] != dataset:
            continue
        for model, values in run["metrics"]["summary"].items():
            family = _summary_family(model)
            if family is None:
                continue
            label = _comparison_group_label(model)
            grouped.setdefault(label, {})[family] = float(values["accuracy_mean"])
            selected[model] = values
    return grouped, selected


def _save_family_grid(
    runs: dict[str, dict[str, Any]],
    experiment: str,
    path: Path,
) -> None:
    panels = [("url2016", "URL2016"), ("ids2012", "IDS2012")]
    grouped_by_dataset = {}
    summaries_by_dataset = {}
    ordered_labels: list[str] = []
    for dataset, _title in panels:
        grouped, summary = _family_groups_for_dataset(
            runs,
            experiment=experiment,
            dataset=dataset,
        )
        grouped_by_dataset[dataset] = grouped
        summaries_by_dataset[dataset] = summary
        for label in grouped:
            if label not in ordered_labels:
                ordered_labels.append(label)

    if not ordered_labels:
        return

    x = np.arange(len(ordered_labels))
    width = 0.36
    fig, axes = plt.subplots(
        1, 2, figsize=(max(13.5, 1.0 * len(ordered_labels) + 8), 5.7)
    )
    for ax, (dataset, title) in zip(axes, panels):
        grouped = grouped_by_dataset[dataset]
        svm_values = np.asarray(
            [grouped.get(label, {}).get("svm", np.nan) for label in ordered_labels],
            dtype=float,
        )
        knn_values = np.asarray(
            [grouped.get(label, {}).get("knn", np.nan) for label in ordered_labels],
            dtype=float,
        )
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
        ax.set_title(_append_n_test_to_title(title, summaries_by_dataset[dataset]))
        ax.set_ylabel("Accuracy")
        ax.set_ylim(*_dynamic_accuracy_ylim(np.concatenate([svm_values, knn_values])))
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"{experiment.upper()} SVM/KNN accuracy comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _read_roc_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    path = run["run_dir"] / "roc_curves.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _plot_roc_panel(
    ax: plt.Axes,
    runs: dict[str, dict[str, Any]],
    *,
    experiment: str,
    dataset: str,
    family: str,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_xlabel: bool,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    any_curve = False
    for run in runs.values():
        if run["experiment"] != experiment or run["dataset"] != dataset:
            continue
        rows = [
            row
            for row in _read_roc_rows(run)
            if row["family"] == family and family in row["model"]
        ]
        by_model: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_model.setdefault(row["model"], []).append(row)
        for model, model_rows in by_model.items():
            fpr = [float(row["fpr"]) for row in model_rows]
            tpr = [float(row["tpr"]) for row in model_rows]
            tpr_std = [_float_or_default(row.get("tpr_std"), 0.0) for row in model_rows]
            auc_value = float(model_rows[0]["roc_auc"])
            auc_std = _float_or_default(model_rows[0].get("roc_auc_std"), 0.0)
            if auc_std > 0.0:
                label = f"{model} ({auc_value:.3f}+/-{auc_std:.3f})"
            else:
                label = f"{model} ({auc_value:.3f})"
            ax.plot(fpr, tpr, linewidth=1.5, label=label)
            if any(value > 0.0 for value in tpr_std):
                tpr_array = np.asarray(tpr, dtype=float)
                std_array = np.asarray(tpr_std, dtype=float)
                lower = np.clip(tpr_array - std_array, 0.0, 1.0)
                upper = np.clip(tpr_array + std_array, 0.0, 1.0)
                ax.fill_between(fpr, lower, upper, alpha=0.10)
            any_curve = True
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    if show_xlabel:
        ax.set_xlabel("False positive rate")
    if show_ylabel:
        ax.set_ylabel("True positive rate")
    if any_curve and show_legend:
        ax.legend(
            fontsize=6,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
        )


def _save_roc_grid(
    runs: dict[str, dict[str, Any]],
    experiment: str,
    path: Path,
) -> None:
    panels = [
        ("url2016", "svm", "URL2016 SVM"),
        ("url2016", "knn", "URL2016 KNN"),
        ("ids2012", "svm", "IDS2012 SVM"),
        ("ids2012", "knn", "IDS2012 KNN"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(16.5, 17.5), sharex=False, sharey=False)
    for row_idx, (dataset, family, title) in enumerate(panels):
        summary = _summary_by_panel(runs, experiment, dataset, family)
        _plot_roc_panel(
            axes[row_idx, 0],
            runs,
            experiment=experiment,
            dataset=dataset,
            family=family,
            title=_append_n_test_to_title(f"{title} - full view", summary),
            xlim=ROC_FULL_XLIM,
            ylim=ROC_FULL_YLIM,
            show_xlabel=row_idx == len(panels) - 1,
            show_ylabel=True,
            show_legend=False,
        )
        _plot_roc_panel(
            axes[row_idx, 1],
            runs,
            experiment=experiment,
            dataset=dataset,
            family=family,
            title=_append_n_test_to_title(
                f"{title} - zoomed view\n(FPR <= 0.2, TPR >= 0.8)",
                summary,
            ),
            xlim=ROC_ZOOM_XLIM,
            ylim=ROC_ZOOM_YLIM,
            show_xlabel=row_idx == len(panels) - 1,
            show_ylabel=False,
            show_legend=True,
        )
    fig.suptitle(f"{experiment.upper()} ROC curves: full and zoomed views", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_runtime_grid(
    runs: dict[str, dict[str, Any]],
    experiment: str,
    path: Path,
) -> None:
    panels = [
        ("url2016", "svm", "URL2016 SVM"),
        ("url2016", "knn", "URL2016 KNN"),
        ("ids2012", "svm", "IDS2012 SVM"),
        ("ids2012", "knn", "IDS2012 KNN"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.5), sharex=False)
    for ax, (dataset, family, title) in zip(axes.ravel(), panels):
        summary = _summary_by_panel(runs, experiment, dataset, family)
        if not summary:
            ax.axis("off")
            continue
        rows = []
        for model, values in summary.items():
            train_time = _float_or_default(values.get("train_time_mean_s"), 0.0)
            predict_time = _float_or_default(values.get("predict_time_mean_s"), 0.0)
            rows.append((model, train_time, predict_time, train_time + predict_time))
        rows.sort(key=lambda item: item[3], reverse=True)
        y = np.arange(len(rows))
        train_times = [row[1] for row in rows]
        predict_times = [row[2] for row in rows]
        labels = [row[0] for row in rows]
        ax.barh(y, train_times, color="#3b6f8f", label="fit/features")
        ax.barh(
            y,
            predict_times,
            left=train_times,
            color="#b87533",
            label="predict",
        )
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Mean wall-clock time per seed (s)")
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
    axes[0, 1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{experiment.upper()} runtime profiles", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_latest_summary(runs: dict[str, dict[str, Any]]) -> None:
    payload = {}
    for artifact, run in runs.items():
        cfg = run["config"]
        data = run["metrics"].get("data", [{}])[0]
        payload[artifact] = {
            "experiment": run["experiment"].upper(),
            "dataset": run["dataset"],
            "family": run["family"],
            "artifact": artifact,
            "description": cfg.get("description"),
            "subset_size": cfg.get("subset_size"),
            "feature_limit": cfg.get("feature_limit"),
            "n_train": data.get("n_train"),
            "n_test": data.get("n_test"),
            "class_counts_train": data.get("class_counts_train"),
            "class_counts_test": data.get("class_counts_test"),
            "models": cfg.get("models"),
            "seeds": cfg.get("seeds"),
        }
    (RESULTS / "latest_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _mean_total_time(values: dict[str, Any]) -> float:
    return _float_or_default(values.get("train_time_mean_s")) + _float_or_default(
        values.get("predict_time_mean_s")
    )


def _state_counterpart(model: str) -> str | None:
    if model.startswith("qsvm_"):
        return "state_svm_" + model.removeprefix("qsvm_")
    if model.startswith("qknn_"):
        return "state_knn_" + model.removeprefix("qknn_")
    if model.startswith("photonic_fidelity_svm_"):
        return "photonic_state_svm_" + model.removeprefix("photonic_fidelity_svm_")
    if model.startswith("photonic_fidelity_knn_"):
        return "photonic_state_knn_" + model.removeprefix("photonic_fidelity_knn_")
    return None


def _prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prediction_map(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    mapped = {}
    for row in rows:
        mapped[
            (
                str(row["model"]),
                int(row["seed"]),
                int(row["sample_index"]),
            )
        ] = row
    return mapped


def _score_differences(
    kernel_rows: list[dict[str, Any]],
    state_rows: dict[tuple[str, int, int], dict[str, Any]],
    *,
    state_model: str,
) -> list[float]:
    differences = []
    for row in kernel_rows:
        state_row = state_rows.get(
            (
                state_model,
                int(row["seed"]),
                int(row["sample_index"]),
            )
        )
        if state_row is None:
            continue
        if row.get("y_score") in ("", None) or state_row.get("y_score") in ("", None):
            continue
        differences.append(abs(float(row["y_score"]) - float(state_row["y_score"])))
    return differences


def _prediction_agreement_rows(
    run: dict[str, Any], target: Path
) -> list[dict[str, Any]]:
    summary = run["metrics"].get("summary", {})
    all_rows = []
    for split in ("train", "test"):
        rows = _prediction_rows(target / f"{split}_predictions.csv")
        if not rows:
            continue
        mapped = _prediction_map(rows)
        for kernel_model in summary:
            state_model = _state_counterpart(kernel_model)
            if state_model is None or state_model not in summary:
                continue
            kernel_rows = [row for row in rows if row["model"] == kernel_model]
            state_keys = {
                (int(row["seed"]), int(row["sample_index"]))
                for row in rows
                if row["model"] == state_model
            }
            common_rows = [
                row
                for row in kernel_rows
                if (int(row["seed"]), int(row["sample_index"])) in state_keys
            ]
            prediction_matches = 0
            true_label_mismatches = 0
            for row in common_rows:
                state_row = mapped[
                    (
                        state_model,
                        int(row["seed"]),
                        int(row["sample_index"]),
                    )
                ]
                prediction_matches += int(row["y_pred"] == state_row["y_pred"])
                true_label_mismatches += int(row["y_true"] != state_row["y_true"])
            score_diffs = _score_differences(
                common_rows,
                mapped,
                state_model=state_model,
            )
            n_common = len(common_rows)
            n_kernel = len(kernel_rows)
            n_state = sum(1 for row in rows if row["model"] == state_model)
            all_rows.append(
                {
                    "artifact": run["artifact"],
                    "dataset": run["dataset"],
                    "split": split,
                    "kernel_model": kernel_model,
                    "state_model": state_model,
                    "n_kernel_predictions": n_kernel,
                    "n_state_predictions": n_state,
                    "n_common_predictions": n_common,
                    "n_missing_from_state": n_kernel - n_common,
                    "n_extra_in_state": n_state - n_common,
                    "true_label_mismatches": true_label_mismatches,
                    "prediction_matches": prediction_matches,
                    "prediction_disagreements": n_common - prediction_matches,
                    "prediction_agreement": (
                        prediction_matches / n_common if n_common else ""
                    ),
                    "score_count": len(score_diffs),
                    "max_abs_score_diff": max(score_diffs) if score_diffs else "",
                    "mean_abs_score_diff": (
                        float(np.mean(score_diffs)) if score_diffs else ""
                    ),
                }
            )
    return all_rows


def _write_prediction_agreement_json(rows: list[dict[str, Any]], path: Path) -> None:
    payload = {
        "metric": "prediction_agreement",
        "definition": (
            "Fraction of shared sample predictions where the fidelity-kernel "
            "model and its explicit state-fidelity counterpart predict the "
            "same class."
        ),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_prediction_agreement_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    pair_order: list[tuple[str, str]] = []
    for row in rows:
        pair = (str(row["kernel_model"]), str(row["state_model"]))
        if pair not in pair_order:
            pair_order.append(pair)

    labels = [f"{kernel}\nvs\n{state}" for kernel, state in pair_order]
    train_values = []
    test_values = []
    train_n = []
    test_n = []
    by_key = {
        (row["kernel_model"], row["state_model"], row["split"]): row for row in rows
    }
    for kernel, state in pair_order:
        train = by_key.get((kernel, state, "train"))
        test = by_key.get((kernel, state, "test"))
        train_values.append(
            float(train["prediction_agreement"]) if train is not None else np.nan
        )
        test_values.append(
            float(test["prediction_agreement"]) if test is not None else np.nan
        )
        train_n.append(int(train["n_common_predictions"]) if train is not None else 0)
        test_n.append(int(test["n_common_predictions"]) if test is not None else 0)

    values = np.asarray(train_values + test_values, dtype=float)
    finite_values = values[np.isfinite(values)]
    ymin = 0.9 if finite_values.size and float(np.min(finite_values)) >= 0.9 else 0.0
    x = np.arange(len(pair_order))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10.0, 1.05 * len(pair_order) + 3.0), 5.8))
    train_bars = ax.bar(
        x - width / 2,
        train_values,
        width,
        label="Train",
        color="#2f6f73",
    )
    test_bars = ax.bar(
        x + width / 2,
        test_values,
        width,
        label="Test",
        color="#b36b24",
    )
    for bars, counts in ((train_bars, train_n), (test_bars, test_n)):
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if not np.isfinite(height):
                continue
            ax.annotate(
                f"n={count}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    ax.set_title("Kernel vs state prediction agreement")
    ax.set_ylabel("Prediction agreement")
    ax.set_ylim(ymin, 1.015)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_kernel_validation_artifacts(runs: dict[str, dict[str, Any]]) -> None:
    global_runtime_rows = []
    global_speedup_rows = []
    for artifact, run in runs.items():
        if run["experiment"] != "kernel_vs_state_fidelity":
            continue
        target = RESULTS / artifact
        summary = run["metrics"]["summary"]
        runtime_rows = []
        for model, values in summary.items():
            runtime_rows.append(
                {
                    "dataset": run["dataset"],
                    "model": model,
                    "family": "svm" if "svm" in model else "knn",
                    "train_time_s": values.get("train_time_mean_s"),
                    "predict_time_s": values.get("predict_time_mean_s"),
                    "total_time_s": _mean_total_time(values),
                }
            )
        _write_table(runtime_rows, target / "runtime_table.csv")
        global_runtime_rows.extend(runtime_rows)

        speedup_rows = []
        for kernel_model, kernel_values in summary.items():
            state_model = _state_counterpart(kernel_model)
            if state_model is None or state_model not in summary:
                continue
            state_values = summary[state_model]
            kernel_time = _mean_total_time(kernel_values)
            state_time = _mean_total_time(state_values)
            speedup_rows.append(
                {
                    "dataset": run["dataset"],
                    "kernel_model": kernel_model,
                    "state_model": state_model,
                    "accuracy_match": abs(
                        float(kernel_values["accuracy_mean"])
                        - float(state_values["accuracy_mean"])
                    )
                    < 1e-12,
                    "kernel_total_time_s": kernel_time,
                    "state_total_time_s": state_time,
                    "speedup": kernel_time / state_time if state_time else "",
                }
            )
        _write_table(speedup_rows, target / "state_fidelity_speedup.csv")
        global_speedup_rows.extend(speedup_rows)

        rows = sorted(runtime_rows, key=lambda row: row["total_time_s"], reverse=True)
        if rows:
            labels = [str(row["model"]) for row in rows]
            train_times = [_float_or_default(row["train_time_s"]) for row in rows]
            predict_times = [_float_or_default(row["predict_time_s"]) for row in rows]
            y = np.arange(len(rows))
            fig, ax = plt.subplots(figsize=(9.0, max(4.5, 0.35 * len(rows) + 1.8)))
            ax.barh(y, train_times, color="#3b6f8f", label="fit/features")
            ax.barh(
                y, predict_times, left=train_times, color="#b87533", label="predict"
            )
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("Wall-clock time (s)")
            ax.set_title(f"{artifact} runtime profile")
            ax.grid(axis="x", alpha=0.25)
            ax.invert_yaxis()
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(target / "runtime_profile.png", dpi=140, bbox_inches="tight")
            plt.close(fig)

        agreement_rows = _prediction_agreement_rows(run, target)
        _write_table(agreement_rows, target / "predictions_agreement.csv")
        _write_prediction_agreement_json(
            agreement_rows,
            target / "predictions_agreement.json",
        )
        _save_prediction_agreement_plot(
            agreement_rows,
            target / "predictions_agreement.png",
        )
    _write_table(
        global_runtime_rows,
        RESULTS / "kernel_vs_state_fidelity_runtime_table.csv",
    )
    _write_table(
        global_speedup_rows,
        RESULTS / "kernel_vs_state_fidelity_speedup_table.csv",
    )


def _roc_curves_from_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _read_roc_rows(run)
    if not rows:
        return []
    summary = run["metrics"].get("summary", {})
    by_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault((row["family"], row["model"]), []).append(row)

    curves = []
    for (family, model), model_rows in by_model.items():
        curves.append(
            {
                "family": family,
                "model": model,
                "n_seeds": int(model_rows[0]["n_seeds"]),
                "n_test": summary.get(model, {}).get("n_test"),
                "roc_auc": float(model_rows[0]["roc_auc"]),
                "roc_auc_std": _float_or_default(model_rows[0].get("roc_auc_std")),
                "fpr": np.asarray([float(row["fpr"]) for row in model_rows]),
                "tpr": np.asarray([float(row["tpr"]) for row in model_rows]),
                "tpr_std": np.asarray(
                    [_float_or_default(row.get("tpr_std")) for row in model_rows]
                ),
            }
        )
    return curves


def _refresh_run_metric_plots(runs: dict[str, dict[str, Any]]) -> None:
    for artifact, run in runs.items():
        target = RESULTS / artifact
        if not target.is_dir():
            continue
        summary = run["metrics"].get("summary", {})
        obsolete_metric_plot = target / "model_metric_comparison.png"
        if obsolete_metric_plot.exists():
            obsolete_metric_plot.unlink()
        obsolete_combined_radar = target / "radar_chart.png"
        if obsolete_combined_radar.exists():
            obsolete_combined_radar.unlink()
        obsolete_combined_roc = target / "roc_curves.png"
        if obsolete_combined_roc.exists():
            obsolete_combined_roc.unlink()
        _save_family_plot(summary, target / "svm_knn_comparison.png")
        _save_radar_plot(
            summary,
            target / "svm_radar_chart.png",
            title="SVM metric radar chart",
            family="svm",
        )
        _save_radar_plot(
            summary,
            target / "knn_radar_chart.png",
            title="KNN metric radar chart",
            family="knn",
        )
        curves = _roc_curves_from_rows(run)
        if curves:
            _save_roc_plot(
                curves,
                target / "svm_roc_curves.png",
                title="SVM ROC curves",
                family="svm",
            )
            _save_roc_plot(
                curves,
                target / "knn_roc_curves.png",
                title="KNN ROC curves",
                family="knn",
            )


def main() -> None:
    """Aggregate managed run artifacts into ``papers/qSVM_qKNN/results``."""

    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = _load_curated_runs()
    runs.update(_load_runs())
    if not runs:
        raise RuntimeError(
            "No managed qSVM runs found under papers/qSVM_qKNN/outdir or results."
        )
    _clean_managed_results()
    _copy_runs(runs)
    all_rows: list[dict[str, Any]] = []
    for experiment in COMPARISON_EXPERIMENTS:
        rows = _metric_rows(runs, experiment)
        all_rows.extend(rows)
        _write_table(rows, RESULTS / f"{experiment}_results_table.csv")
        _save_radar_grid(runs, experiment, RESULTS / f"{experiment}_radar_chart.png")
        _save_roc_grid(runs, experiment, RESULTS / f"{experiment}_roc_curves.png")
        _save_runtime_grid(
            runs,
            experiment,
            RESULTS / f"{experiment}_runtime_profile.png",
        )
    for experiment in EXTRA_EXPERIMENTS:
        rows = _metric_rows(runs, experiment)
        all_rows.extend(rows)
        _write_table(rows, RESULTS / f"{experiment}_results_table.csv")
    for experiment in EXTRA_RADAR_EXPERIMENTS:
        _save_radar_grid(runs, experiment, RESULTS / f"{experiment}_radar_chart.png")
    for experiment in EXTRA_FAMILY_GRID_EXPERIMENTS:
        _save_family_grid(
            runs,
            experiment,
            RESULTS / f"{experiment}_svm_knn_comparison.png",
        )
    _write_kernel_validation_artifacts(runs)
    _refresh_run_metric_plots(runs)
    _write_table(all_rows, RESULTS / "results_table.csv")
    _write_latest_summary(runs)


if __name__ == "__main__":
    main()
