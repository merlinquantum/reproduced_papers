from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


DEFAULT_RUN_DIR = Path("papers/photonic_QGAN/results/run_20260213-104332")
PARAM_PREFIX = "param_"


@dataclass
class CVRow:
    rank: int
    score: float
    std: float
    fit_time: float
    resource: int
    params: dict[str, Any]


def _to_num(text: str) -> Any:
    value = text.strip()
    if value == "":
        return value
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(ch in lowered for ch in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cv_rows(path: Path) -> list[CVRow]:
    rows: list[CVRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            params: dict[str, Any] = {}
            for key, value in row.items():
                if key.startswith(PARAM_PREFIX):
                    params[key[len(PARAM_PREFIX) :]] = _to_num(value)
            resource = int(params.get("opt_iter_num", 0))
            rows.append(
                CVRow(
                    rank=int(float(row.get("rank_test_score", "0"))),
                    score=float(row.get("mean_test_score", "nan")),
                    std=float(row.get("std_test_score", "nan")),
                    fit_time=float(row.get("mean_fit_time", "nan")),
                    resource=resource,
                    params=params,
                )
            )
    return rows


def _param_key(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((k, v) for k, v in params.items() if k != "opt_iter_num"))


def _group_unique(rows: list[CVRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[tuple[str, Any], ...], list[CVRow]] = defaultdict(list)
    for row in rows:
        grouped[_param_key(row.params)].append(row)

    out: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        best_any = max(group_rows, key=lambda r: r.score)
        out.append(
            {
                "params": dict(key),
                "best_score_any": best_any.score,
                "best_rank_any": min(item.rank for item in group_rows),
                "scores_by_resource": {
                    str(resource): max(
                        item.score for item in group_rows if item.resource == resource
                    )
                    for resource in sorted({item.resource for item in group_rows})
                },
                "resources_seen": sorted({item.resource for item in group_rows}),
            }
        )
    return out


def _rows_at_resource(rows: list[CVRow], resource: int) -> list[CVRow]:
    return [row for row in rows if row.resource == resource]


def _mean_by_param(rows: list[CVRow]) -> dict[str, list[dict[str, Any]]]:
    by_param: dict[str, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, value in row.params.items():
            if key == "opt_iter_num":
                continue
            by_param[key][value].append(row.score)

    out: dict[str, list[dict[str, Any]]] = {}
    for param, values in by_param.items():
        rows_for_param: list[dict[str, Any]] = []
        for value, scores in values.items():
            rows_for_param.append(
                {
                    "value": value,
                    "mean_score": mean(scores),
                    "count": len(scores),
                }
            )
        out[param] = sorted(
            rows_for_param, key=lambda item: item["mean_score"], reverse=True
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _compute_param_importance(rows: list[CVRow]) -> list[dict[str, Any]]:
    """Compute simple per-parameter importance using eta^2 (ANOVA-style effect size)."""
    if not rows:
        return []
    scores = [float(row.score) for row in rows]
    if not scores:
        return []
    global_mean = mean(scores)
    sst = sum((score - global_mean) ** 2 for score in scores)
    if sst <= 0:
        sst = 1e-12

    by_param: dict[str, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, value in row.params.items():
            if key == "opt_iter_num":
                continue
            by_param[key][value].append(float(row.score))

    importance_rows: list[dict[str, Any]] = []
    for param, value_scores in by_param.items():
        ss_between = 0.0
        level_means: list[tuple[Any, float, int]] = []
        for value, values in value_scores.items():
            m = mean(values)
            n = len(values)
            ss_between += n * (m - global_mean) ** 2
            level_means.append((value, m, n))
        eta2 = ss_between / sst
        level_means.sort(key=lambda item: item[1], reverse=True)
        best_level = level_means[0]
        worst_level = level_means[-1]
        importance_rows.append(
            {
                "param": param,
                "importance_eta2": float(eta2),
                "num_levels": len(level_means),
                "best_value": best_level[0],
                "best_mean_score": float(best_level[1]),
                "best_count": int(best_level[2]),
                "worst_value": worst_level[0],
                "worst_mean_score": float(worst_level[1]),
                "worst_count": int(worst_level[2]),
                "spread_best_minus_worst": float(best_level[1] - worst_level[1]),
            }
        )
    importance_rows.sort(key=lambda item: item["importance_eta2"], reverse=True)
    return importance_rows


def _plot_param_importance(
    importance_rows: list[dict[str, Any]], out_path: Path
) -> str | None:
    if plt is None or not importance_rows:
        return None
    labels = [row["param"] for row in importance_rows]
    values = [float(row["importance_eta2"]) for row in importance_rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(labels, values, color="#2a9d8f")
    ax.set_ylabel("Importance (eta^2)")
    ax.set_xlabel("Hyperparameter")
    ax.set_title("Hyperparameter Importance on SSIM")
    ax.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _plot_interaction_demo(
    rows: list[CVRow], param_x: str, param_y: str, out_path: Path
) -> str | None:
    if plt is None or not rows:
        return None
    grouped: dict[tuple[Any, Any], list[float]] = defaultdict(list)
    x_values = sorted(
        {row.params.get(param_x) for row in rows if param_x in row.params}
    )
    y_values = sorted(
        {row.params.get(param_y) for row in rows if param_y in row.params}
    )
    if not x_values or not y_values:
        return None
    for row in rows:
        if param_x not in row.params or param_y not in row.params:
            continue
        grouped[(row.params[param_x], row.params[param_y])].append(float(row.score))

    matrix: list[list[float]] = []
    for y in y_values:
        matrix_row: list[float] = []
        for x in x_values:
            values = grouped.get((x, y), [])
            matrix_row.append(mean(values) if values else float("nan"))
        matrix.append(matrix_row)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    image = ax.imshow(matrix, cmap="cividis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(v) for v in x_values])
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([str(v) for v in y_values])
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Interaction Demo: mean SSIM ({param_x} x {param_y})")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean SSIM score")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _plot_ssim_vs_param(
    rows: list[CVRow], param_name: str, out_path: Path
) -> str | None:
    if plt is None:
        return None
    if not rows:
        return None
    x_values: list[float] = []
    y_values: list[float] = []
    colors: list[float] = []
    for row in rows:
        value = row.params.get(param_name)
        if value is None:
            continue
        try:
            x_values.append(float(value))
            y_values.append(float(row.score))
            colors.append(float(row.resource))
        except (TypeError, ValueError):
            continue
    if not x_values:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5))
    scatter = ax.scatter(x_values, y_values, c=colors, cmap="viridis", s=36, alpha=0.85)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Mean SSIM score")
    ax.set_title(f"SSIM vs {param_name}")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("opt_iter_num")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _plot_adam_beta_heatmap(
    rows: list[CVRow], out_path: Path, max_resource: int
) -> str | None:
    if plt is None:
        return None
    if not rows:
        return None
    grouped: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in rows:
        beta1 = row.params.get("adam_beta1")
        beta2 = row.params.get("adam_beta2")
        if beta1 is None or beta2 is None:
            continue
        try:
            grouped[(float(beta1), float(beta2))].append(float(row.score))
        except (TypeError, ValueError):
            continue
    if not grouped:
        return None

    beta1_values = sorted({key[0] for key in grouped})
    beta2_values = sorted({key[1] for key in grouped})
    matrix: list[list[float]] = []
    for beta2 in beta2_values:
        row_vals: list[float] = []
        for beta1 in beta1_values:
            scores = grouped.get((beta1, beta2), [])
            row_vals.append(mean(scores) if scores else float("nan"))
        matrix.append(row_vals)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    image = ax.imshow(matrix, cmap="magma", aspect="auto", origin="lower")
    ax.set_xticks(range(len(beta1_values)))
    ax.set_xticklabels([str(v) for v in beta1_values])
    ax.set_yticks(range(len(beta2_values)))
    ax.set_yticklabels([str(v) for v in beta2_values])
    ax.set_xlabel("adam_beta1")
    ax.set_ylabel("adam_beta2")
    ax.set_title(f"Mean SSIM at opt_iter_num={max_resource}")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean SSIM score")
    for y_idx, beta2 in enumerate(beta2_values):
        for x_idx, beta1 in enumerate(beta1_values):
            scores = grouped.get((beta1, beta2), [])
            if not scores:
                continue
            ax.text(
                x_idx,
                y_idx,
                f"{mean(scores):.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def analyze_hp_study(run_dir: Path, top_k: int = 3) -> dict[str, Any]:
    study_dir = run_dir / "hp_study"
    cv_path = study_dir / "cv_results.csv"
    best_path = study_dir / "best_result.json"
    snapshot_path = run_dir / "config_snapshot.json"

    if not cv_path.exists():
        raise FileNotFoundError(f"Missing file: {cv_path}")
    if not best_path.exists():
        raise FileNotFoundError(f"Missing file: {best_path}")
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing file: {snapshot_path}")

    cv_rows = _load_cv_rows(cv_path)
    if not cv_rows:
        raise ValueError(f"No rows found in: {cv_path}")
    best_result = _load_json(best_path)
    snapshot = _load_json(snapshot_path)

    resource_levels = sorted({row.resource for row in cv_rows})
    max_resource = max(resource_levels)
    rows_max = _rows_at_resource(cv_rows, max_resource)

    unique_candidates = _group_unique(cv_rows)
    unique_any = sorted(
        unique_candidates, key=lambda item: item["best_score_any"], reverse=True
    )
    unique_max = sorted(
        unique_candidates,
        key=lambda item: item["scores_by_resource"].get(
            str(max_resource), float("-inf")
        ),
        reverse=True,
    )

    top_unique_any = unique_any[:top_k]
    top_unique_max = [
        item for item in unique_max if str(max_resource) in item["scores_by_resource"]
    ][:top_k]

    max_resource_rows_sorted = sorted(rows_max, key=lambda row: row.score, reverse=True)
    param_effect_max = _mean_by_param(rows_max) if rows_max else {}
    importance_max = _compute_param_importance(rows_max) if rows_max else []
    importance_all = _compute_param_importance(cv_rows)
    top2_for_demo = [row["param"] for row in importance_max[:2]]
    interaction_demo_plot = None
    if len(top2_for_demo) == 2:
        interaction_demo_plot = _plot_interaction_demo(
            rows_max,
            top2_for_demo[0],
            top2_for_demo[1],
            study_dir / "plot_importance_top2_interaction_demo.png",
        )
    generated_plots = {
        "ssim_vs_adam_beta1": _plot_ssim_vs_param(
            cv_rows,
            "adam_beta1",
            study_dir / "plot_ssim_vs_adam_beta1.png",
        ),
        "ssim_vs_adam_beta2": _plot_ssim_vs_param(
            cv_rows,
            "adam_beta2",
            study_dir / "plot_ssim_vs_adam_beta2.png",
        ),
        "ssim_vs_lrD": _plot_ssim_vs_param(
            cv_rows,
            "lrD",
            study_dir / "plot_ssim_vs_lrD.png",
        ),
        "ssim_vs_lrG": _plot_ssim_vs_param(
            cv_rows,
            "lrG",
            study_dir / "plot_ssim_vs_lrG.png",
        ),
        "ssim_vs_d_steps": _plot_ssim_vs_param(
            cv_rows,
            "d_steps",
            study_dir / "plot_ssim_vs_d_steps.png",
        ),
        "ssim_vs_g_steps": _plot_ssim_vs_param(
            cv_rows,
            "g_steps",
            study_dir / "plot_ssim_vs_g_steps.png",
        ),
        "ssim_heatmap_adam_betas_max_resource": _plot_adam_beta_heatmap(
            rows_max,
            study_dir / "plot_ssim_heatmap_adam_betas_max_resource.png",
            max_resource=max_resource,
        ),
        "importance_bar_max_resource": _plot_param_importance(
            importance_max,
            study_dir / "plot_param_importance_max_resource.png",
        ),
        "importance_bar_all_resources": _plot_param_importance(
            importance_all,
            study_dir / "plot_param_importance_all_resources.png",
        ),
        "importance_top2_interaction_demo": interaction_demo_plot,
    }

    analysis = {
        "run_dir": str(run_dir),
        "study_dir": str(study_dir),
        "cases": best_result.get("cases", []),
        "resource_levels": resource_levels,
        "max_resource": max_resource,
        "n_rows": len(cv_rows),
        "n_unique_candidates": len(unique_candidates),
        "best_result_json": {
            "best_score": best_result.get("best_score"),
            "best_params": best_result.get("best_params"),
            "best_index": best_result.get("best_index"),
        },
        "top_unique_any_resource": top_unique_any,
        "top_unique_max_resource": top_unique_max,
        "param_effect_at_max_resource": param_effect_max,
        "param_importance_max_resource": importance_max,
        "param_importance_all_resources": importance_all,
        "plots": generated_plots,
        "defaults_training_ideal": snapshot.get("training", {}).get("ideal", {}),
    }

    csv_rows_any = [
        {
            "rank": idx + 1,
            "best_score_any": item["best_score_any"],
            "best_rank_any": item["best_rank_any"],
            "resources_seen": "|".join(str(v) for v in item["resources_seen"]),
            "params": _format_params(item["params"]),
        }
        for idx, item in enumerate(top_unique_any)
    ]
    _write_csv(
        study_dir / "analysis_top_unique_any_resource.csv",
        csv_rows_any,
        ["rank", "best_score_any", "best_rank_any", "resources_seen", "params"],
    )

    csv_rows_max = [
        {
            "rank": idx + 1,
            "score_at_max_resource": item["scores_by_resource"][str(max_resource)],
            "params": _format_params(item["params"]),
        }
        for idx, item in enumerate(top_unique_max)
    ]
    _write_csv(
        study_dir / "analysis_top_unique_max_resource.csv",
        csv_rows_max,
        ["rank", "score_at_max_resource", "params"],
    )
    _write_csv(
        study_dir / "analysis_param_importance_max_resource.csv",
        importance_max,
        [
            "param",
            "importance_eta2",
            "num_levels",
            "best_value",
            "best_mean_score",
            "best_count",
            "worst_value",
            "worst_mean_score",
            "worst_count",
            "spread_best_minus_worst",
        ],
    )
    _write_csv(
        study_dir / "analysis_param_importance_all_resources.csv",
        importance_all,
        [
            "param",
            "importance_eta2",
            "num_levels",
            "best_value",
            "best_mean_score",
            "best_count",
            "worst_value",
            "worst_mean_score",
            "worst_count",
            "spread_best_minus_worst",
        ],
    )

    analysis_path = study_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# HP Study Analysis: {run_dir.name}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Rows in `cv_results.csv`: {len(cv_rows)}")
    lines.append(f"- Unique hyperparameter sets: {len(unique_candidates)}")
    lines.append(f"- Resource levels (`opt_iter_num`): {resource_levels}")
    lines.append(f"- Final stage resource: {max_resource} ({len(rows_max)} candidates)")
    lines.append(
        "- Reported best in `best_result.json`: "
        f"score={best_result.get('best_score'):.12f}, "
        f"params={_format_params(best_result.get('best_params', {}))}"
    )
    lines.append("")
    lines.append("## Top Unique Candidates (Any Resource)")
    for idx, item in enumerate(top_unique_any, start=1):
        lines.append(
            f"{idx}. score={item['best_score_any']:.12f}, "
            f"resources={item['resources_seen']}, "
            f"params={_format_params(item['params'])}"
        )
    lines.append("")
    lines.append(f"## Final Stage Candidates (opt_iter_num={max_resource})")
    for idx, row in enumerate(max_resource_rows_sorted, start=1):
        if idx > top_k:
            break
        lines.append(
            f"{idx}. score={row.score:.12f}, std={row.std:.12f}, "
            f"params={_format_params({k: v for k, v in row.params.items() if k != 'opt_iter_num'})}"
        )
    lines.append("")
    lines.append("## Parameter Effect at Final Stage")
    for param, values in sorted(param_effect_max.items()):
        if not values:
            continue
        best_value = values[0]
        lines.append(
            f"- {param}: best_mean={best_value['mean_score']:.12f} "
            f"at value={best_value['value']} (n={best_value['count']})"
        )
    lines.append("")
    lines.append("## Parameter Importance")
    if importance_max:
        lines.append("- At final resource (`opt_iter_num=600`):")
        for idx, item in enumerate(importance_max, start=1):
            lines.append(
                f"  {idx}. {item['param']}: eta2={item['importance_eta2']:.6f}, "
                f"best={item['best_value']} ({item['best_mean_score']:.6f}), "
                f"worst={item['worst_value']} ({item['worst_mean_score']:.6f}), "
                f"spread={item['spread_best_minus_worst']:.6f}"
            )
    if importance_all:
        lines.append("- Across all halving resources:")
        for idx, item in enumerate(importance_all, start=1):
            lines.append(
                f"  {idx}. {item['param']}: eta2={item['importance_eta2']:.6f}, "
                f"best={item['best_value']} ({item['best_mean_score']:.6f}), "
                f"worst={item['worst_value']} ({item['worst_mean_score']:.6f}), "
                f"spread={item['spread_best_minus_worst']:.6f}"
            )
    lines.append("")
    lines.append("## Plots")
    if plt is None:
        lines.append(
            "- Plot generation skipped: `matplotlib` is not installed in this Python environment."
        )
    for plot_name, plot_path in generated_plots.items():
        if plot_path:
            rel = Path(plot_path).as_posix()
            lines.append(f"- {plot_name}: `{rel}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Successive-halving ranks include many 200-iteration evaluations.")
    lines.append(
        "- For deployment-sized training, prioritize the final-stage (`opt_iter_num=600`) view."
    )

    analysis_md = study_dir / "analysis.md"
    analysis_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze photonic_QGAN hp_study outputs and write structured summaries "
            "(JSON, Markdown, and top-candidate CSV files)."
        )
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=(
            "Run directory containing config_snapshot.json and hp_study/*. "
            "Default: papers/photonic_QGAN/results/run_20260213-104332"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top candidates to include in summary tables (default: 3).",
    )
    args = parser.parse_args()

    analysis = analyze_hp_study(args.run_dir, top_k=args.top_k)
    print(f"Run: {analysis['run_dir']}")
    print(f"Rows: {analysis['n_rows']}")
    print(f"Unique candidates: {analysis['n_unique_candidates']}")
    print(f"Resource levels: {analysis['resource_levels']}")
    print(f"Max resource: {analysis['max_resource']}")
    print(f"Analysis JSON: {Path(analysis['study_dir']) / 'analysis.json'}")
    print(f"Analysis Markdown: {Path(analysis['study_dir']) / 'analysis.md'}")
    print(
        "Top-any-resource CSV: "
        f"{Path(analysis['study_dir']) / 'analysis_top_unique_any_resource.csv'}"
    )
    print(
        "Top-max-resource CSV: "
        f"{Path(analysis['study_dir']) / 'analysis_top_unique_max_resource.csv'}"
    )


if __name__ == "__main__":
    main()
