#!/usr/bin/env python3
"""
Build objective summary tables for RetinaMNIST CPU results.

Usage:
    python scripts/analysis/analyze_retina_cpu_results.py outdir/

Writes CSV tables under reports/retina_cpu/ by default.
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from generate_figures import (
    MODEL_LABELS,
    base_model_key,
    collect_results,
    deduplicate_results,
    model_variant_key,
    pretty_model_label,
    result_family,
    result_profile,
)


def default_report_root(results_root: str) -> str:
    norm_root = os.path.normpath(results_root)
    if os.path.basename(norm_root) == "outdir":
        project_root = os.path.dirname(norm_root)
        return os.path.join(project_root, "reports", "retina_cpu")
    return os.path.join(norm_root, "reports", "retina_cpu")


def keep_retina_results(results):
    kept = []
    for r in results:
        dataset = r.get("dataset", r.get("config", {}).get("dataset", "?"))
        if dataset != "retinamnist":
            continue
        kept.append(r)
    return kept


def summarize_runs(results):
    grouped = defaultdict(list)
    for r in results:
        grouped[model_variant_key(r)].append(r)

    rows = []
    for variant, runs in sorted(grouped.items()):
        aucs = np.array([r.get("test_auc", np.nan) for r in runs], dtype=float)
        accs = np.array([r.get("test_acc", np.nan) for r in runs], dtype=float)
        vals = np.array([r.get("best_val_auc", np.nan) for r in runs], dtype=float)
        times = np.array([r.get("total_time_s", np.nan) for r in runs], dtype=float)
        params = [r.get("param_counts", {}).get("total", np.nan) for r in runs]
        attn_params = [r.get("param_counts", {}).get("attention", np.nan) for r in runs]
        family = result_family(runs[0])
        profile = result_profile(runs[0])
        base_model = base_model_key(runs[0])
        rows.append(
            {
                "variant": variant,
                "variant_label": pretty_model_label(variant),
                "base_model": base_model,
                "family": family,
                "profile": profile,
                "n_runs": len(runs),
                "mean_test_auc": float(np.nanmean(aucs)),
                "std_test_auc": float(np.nanstd(aucs)),
                "mean_test_acc": float(np.nanmean(accs)),
                "std_test_acc": float(np.nanstd(accs)),
                "mean_best_val_auc": float(np.nanmean(vals)),
                "mean_time_s": float(np.nanmean(times)),
                "total_params": int(np.nanmax(params)) if params else "",
                "attn_params": int(np.nanmax(attn_params)) if attn_params else "",
                "auc_per_1k_params": float(np.nanmean(aucs) / max((np.nanmax(params) / 1000.0), 1e-9)),
                "auc_per_hour": float(np.nanmean(aucs) / max((np.nanmean(times) / 3600.0), 1e-9)),
            }
        )
    return rows


def pareto_front(rows):
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            better_or_equal_auc = other["mean_test_auc"] >= row["mean_test_auc"]
            better_or_equal_time = other["mean_time_s"] <= row["mean_time_s"]
            strictly_better = (
                other["mean_test_auc"] > row["mean_test_auc"]
                or other["mean_time_s"] < row["mean_time_s"]
            )
            if better_or_equal_auc and better_or_equal_time and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows:
        print(
            f"{row['variant_label']:<44} "
            f"AUC {row['mean_test_auc']:.4f} +- {row['std_test_auc']:.4f}   "
            f"ACC {row['mean_test_acc']:.4f} +- {row['std_test_acc']:.4f}   "
            f"time {row['mean_time_s'] / 60:.1f} min   runs {row['n_runs']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize RetinaMNIST CPU results")
    parser.add_argument("root", help="Directory containing results")
    parser.add_argument("--out", default=None, help="Output directory for CSV tables")
    args = parser.parse_args()

    results = deduplicate_results(collect_results(args.root))
    results = keep_retina_results(results)
    rows = summarize_runs(results)
    if not rows:
        raise SystemExit("No RetinaMNIST CPU results found")

    out_dir = args.out or default_report_root(args.root)
    os.makedirs(out_dir, exist_ok=True)

    full_rows = [r for r in rows if r["profile"] == "full"]
    lite_rows = [r for r in rows if r["profile"] == "lite"]
    robust_rows = [r for r in full_rows if r["n_runs"] >= 2]
    exploratory_rows = [r for r in full_rows if r["n_runs"] == 1]
    paper_rows = [r for r in full_rows if r["base_model"] in {"VisionTransformer", "OrthoFNN", "A", "B", "D"}]
    butterfly_rows = [r for r in rows if r["family"] == "butterfly"]

    auc_rank = sorted(robust_rows, key=lambda r: (-r["mean_test_auc"], r["mean_time_s"]))
    acc_rank = sorted(robust_rows, key=lambda r: (-r["mean_test_acc"], r["mean_time_s"]))
    pareto = sorted(pareto_front(robust_rows), key=lambda r: (r["mean_time_s"], -r["mean_test_auc"]))
    exploratory_auc = sorted(exploratory_rows, key=lambda r: (-r["mean_test_auc"], r["mean_time_s"]))

    write_csv(os.path.join(out_dir, "all_retina_cpu_summary.csv"), rows)
    write_csv(os.path.join(out_dir, "full_rank_by_auc.csv"), auc_rank)
    write_csv(os.path.join(out_dir, "full_rank_by_acc.csv"), acc_rank)
    write_csv(os.path.join(out_dir, "exploratory_single_seed_full.csv"), exploratory_auc)
    write_csv(os.path.join(out_dir, "paper_family_summary.csv"), paper_rows)
    write_csv(os.path.join(out_dir, "butterfly_summary.csv"), butterfly_rows)
    write_csv(os.path.join(out_dir, "lite_summary.csv"), lite_rows)
    write_csv(os.path.join(out_dir, "pareto_auc_vs_time.csv"), pareto)

    print_table("Retina full models ranked by AUC (multi-seed only)", auc_rank)
    print_table("Retina full models ranked by ACC (multi-seed only)", acc_rank)
    print_table("Retina Pareto frontier (AUC vs time, multi-seed only)", pareto)
    print_table("Retina exploratory full models (single seed only)", exploratory_auc)
    print(f"\nWrote CSV tables to {out_dir}")


if __name__ == "__main__":
    main()
