#!/usr/bin/env python3
"""
Generate figures from QVT experiment results.

Usage:
    python scripts/analysis/generate_figures.py outdir/
    python scripts/analysis/generate_figures.py outdir/ --dataset retinamnist
    python scripts/analysis/generate_figures.py outdir/ --out results/figures/
    python scripts/analysis/generate_figures.py outdir/ --profile lite

Scans for results.json files, groups by model/dataset/profile/seed, and produces:
  1. training_curves_{dataset}.pdf    — loss & accuracy over epochs per model
  2. comparison_{dataset}.pdf         — bar chart: AUC & ACC per model (mean ± std)
  3. sector_mass_{dataset}.pdf        — sector mass evolution (multi-sector models)
  4. param_comparison.pdf             — trainable parameter counts
  5. summary.csv                      — all results in one table

Reference values from the paper (Table 3, RetinaMNIST) are overlaid when
available. Paper references are only overlaid for "full" profile runs.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

PAPER_RESULTS = {
    "retinamnist": {
        "OrthoFNN": {"auc": 0.731, "acc": 0.548},
        "OrthoPatchWise (A)": {"auc": 0.739, "acc": 0.560},
        "VisionTransformer": {"auc": 0.736, "acc": 0.548},
        "OrthoTransformer (B)": {"auc": 0.745, "acc": 0.542},
        "CompoundTransformer (D)": {"auc": 0.740, "acc": 0.565},
    }
}

MODEL_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "D_full": "#9467bd",
    "E": "#8c564b",
    "F": "#e377c2",
    "VisionTransformer": "#4c4c4c",
    "OrthoFNN": "#17becf",
}

MODEL_LABELS = {
    "A": "A: OrthoPatch",
    "B": "B: OrthoTransformer",
    "C": "C: DirectAttn",
    "D": "D: Compound",
    "D_full": "D: Compound (full sector)",
    "E": "E: Multi-sector",
    "F": "F: Hierarchical (3ph)",
    "VisionTransformer": "VisionTransformer",
    "OrthoFNN": "OrthoFNN",
}

VARIANT_SEP = "|"
CLASSICAL_BASELINES = {"VisionTransformer", "OrthoFNN"}


def default_figure_root(results_root: str) -> str:
    norm_root = os.path.normpath(results_root)
    if os.path.basename(norm_root) == "outdir":
        project_root = os.path.dirname(norm_root)
        return os.path.join(project_root, "results", "figures")
    return os.path.join(norm_root, "results", "figures")


def result_profile(r: dict) -> str:
    cfg = r.get("config", {})
    return r.get("profile") or cfg.get("profile") or "full"


def result_family(r: dict) -> str:
    cfg = r.get("config", {})
    return r.get("circuit_family") or cfg.get("circuit_family") or "generic"


def base_model_key(r: dict) -> str:
    cfg = r.get("config", {})
    model = r.get("model_type", cfg.get("model_type", "?"))
    if model == "D" and cfg.get("compound_readout") == "full_sector":
        return "D_full"
    return model


def result_data_regime(r: dict) -> str:
    cfg = r.get("config", {})
    return r.get("data_regime") or cfg.get("data_regime") or "standard"


def make_variant_key(
    model: str, family: str, profile: str, data_regime: str = "standard"
) -> str:
    return VARIANT_SEP.join((model, family, profile, data_regime))


def split_variant_key(variant: str) -> tuple[str, str, str, str]:
    parts = variant.split(VARIANT_SEP)
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2], "standard"
    return variant, "generic", "full", "standard"


def model_variant_key(r: dict) -> str:
    model = base_model_key(r)
    profile = result_profile(r)
    family = "baseline" if model in CLASSICAL_BASELINES else result_family(r)
    return make_variant_key(model, family, profile, result_data_regime(r))


def pretty_model_label(model: str) -> str:
    base, family, profile, data_regime = split_variant_key(model)
    label = MODEL_LABELS.get(base, base)
    regime_suffix = "" if data_regime == "standard" else f", {data_regime}"
    if base in CLASSICAL_BASELINES:
        return f"{label} [baseline, {profile}{regime_suffix}]"
    return f"{label} [{family}, {profile}{regime_suffix}]"


def model_color(model: str) -> str:
    base, family, profile, _ = split_variant_key(model)
    color = MODEL_COLORS.get(base, "gray")

    if family == "butterfly":
        color = {
            "A": "#2a6fbb",
            "B": "#f08c26",
            "C": "#33995a",
            "D": "#c73b3b",
            "D_full": "#8f5bc2",
            "E": "#9c6b55",
            "F": "#d86cb3",
            "OrthoFNN": "#2aa9ba",
        }.get(base, color)

    if profile == "lite":
        color = {
            "A": "#5fa2d9" if family == "generic" else "#84bce7",
            "B": "#ffb066" if family == "generic" else "#ffc185",
            "C": "#58b958" if family == "generic" else "#7acb7a",
            "D": "#e26a6a" if family == "generic" else "#ea8a8a",
            "D_full": "#b28ad8" if family == "generic" else "#c1a2e2",
            "E": "#b07b63" if family == "generic" else "#c29178",
            "F": "#f09cd7" if family == "generic" else "#f4b5e2",
            "VisionTransformer": "#8a8a8a",
            "OrthoFNN": "#6fd5df" if family == "generic" else "#87dfe7",
        }.get(base, color)

    return color


def profile_rank(profile: str) -> int:
    return {"full": 0, "lite": 1}.get(profile, 99)


def family_rank(family: str) -> int:
    return {"baseline": 0, "generic": 1, "butterfly": 2}.get(family, 99)


def variant_sort_key(variant: str):
    base, family, profile, data_regime = split_variant_key(variant)
    return (base, family_rank(family), profile_rank(profile), data_regime)


def collect_results(root: str):
    skip_dir_prefixes = ("epoch_benchmarks", "device_profile")
    skip_dir_names = {"figures", "__pycache__"}
    results = []
    for dirpath, _, files in os.walk(root):
        parts = set(os.path.normpath(dirpath).split(os.sep))
        if skip_dir_names & parts:
            continue
        leaf = os.path.basename(dirpath)
        if any(leaf.startswith(prefix) for prefix in skip_dir_prefixes):
            continue
        if "results.json" not in files:
            continue
        with open(os.path.join(dirpath, "results.json")) as f:
            r = json.load(f)
        r["_dir"] = dirpath
        results.append(r)
    return results


def result_dedup_key(r: dict):
    dataset = r.get("dataset", r.get("config", {}).get("dataset", "?"))
    seed = r.get("seed", r.get("config", {}).get("seed", "?"))
    return (model_variant_key(r), dataset, str(seed))


def result_priority(r: dict):
    path = os.path.normpath(r.get("_dir", ""))
    parts = set(path.split(os.sep))
    history_len = len(r.get("history", []))
    basename = os.path.basename(path)
    score = 0
    if "figures" not in parts:
        score += 1000
    if not basename.startswith("device_profile"):
        score += 100
    if not basename.startswith("epoch_benchmarks"):
        score += 100
    if "_paper_" not in basename:
        score += 10
    score += history_len
    return score


def deduplicate_results(results):
    chosen = {}
    for r in results:
        key = result_dedup_key(r)
        current = chosen.get(key)
        if current is None or result_priority(r) > result_priority(current):
            chosen[key] = r
    return list(chosen.values())


def include_for_bundle(
    r: dict, family: str | None = None, profile: str | None = None
) -> bool:
    base_model = base_model_key(r)
    result_prof = result_profile(r)
    result_fam = result_family(r)

    if profile is not None and result_prof != profile:
        return False

    if family is None:
        return True

    if base_model in CLASSICAL_BASELINES and result_prof == "full":
        return True

    return result_fam == family


def group_by(results):
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        model = model_variant_key(r)
        dataset = r.get("dataset", r.get("config", {}).get("dataset", "?"))
        groups[model][dataset].append(r)
    return groups


def plot_training_curves(groups, dataset, out_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for model in sorted(groups.keys(), key=variant_sort_key):
        runs = groups[model].get(dataset, [])
        if not runs:
            continue
        color = model_color(model)
        label = pretty_model_label(model)

        all_loss = [np.array([e["train_loss"] for e in r["history"]]) for r in runs]
        all_acc = [np.array([e["train_acc"] for e in r["history"]]) for r in runs]
        all_vauc = [np.array([e["val_auc"] for e in r["history"]]) for r in runs]

        min_len = min(len(a) for a in all_loss)
        epochs = np.arange(1, min_len + 1)

        for data, ax, ylabel in (
            (all_loss, axes[0], "Train Loss"),
            (all_acc, axes[1], "Train Acc"),
            (all_vauc, axes[2], "Val AUC"),
        ):
            arr = np.array([a[:min_len] for a in data])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(epochs, mean, color=color, label=label, linewidth=1.5)
            if len(runs) > 1:
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.suptitle(f"Training Curves — {dataset}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"training_curves_{dataset}.pdf")
    png_path = os.path.join(out_dir, f"training_curves_{dataset}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def plot_comparison(groups, dataset, out_dir):
    import matplotlib.pyplot as plt

    models = sorted((m for m in groups if groups[m].get(dataset)), key=variant_sort_key)
    if not models:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax_idx, (metric, title) in enumerate(
        (("test_auc", "Test AUC"), ("test_acc", "Test ACC"))
    ):
        ax = axes[ax_idx]
        x_pos = np.arange(len(models))
        means, stds = [], []
        for model in models:
            vals = [r[metric] for r in groups[model][dataset]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        colors = [model_color(model) for model in models]
        bars = ax.bar(x_pos, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [pretty_model_label(m) for m in models], fontsize=8, rotation=15
        )
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

        paper = PAPER_RESULTS.get(dataset, {})
        paper_key_map = {
            "A": "OrthoPatchWise (A)",
            "B": "OrthoTransformer (B)",
            "D": "CompoundTransformer (D)",
            "VisionTransformer": "VisionTransformer",
            "OrthoFNN": "OrthoFNN",
        }
        metric_key = "auc" if "auc" in metric else "acc"
        added_legend = False
        for i, model in enumerate(models):
            base, family, profile, _ = split_variant_key(model)
            if profile != "full":
                continue
            ref_key = paper_key_map.get(base)
            if ref_key and ref_key in paper:
                ax.hlines(
                    paper[ref_key][metric_key],
                    i - 0.35,
                    i + 0.35,
                    colors="black",
                    linestyles="--",
                    linewidths=1.2,
                )
                added_legend = True

        if paper and added_legend:
            ax.plot([], [], "k--", label="Paper reference")
            ax.legend(fontsize=8)

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle(f"Model Comparison — {dataset}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"comparison_{dataset}.pdf")
    png_path = os.path.join(out_dir, f"comparison_{dataset}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def plot_sector_mass(groups, dataset, out_dir):
    import matplotlib.pyplot as plt

    variants = {
        key: groups[key].get(dataset, []) for key in groups if groups[key].get(dataset)
    }
    sector_keys_any = (
        "sector_mass_cross",
        "sector_mass_pp",
        "sector_mass_ff",
        "sector_mass_triple_cross",
        "sector_mass_rpp",
    )
    variants = {
        key: runs
        for key, runs in variants.items()
        if any(
            any(sk in e for sk in sector_keys_any for e in r.get("history", []))
            for r in runs
        )
    }
    if not variants:
        return

    fig, axes = plt.subplots(
        1, len(variants), figsize=(7 * len(variants), 4), squeeze=False
    )

    for col, (variant, runs) in enumerate(
        sorted(variants.items(), key=lambda item: variant_sort_key(item[0]))
    ):
        ax = axes[0, col]
        label = pretty_model_label(variant)
        color = model_color(variant)

        sector_styles = {
            "sector_mass_cross": ("solid", color, "cross"),
            "sector_mass_pp": ("--", "steelblue", "patch-patch"),
            "sector_mass_ff": (":", "seagreen", "feat-feat"),
            "sector_mass_triple_cross": ("solid", color, "triple-cross"),
            "sector_mass_rpp": ("-.", "darkorange", "region-patch-patch"),
        }

        for i, run in enumerate(runs):
            hist = run.get("history", [])
            if not hist:
                continue
            epochs = [e["epoch"] for e in hist]

            for key, (linestyle, line_color, sector_label) in sector_styles.items():
                if any(key in e for e in hist):
                    vals = [e.get(key, np.nan) for e in hist]
                    ax.plot(
                        epochs,
                        vals,
                        linestyle=linestyle,
                        color=line_color,
                        alpha=0.6,
                        label=sector_label if i == 0 else None,
                    )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Sector Fraction")
        ax.set_title(f"{label} — {dataset}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    path = os.path.join(out_dir, f"sector_mass_{dataset}.pdf")
    png_path = os.path.join(out_dir, f"sector_mass_{dataset}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def plot_param_comparison(groups, out_dir):
    import matplotlib.pyplot as plt

    model_params = {}
    for model, ds_dict in groups.items():
        for _, runs in ds_dict.items():
            if runs and "param_counts" in runs[0] and model not in model_params:
                model_params[model] = runs[0]["param_counts"]

    if not model_params:
        return

    models = sorted(model_params.keys(), key=variant_sort_key)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    attn = [model_params[m].get("attention", 0) for m in models]
    total = [model_params[m].get("total", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35
    bars1 = ax.bar(
        x - width / 2,
        attn,
        width,
        label="Attention layers",
        color=[model_color(m) for m in models],
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        total,
        width,
        label="Total",
        color="lightgray",
        edgecolor="gray",
    )

    ax.axhline(
        2048,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Classical ViT attn (2d^2x4)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_model_label(m) for m in models], fontsize=9, rotation=15)
    ax.set_ylabel("Trainable Parameters")
    ax.set_title("Parameter Count Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars1, attn):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 20,
            str(val),
            ha="center",
            fontsize=7,
        )

    fig.tight_layout()
    path = os.path.join(out_dir, "param_comparison.pdf")
    png_path = os.path.join(out_dir, "param_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def write_summary(results, out_dir):
    rows = []
    for r in results:
        variant = model_variant_key(r)
        _, family, profile, data_regime = split_variant_key(variant)
        rows.append(
            {
                "model": variant,
                "variant_label": pretty_model_label(variant),
                "base_model": base_model_key(r),
                "family": family,
                "profile": profile,
                "data_regime": data_regime,
                "dataset": r.get("dataset", r.get("config", {}).get("dataset", "?")),
                "seed": r.get("seed", r.get("config", {}).get("seed", "?")),
                "test_acc": round(r.get("test_acc", 0), 4),
                "test_auc": round(r.get("test_auc", 0), 4),
                "best_val_auc": round(r.get("best_val_auc", 0), 4),
                "best_epoch": r.get("best_epoch", "?"),
                "attn_params": r.get("param_counts", {}).get("attention", "?"),
                "total_params": r.get("param_counts", {}).get("total", "?"),
                "time_s": r.get("total_time_s", "?"),
            }
        )

    rows.sort(
        key=lambda x: (
            x["dataset"],
            family_rank(x["family"]),
            profile_rank(x["profile"]),
            x["data_regime"],
            x["base_model"],
            str(x["seed"]),
        )
    )

    print("\n" + "=" * 150)
    print(
        f"{'Variant':<38} {'Fam':<9} {'Profile':<8} {'Regime':<18} {'Dataset':<15} {'Seed':<6} "
        f"{'ACC':>7} {'AUC':>7} {'ValAUC':>7} {'Ep':>4} {'AttnP':>7} {'Time':>7}"
    )
    print("-" * 150)
    for row in rows:
        print(
            f"{row['variant_label']:<38} {row['family']:<9} {row['profile']:<8} {row['data_regime']:<18} {row['dataset']:<15} {row['seed']:<6} "
            f"{row['test_acc']:>7.4f} {row['test_auc']:>7.4f} "
            f"{row['best_val_auc']:>7.4f} {row['best_epoch']:>4} "
            f"{str(row['attn_params']):>7} {str(row['time_s']):>7}"
        )
    print("=" * 150)

    path = os.path.join(out_dir, "summary.csv")
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> {path}")

    agg = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (
            row["model"],
            row["family"],
            row["profile"],
            row["data_regime"],
            row["dataset"],
        )
        agg[key]["acc"].append(row["test_acc"])
        agg[key]["auc"].append(row["test_auc"])

    print(
        f"\n{'Variant':<38} {'Fam':<9} {'Profile':<8} {'Regime':<18} {'Dataset':<15} {'ACC (mean+-std)':>18} {'AUC (mean+-std)':>18}"
    )
    print("-" * 123)
    for (model, family, profile, data_regime, dataset), metrics in sorted(agg.items()):
        acc_mean = np.mean(metrics["acc"])
        acc_std = np.std(metrics["acc"])
        auc_mean = np.mean(metrics["auc"])
        auc_std = np.std(metrics["auc"])
        variant_label = pretty_model_label(model)
        print(
            f"{variant_label:<38} {family:<9} {profile:<8} {data_regime:<18} {dataset:<15} "
            f"{acc_mean:>7.4f} +- {acc_std:<7.4f}  {auc_mean:>7.4f} +- {auc_std:<7.4f}"
        )
    print()


def render_bundle(results, out_dir, dataset_filter=None):
    groups = group_by(results)
    os.makedirs(out_dir, exist_ok=True)

    datasets = set()
    for model in groups:
        datasets.update(groups[model].keys())
    if dataset_filter:
        datasets = {dataset_filter}

    try:
        import matplotlib  # noqa: F401

        for dataset in sorted(datasets):
            print(f"\n-- {dataset} --")
            plot_training_curves(groups, dataset, out_dir)
            plot_comparison(groups, dataset, out_dir)
            plot_sector_mass(groups, dataset, out_dir)
        plot_param_comparison(groups, out_dir)
    except ImportError:
        pass

    summary_results = results
    if dataset_filter is not None:
        summary_results = [
            r
            for r in results
            if r.get("dataset", r.get("config", {}).get("dataset", "?"))
            == dataset_filter
        ]
    write_summary(summary_results, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate QVT figures from results")
    parser.add_argument("root", help="Directory containing outdir/**/results.json")
    parser.add_argument(
        "--dataset", default=None, help="Filter to one dataset (default: all found)"
    )
    parser.add_argument("--out", default=None, help="Output directory for figures")
    parser.add_argument(
        "--profile", default=None, help="Filter by profile, e.g. full or lite"
    )
    parser.add_argument(
        "--circuit-family",
        default=None,
        help="Filter by circuit family, e.g. generic or butterfly",
    )
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not installed - will print summary table only.")

    results = collect_results(args.root)
    if args.profile is not None or args.circuit_family is not None:
        results = [
            r
            for r in results
            if include_for_bundle(r, family=args.circuit_family, profile=args.profile)
        ]
    results = deduplicate_results(results)

    if not results:
        print(f"No results.json found under {args.root}")
        sys.exit(1)

    print(f"Found {len(results)} result files")

    if args.out:
        targets = [(args.out, results)]
    else:
        targets = []
        figure_root = default_figure_root(args.root)
        fam_dir = args.circuit_family if args.circuit_family else "all"
        prof_dir = args.profile if args.profile else "all"
        targets.append((os.path.join(figure_root, fam_dir, prof_dir), results))

        # When no explicit output directory is provided, also emit per-family/profile
        # bundles so comparisons are organized without extra CLI calls.
        if args.circuit_family is None and args.profile is None:
            combos = sorted({(result_family(r), result_profile(r)) for r in results})
            for family, profile in combos:
                subset = [
                    r
                    for r in results
                    if include_for_bundle(r, family=family, profile=profile)
                ]
                targets.append((os.path.join(figure_root, family, profile), subset))
        elif args.circuit_family is None and args.profile is not None:
            families = sorted({result_family(r) for r in results})
            for family in families:
                subset = [
                    r
                    for r in results
                    if include_for_bundle(r, family=family, profile=args.profile)
                ]
                targets.append(
                    (os.path.join(figure_root, family, args.profile), subset)
                )
        elif args.circuit_family is not None and args.profile is None:
            profiles = sorted({result_profile(r) for r in results})
            for profile in profiles:
                subset = [
                    r
                    for r in results
                    if include_for_bundle(
                        r, family=args.circuit_family, profile=profile
                    )
                ]
                targets.append(
                    (os.path.join(figure_root, args.circuit_family, profile), subset)
                )

    seen_out_dirs = set()
    for out_dir, bundle_results in targets:
        if out_dir in seen_out_dirs:
            continue
        seen_out_dirs.add(out_dir)
        print(f"\n== Writing figures to {out_dir} ==")
        render_bundle(bundle_results, out_dir, dataset_filter=args.dataset)


if __name__ == "__main__":
    main()
