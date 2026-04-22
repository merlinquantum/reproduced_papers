#!/usr/bin/env python3
"""
Generate figures from QVT experiment results.

Usage:
    python scripts/analysis/generate_figures.py outdir/
    python scripts/analysis/generate_figures.py outdir/ --dataset retinamnist
    python scripts/analysis/generate_figures.py outdir/ --profile lite
    python scripts/analysis/generate_figures.py outdir/ --out figures/

Scans for results.json files, groups by model/dataset/profile/seed, and produces:
  1. training_curves_{dataset}.pdf    — loss & accuracy over epochs per model
  2. comparison_{dataset}.pdf         — bar chart: AUC & ACC per model (mean ± std)
  3. sector_mass_{dataset}.pdf        — sector mass evolution for models with sector logs
  4. param_comparison.pdf             — trainable parameter counts
  5. summary.csv                      — all results in one table

Reference values from the paper (Table 3, RetinaMNIST) are overlaid when
available for full-profile A/B/D runs.
"""

import argparse, csv, json, os, sys
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
}

MODEL_LABELS = {
    "A": "A: OrthoPatch",
    "B": "B: OrthoTransformer",
    "C": "C: DirectAttn",
    "D": "D: Compound",
    "D_full": "D: Compound (full sector)",
    "E": "E: Multi-sector",
    "F": "F: Hierarchical (3ph)",
}


def collect_results(root: str):
    results = []
    for dirpath, _, files in os.walk(root):
        if "results.json" in files:
            with open(os.path.join(dirpath, "results.json")) as f:
                r = json.load(f)
            r["_dir"] = dirpath
            results.append(r)
    return results


def get_profile(r):
    return r.get("profile", r.get("config", {}).get("profile", "full"))


def base_model_key(r):
    cfg = r.get("config", {})
    m = r.get("model_type", cfg.get("model_type", "?"))
    if m == "D" and cfg.get("compound_readout") == "full_sector":
        m = "D_full"
    return m


def variant_model_key(r):
    base = base_model_key(r)
    profile = get_profile(r)
    return f"{base}_{profile}" if profile != "full" else base


def pretty_model_label(model_key):
    if model_key.endswith("_lite"):
        base = model_key[:-5]
        return f"{MODEL_LABELS.get(base, base)} [lite]"
    return MODEL_LABELS.get(model_key, model_key)


def model_color(model_key):
    base = model_key[:-5] if model_key.endswith("_lite") else model_key
    return MODEL_COLORS.get(base, "gray")


def group_by(results, key2="dataset"):
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        m = variant_model_key(r)
        d = r.get(key2, r.get("config", {}).get(key2, "?"))
        groups[m][d].append(r)
    return groups


def plot_training_curves(groups, dataset, out_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for model in sorted(groups.keys()):
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

        for data, ax, ylabel in [
            (all_loss, axes[0], "Train Loss"),
            (all_acc, axes[1], "Train Acc"),
            (all_vauc, axes[2], "Val AUC"),
        ]:
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
    fig.suptitle(f"Training Curves - {dataset}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"training_curves_{dataset}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def plot_comparison(groups, dataset, out_dir):
    import matplotlib.pyplot as plt

    models = sorted(m for m in groups if groups[m].get(dataset))
    if not models:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax_idx, (metric, title) in enumerate([("test_auc", "Test AUC"), ("test_acc", "Test ACC")]):
        ax = axes[ax_idx]
        x_pos = np.arange(len(models))
        means, stds = [], []
        for m in models:
            vals = [r[metric] for r in groups[m][dataset]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        colors = [model_color(m) for m in models]
        bars = ax.bar(x_pos, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([pretty_model_label(m) for m in models], fontsize=8, rotation=15)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

        paper = PAPER_RESULTS.get(dataset, {})
        paper_key_map = {"A": "OrthoPatchWise (A)", "B": "OrthoTransformer (B)", "D": "CompoundTransformer (D)"}
        mk = "auc" if "auc" in metric else "acc"
        for i, m in enumerate(models):
            if m.endswith("_lite"):
                continue
            pk = paper_key_map.get(m)
            if pk and pk in paper:
                ax.hlines(paper[pk][mk], i - 0.35, i + 0.35, colors="black", linestyles="--", linewidths=1.2)

        if paper and any(not m.endswith("_lite") for m in models):
            ax.plot([], [], "k--", label="Paper reference")
            ax.legend(fontsize=8)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{mean:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"Model Comparison - {dataset}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"comparison_{dataset}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def plot_sector_mass(groups, dataset, out_dir):
    import matplotlib.pyplot as plt

    d_variants = {k: groups[k].get(dataset, []) for k in groups.keys() if groups[k].get(dataset)}
    sector_keys_any = ("sector_mass_cross", "sector_mass_pp", "sector_mass_triple_cross", "sector_mass_rpp", "sector_mass_ff")
    has_any = any(
        any(sk in e for sk in sector_keys_any for e in r.get("history", []))
        for runs in d_variants.values() for r in runs
    )
    if not has_any:
        return

    sector_models = {k: v for k, v in d_variants.items() if any(any(sk in e for sk in sector_keys_any for e in r.get("history", [])) for r in v)}
    if not sector_models:
        return

    fig, axes = plt.subplots(1, len(sector_models), figsize=(7 * len(sector_models), 4), squeeze=False)

    for col, (variant, runs) in enumerate(sorted(sector_models.items())):
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

        for i, r in enumerate(runs):
            hist = r.get("history", [])
            if not hist:
                continue
            epochs = [e["epoch"] for e in hist]

            for key, (ls, c, lbl) in sector_styles.items():
                if any(key in e for e in hist):
                    vals = [e.get(key, np.nan) for e in hist]
                    ax.plot(epochs, vals, linestyle=ls, color=c, alpha=0.6, label=lbl if i == 0 else None)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Sector Fraction")
        ax.set_title(f"{label} - {dataset}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    path = os.path.join(out_dir, f"sector_mass_{dataset}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def plot_param_comparison(groups, out_dir):
    import matplotlib.pyplot as plt

    model_params = {}
    for m, ds_dict in groups.items():
        for runs in ds_dict.values():
            if runs and "param_counts" in runs[0] and m not in model_params:
                model_params[m] = runs[0]["param_counts"]

    if not model_params:
        return

    models = sorted(model_params.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))

    attn = [model_params[m].get("attention", 0) for m in models]
    total = [model_params[m].get("total", 0) for m in models]

    x = np.arange(len(models))
    w = 0.35
    bars1 = ax.bar(x - w / 2, attn, w, label="Attention layers", color=[model_color(m) for m in models], alpha=0.85)
    bars2 = ax.bar(x + w / 2, total, w, label="Total", color="lightgray", edgecolor="gray")

    ax.axhline(2048, color="black", linestyle="--", linewidth=1, label="Classical ViT attn (2d^2x4)")

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_model_label(m) for m in models], fontsize=9)
    ax.set_ylabel("Trainable Parameters")
    ax.set_title("Parameter Count Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars1, attn):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 20, str(val), ha="center", fontsize=7)

    fig.tight_layout()
    path = os.path.join(out_dir, "param_comparison.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def write_summary(results, out_dir):
    rows = []
    for r in results:
        base_model = base_model_key(r)
        rows.append({
            "model": base_model,
            "model_variant": variant_model_key(r),
            "profile": get_profile(r),
            "dataset": r.get("dataset", r.get("config", {}).get("dataset", "?")),
            "seed": r.get("seed", r.get("config", {}).get("seed", "?")),
            "test_acc": round(r.get("test_acc", 0), 4),
            "test_auc": round(r.get("test_auc", 0), 4),
            "best_val_auc": round(r.get("best_val_auc", 0), 4),
            "best_epoch": r.get("best_epoch", "?"),
            "attn_params": r.get("param_counts", {}).get("attention", "?"),
            "total_params": r.get("param_counts", {}).get("total", "?"),
            "time_s": r.get("total_time_s", "?"),
        })

    rows.sort(key=lambda x: (x["dataset"], x["profile"], x["model_variant"], str(x["seed"])))

    print("\n" + "=" * 115)
    print(f"{'Model':<8} {'Variant':<18} {'Profile':<8} {'Dataset':<15} {'Seed':<6} {'ACC':>7} {'AUC':>7} {'ValAUC':>7} {'Ep':>4} {'AttnP':>7} {'Time':>7}")
    print("-" * 115)
    for r in rows:
        print(f"{r['model']:<8} {r['model_variant']:<18} {r['profile']:<8} {r['dataset']:<15} {r['seed']:<6} {r['test_acc']:>7.4f} {r['test_auc']:>7.4f} {r['best_val_auc']:>7.4f} {r['best_epoch']:>4} {str(r['attn_params']):>7} {str(r['time_s']):>7}")
    print("=" * 115)

    path = os.path.join(out_dir, "summary.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  -> {path}")

    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["model_variant"], r["dataset"])
        agg[key]["acc"].append(r["test_acc"])
        agg[key]["auc"].append(r["test_auc"])

    print(f"\n{'Variant':<18} {'Dataset':<15} {'ACC (mean+-std)':>18} {'AUC (mean+-std)':>18}")
    print("-" * 72)
    for (m, d), metrics in sorted(agg.items()):
        acc_m, acc_s = np.mean(metrics["acc"]), np.std(metrics["acc"])
        auc_m, auc_s = np.mean(metrics["auc"]), np.std(metrics["auc"])
        print(f"{m:<18} {d:<15} {acc_m:>7.4f} +- {acc_s:<7.4f}  {auc_m:>7.4f} +- {auc_s:<7.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate QVT figures from results")
    parser.add_argument("root", help="Directory containing outdir/**/results.json")
    parser.add_argument("--dataset", default=None, help="Filter to one dataset (default: all found)")
    parser.add_argument("--profile", default=None, help="Filter by profile, e.g. full or lite")
    parser.add_argument("--out", default=None, help="Output directory for figures (default: <root>/figures)")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not installed - will print summary table only.")

    results = collect_results(args.root)
    if args.profile is not None:
        results = [r for r in results if get_profile(r) == args.profile]
    if not results:
        print(f"No results.json found under {args.root}")
        sys.exit(1)

    print(f"Found {len(results)} result files")
    groups = group_by(results)
    out_dir = args.out or os.path.join(args.root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    datasets = set()
    for m in groups:
        datasets.update(groups[m].keys())
    if args.dataset:
        datasets = {args.dataset}

    try:
        import matplotlib  # noqa: F401
        for ds in sorted(datasets):
            print(f"\n-- {ds} --")
            plot_training_curves(groups, ds, out_dir)
            plot_comparison(groups, ds, out_dir)
            plot_sector_mass(groups, ds, out_dir)
        plot_param_comparison(groups, out_dir)
    except ImportError:
        pass

    write_summary(results, out_dir)


if __name__ == "__main__":
    main()
