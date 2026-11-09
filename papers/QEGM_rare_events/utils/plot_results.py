"""Plot QEGM reproduction figures from a finished run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_LABELS = {
    "vae": "VAE baseline",
    "qegm": "QEGM (gate VQC)",
    "qegm_merlin": "QEGM (MerLin photonic)",
    "qegm_const": "QEGM (const r=0.5)",
}
_COLORS = {
    "vae": "#1f77b4",
    "qegm": "#d62728",
    "qegm_merlin": "#2ca02c",
    "qegm_const": "#9467bd",
}


def _collect_samples(run_dir: Path):
    samples: dict[str, list[np.ndarray]] = {}
    for path in sorted(run_dir.glob("samples_*_seed*.npy")):
        rest = path.stem.replace("samples_", "")
        variant, _ = rest.rsplit("_seed", 1)
        samples.setdefault(variant, []).append(np.load(path))
    return samples


def plot_densities(run_dir: Path) -> Path:
    real = np.load(run_dir / "real_samples_test.npy").flatten()
    samples = _collect_samples(run_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(-7, real.min() - 1), max(7, real.max() + 1), 60)
    ax.hist(real, bins=bins, density=True, alpha=0.35, color="black", label="Real GMM")
    for variant, arrays in samples.items():
        merged = np.concatenate(arrays)
        ax.hist(
            merged,
            bins=bins,
            density=True,
            histtype="step",
            color=_COLORS.get(variant, "gray"),
            linewidth=1.5,
            label=_LABELS.get(variant, variant),
        )
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("Synthetic GMM densities — real vs generated")
    ax.legend()
    fig.tight_layout()
    out = run_dir / "fig_densities.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_tail_kl(run_dir: Path) -> Path:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    summary = metrics["summary"]
    variants = list(summary.keys())
    means = [summary[v]["tail_kl"]["mean"] for v in variants]
    stds = [summary[v]["tail_kl"]["std"] for v in variants]
    fig, ax = plt.subplots(figsize=(5, 4))
    pos = np.arange(len(variants))
    ax.bar(
        pos,
        means,
        yerr=stds,
        capsize=4,
        color=[_COLORS.get(v, "gray") for v in variants],
    )
    ax.set_xticks(pos)
    ax.set_xticklabels([_LABELS.get(v, v) for v in variants], rotation=15)
    ax.set_ylabel("Tail KL divergence")
    ax.set_title("Tail KL divergence (mean ± std)")
    fig.tight_layout()
    out = run_dir / "fig_tail_kl.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_recall(run_dir: Path) -> Path:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    summary = metrics["summary"]
    variants = list(summary.keys())
    means = [summary[v]["rare_recall"]["mean"] for v in variants]
    stds = [summary[v]["rare_recall"]["std"] for v in variants]
    fig, ax = plt.subplots(figsize=(5, 4))
    pos = np.arange(len(variants))
    ax.bar(
        pos,
        means,
        yerr=stds,
        capsize=4,
        color=[_COLORS.get(v, "gray") for v in variants],
    )
    ax.set_xticks(pos)
    ax.set_xticklabels([_LABELS.get(v, v) for v in variants], rotation=15)
    ax.set_ylabel("Rare-event recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Rare-event recall (mean ± std)")
    fig.tight_layout()
    out = run_dir / "fig_rare_recall.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_coverage(run_dir: Path) -> Path:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    summary = metrics["summary"]
    fig, ax = plt.subplots(figsize=(5, 4))
    for variant, sub in summary.items():
        cov = sub.get("coverage", {})
        if not cov:
            continue
        alphas = sorted(float(k) for k in cov.keys())
        empirical = [cov[str(a)]["mean"] for a in alphas]
        ax.plot(
            alphas,
            empirical,
            marker="o",
            label=_LABELS.get(variant, variant),
            color=_COLORS.get(variant, "gray"),
        )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="ideal")
    ax.set_xlabel("Nominal coverage α")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Predictive-interval calibration")
    ax.legend()
    fig.tight_layout()
    out = run_dir / "fig_coverage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_threshold_sweep(run_dir: Path) -> Path:
    """Plot tail KL vs threshold for each variant from metrics_reeval.json."""

    path = run_dir / "metrics_reeval.json"
    if not path.exists():
        raise FileNotFoundError(f"No threshold sweep at {path}")
    data = json.loads(path.read_text())
    thresholds = [float(t) for t in data["thresholds"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    for variant in [k for k in data.keys() if k not in ("thresholds", "rarity_score")]:
        means = [
            data[variant]["summary"][str(t)]["tail_kl"]["mean"] for t in thresholds
        ]
        stds = [data[variant]["summary"][str(t)]["tail_kl"]["std"] for t in thresholds]
        ax.errorbar(
            thresholds,
            means,
            yerr=stds,
            marker="o",
            label=_LABELS.get(variant, variant),
            color=_COLORS.get(variant, "gray"),
        )
    ax.set_xlabel("Tail threshold |x| >")
    ax.set_ylabel("Tail KL divergence")
    ax.set_title("Tail KL across thresholds (mean ± std, 3 seeds)")
    ax.legend()
    fig.tight_layout()
    out = run_dir / "fig_threshold_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_figures(run_dir: Path) -> list[Path]:
    out_paths = []
    for fn in (plot_densities, plot_tail_kl, plot_recall, plot_coverage):
        try:
            out_paths.append(fn(run_dir))
        except Exception as exc:
            print(f"[plot] {fn.__name__} failed: {exc}")
    return out_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate QEGM reproduction figures")
    parser.add_argument("run_dir", help="Path to a finished run directory")
    args = parser.parse_args(argv)
    out = generate_figures(Path(args.run_dir))
    for path in out:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
