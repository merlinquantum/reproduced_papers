from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for path in (PROJECT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib.paper_datasets import CirclesDataset, MoonsDataset  # noqa: E402
from lib.tau_alpha_grid_run import run_tau_alpha_grid  # noqa: E402

from runtime_lib.config import deep_update, load_config  # noqa: E402
from runtime_lib.dtypes import resolve_config_dtypes  # noqa: E402
from runtime_lib.seed import seed_everything  # noqa: E402

DEFAULTS_PATH = PROJECT_DIR / "configs" / "tau_alpha_benchmark_circles.json"


def _load_cfg(config_path: Path | None) -> dict:
    cfg = load_config(DEFAULTS_PATH)
    if config_path is not None:
        cfg = deep_update(cfg, load_config(config_path))
    resolve_config_dtypes(cfg)
    return cfg


def _create_run_dir(cfg: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(cfg.get("outdir", "results"))
    run_dir = outdir / f"tau_alpha_benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_results_path(raw: Path, default_name: str) -> Path:
    path = raw
    if path.is_dir():
        path = path / default_name
    return path


def _load_dataset(cfg: dict):
    dataset_name = cfg["experiment"]["dataset"]
    train_size = cfg["dataset"]["train_size"]
    test_size = cfg["dataset"]["test_size"]
    if dataset_name == "circles":
        dataset = CirclesDataset(n_train=train_size, n_test=test_size)
    elif dataset_name == "moons":
        dataset = MoonsDataset(n_train=train_size, n_test=test_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset_name, dataset


def _key_suffix(depth: int, tau: float, alpha: float) -> str:
    return f"L{depth}_tau{tau:.6f}_alpha{alpha:.6f}"


def _plot_classification_map(ax, diff, grid, X_tr, y_tr, X_te, y_te):
    x = np.linspace(grid["xlim"][0], grid["xlim"][1], grid["resolution"])
    y = np.linspace(grid["ylim"][0], grid["ylim"][1], grid["resolution"])
    xx, yy = np.meshgrid(x, y)

    cmap = LinearSegmentedColormap.from_list("rwb", ["#d62728", "#ffffff", "#1f77b4"])
    ax.contourf(
        xx,
        yy,
        diff,
        levels=np.linspace(-1, 1, 21),
        cmap=cmap,
        alpha=0.75,
        extend="both",
    )
    ax.contour(xx, yy, diff, levels=[0], colors="k", linewidths=2)

    ax.scatter(*X_tr[y_tr == 0].T, c="darkred", s=30, lw=0.6, ec="w")
    ax.scatter(*X_tr[y_tr == 1].T, c="darkblue", s=30, lw=0.6, ec="w")
    if X_te is not None:
        ax.scatter(*X_te[y_te == 0].T, c="darkred", s=40, marker="s", lw=1, ec="w")
        ax.scatter(*X_te[y_te == 1].T, c="darkblue", s=40, marker="s", lw=1, ec="w")

    ax.set(
        xlim=grid["xlim"],
        ylim=grid["ylim"],
        xticks=[],
        yticks=[],
    )


def _plot_probability_axis(ax, feat_tr, feat_te, y_tr, y_te):
    off = 0.12
    ax.scatter(
        feat_tr[y_tr == 0],
        -off * np.ones_like(feat_tr[y_tr == 0]),
        c="red",
        s=22,
        alpha=0.85,
    )
    ax.scatter(
        feat_tr[y_tr == 1],
        off * np.ones_like(feat_tr[y_tr == 1]),
        c="blue",
        s=22,
        alpha=0.85,
    )
    if feat_te is not None:
        ax.scatter(
            feat_te[y_te == 0],
            -2 * off * np.ones_like(feat_te[y_te == 0]),
            c="red",
            s=28,
            marker="s",
            alpha=0.85,
        )
        ax.scatter(
            feat_te[y_te == 1],
            2 * off * np.ones_like(feat_te[y_te == 1]),
            c="blue",
            s=28,
            marker="s",
            alpha=0.85,
        )
    ax.set_xlim(0, 1)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tau/alpha grid figures from benchmark results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config file to control benchmark before plotting.",
    )
    parser.add_argument(
        "--previous_run",
        type=Path,
        default=None,
        help="Path to tau_alpha_benchmark_results.json (or run directory) to plot without re-running.",
    )
    args = parser.parse_args()

    if args.config is not None and args.previous_run is not None:
        parser.error(
            "Use either --config or --previous_run, not both. "
            "Re-run or reuse a prior run so inputs stay consistent."
        )

    if args.previous_run is not None:
        results_path = _resolve_results_path(
            args.previous_run, "tau_alpha_benchmark_results.json"
        )
        results = json.loads(results_path.read_text(encoding="utf-8"))
        run_dir = results_path.parent
        cfg = results["config"]
    else:
        cfg = _load_cfg(args.config)
        seed_everything(cfg.get("seed"))
        run_dir = _create_run_dir(cfg)
        run_tau_alpha_grid(cfg, run_dir)
        results_path = run_dir / "tau_alpha_benchmark_results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))

    seed_everything(cfg.get("seed"))
    dataset_name, dataset = _load_dataset(cfg)
    X_tr, y_tr = dataset.train
    X_te, y_te = dataset.test

    figure_path = Path(results["figure_data_path"])
    if not figure_path.is_absolute():
        figure_path = results_path.parent / figure_path
    grid = results["grid"]
    depths = results["depths"]
    tau_values = results["tau_values"]
    alpha_values = results["alpha_values"]

    with np.load(figure_path) as figure_data:
        for depth in depths:
            n_rows, n_cols = len(tau_values), len(alpha_values)
            fig_h = 4.5 * n_rows
            fig_w = 4.0 * n_cols
            fig, axes = plt.subplots(
                2 * n_rows,
                n_cols,
                figsize=(fig_w, fig_h),
                gridspec_kw={
                    "height_ratios": [4, 1] * n_rows,
                    "hspace": 0.25,
                    "wspace": 0.15,
                },
            )

            for r, tau in enumerate(tau_values):
                for c, alpha in enumerate(alpha_values):
                    key_suffix = _key_suffix(depth, tau, alpha)
                    diff = figure_data[f"diff_{key_suffix}"]
                    feat_tr = figure_data[f"feat_tr_{key_suffix}"]
                    feat_te_key = f"feat_te_{key_suffix}"
                    feat_te = (
                        figure_data[feat_te_key] if feat_te_key in figure_data else None
                    )

                    ax_map = axes[2 * r, c]
                    ax_prob = axes[2 * r + 1, c]
                    _plot_classification_map(ax_map, diff, grid, X_tr, y_tr, X_te, y_te)
                    _plot_probability_axis(ax_prob, feat_tr, feat_te, y_tr, y_te)

                    if c == 0:
                        ax_map.set_ylabel(
                            f"$\\tau = {tau:.2f}$",
                            rotation=0,
                            labelpad=40,
                            fontsize=12,
                            va="center",
                        )
                for c, alpha in enumerate(alpha_values):
                    if r == 0:
                        axes[0, c].set_title(
                            f"$\\alpha = {alpha / np.pi:.2f} \\pi$",
                            fontsize=13,
                            pad=16,
                        )

            fig.suptitle(f"AA design – {depth} layer(s)", fontsize=16, y=0.93)
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            out = run_dir / f"tau_alpha_grid_{depth}_{dataset_name}.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)
            print(f"Saved tau/alpha grid to: {out}")


if __name__ == "__main__":
    main()
