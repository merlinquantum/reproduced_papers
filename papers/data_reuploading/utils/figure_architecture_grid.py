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

from lib.architecture_grid_run import run_architecture_grid  # noqa: E402
from lib.paper_datasets import CirclesDataset, MoonsDataset  # noqa: E402

from runtime_lib.config import deep_update, load_config  # noqa: E402
from runtime_lib.dtypes import resolve_config_dtypes  # noqa: E402
from runtime_lib.seed import seed_everything  # noqa: E402

DEFAULTS_PATH = PROJECT_DIR / "configs" / "design_benchmark_circles.json"


def _load_cfg(config_path: Path | None) -> dict:
    cfg = load_config(DEFAULTS_PATH)
    if config_path is not None:
        cfg = deep_update(cfg, load_config(config_path))
    resolve_config_dtypes(cfg)
    return cfg


def _create_run_dir(cfg: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(cfg.get("outdir", "results"))
    run_dir = outdir / f"design_benchmark_{timestamp}"
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
        description="Generate architecture grid figures from benchmark results."
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
        help="Path to design_benchmark_results.json (or run directory) to plot without re-running.",
    )
    args = parser.parse_args()

    if args.config is not None and args.previous_run is not None:
        parser.error(
            "Use either --config or --previous_run, not both. "
            "Re-run or reuse a prior run so inputs stay consistent."
        )

    if args.previous_run is not None:
        results_path = _resolve_results_path(
            args.previous_run, "design_benchmark_results.json"
        )
        results = json.loads(results_path.read_text(encoding="utf-8"))
        run_dir = results_path.parent
        cfg = results["config"]
    else:
        cfg = _load_cfg(args.config)
        seed_everything(cfg.get("seed"))
        run_dir = _create_run_dir(cfg)
        run_architecture_grid(cfg, run_dir)
        results_path = run_dir / "design_benchmark_results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))

    seed_everything(cfg.get("seed"))
    dataset_name, dataset = _load_dataset(cfg)
    X_tr, y_tr = dataset.train
    X_te, y_te = dataset.test

    figure_path = Path(results["figure_data_path"])
    if not figure_path.is_absolute():
        figure_path = results_path.parent / figure_path
    grid = results["grid"]
    designs = results["designs"]
    depths = results["depths"]

    with np.load(figure_path) as figure_data:
        for depth in depths:
            fig, axes = plt.subplots(
                6,
                3,
                figsize=(12, 18),
                dpi=120,
                gridspec_kw={
                    "height_ratios": [4, 1] * 3,
                    "hspace": 0.25,
                    "wspace": 0.15,
                },
                layout="constrained",
            )

            for idx, design in enumerate(designs):
                row, col = divmod(idx, 3)
                key_suffix = f"L{depth}_{design}"
                diff = figure_data[f"diff_{key_suffix}"]
                feat_tr = figure_data[f"feat_tr_{key_suffix}"]
                feat_te_key = f"feat_te_{key_suffix}"
                feat_te = (
                    figure_data[feat_te_key] if feat_te_key in figure_data else None
                )

                ax_map = axes[2 * row, col]
                ax_prob = axes[2 * row + 1, col]
                _plot_classification_map(ax_map, diff, grid, X_tr, y_tr, X_te, y_te)
                _plot_probability_axis(ax_prob, feat_tr, feat_te, y_tr, y_te)

                if col == 0:
                    ax_map.set_ylabel(
                        f"Data {design[0]}",
                        rotation=0,
                        labelpad=40,
                        fontsize=12,
                        va="center",
                    )
            for col, label in enumerate("ABC"):
                axes[0, col].set_title(f"Train {label}", fontsize=13, pad=18)

            fig.suptitle(
                f"Re-uploading circuits – {depth} layer(s)", fontsize=16, y=0.92
            )
            plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
            out = run_dir / f"architecture_grid_{depth}_{dataset_name}.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)
            print(f"Saved architecture grid to: {out}")


if __name__ == "__main__":
    main()
