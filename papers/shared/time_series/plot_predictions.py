#!/usr/bin/env python3
"""Visualize reference vs predictions for one or more run directories.

Targets the QRNN-style `predictions.csv` (columns: split,prediction,target).

Examples:
    python -m papers.shared.time_series.plot_predictions /tmp/run_2026...
    python -m papers.shared.time_series.plot_predictions /tmp/run_a /tmp/run_b
    python -m papers.shared.time_series.plot_predictions \
        /tmp/run_a:RNN /tmp/run_b:Photonic

If you want the QRNN convenience wrapper, use:
  python papers/QRNN/utils/plot_predictions.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_repo_root_on_syspath() -> None:
    if __package__:
        return
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


try:
    from papers.shared.time_series.plotting import maybe_use_agg_backend, save_figure
except ModuleNotFoundError:
    _ensure_repo_root_on_syspath()
    from papers.shared.time_series.plotting import maybe_use_agg_backend, save_figure


def _load_config(run_dir: Path) -> dict:
    snapshot = run_dir / "config_snapshot.json"
    if snapshot.exists():
        return json.loads(snapshot.read_text())
    return {}


def _build_title(cfg: dict, run_dirs: list[Path]) -> str:
    dataset_name = cfg.get("dataset", {}).get("name", "dataset")
    model_name = cfg.get("model", {}).get("name", "model")

    return f"Predictions overlay — {dataset_name} ({model_name})"


def _parse_labels(labels_csv: str | None, *, expected: int) -> list[str] | None:
    if labels_csv is None:
        return None
    labels = [part.strip() for part in labels_csv.split(",")]
    labels = [lbl for lbl in labels if lbl]
    if len(labels) != expected:
        raise ValueError(
            f"--labels expects {expected} comma-separated values, got {len(labels)}"
        )
    return labels


def _maybe_parse_run_spec(spec: str) -> tuple[Path, str | None]:
    """Parse a positional run spec.

    Supported forms:
    - /path/to/run
    - /path/to/run:My label

    Parsing is done by splitting on the *last* ':' and only treating it as a
    label separator when the left side resolves to an existing path.
    This keeps Windows paths like 'C:\\path\\to\\run' working.
    """

    candidate = Path(spec).expanduser()
    if candidate.exists():
        return candidate, None

    if ":" not in spec:
        return candidate, None

    left, right = spec.rsplit(":", 1)
    left_path = Path(left).expanduser()
    if left_path.exists() and right.strip():
        return left_path, right.strip()
    return candidate, None


def _parse_run_specs(specs: list[str]) -> tuple[list[Path], list[str] | None]:
    run_dirs: list[Path] = []
    labels: list[str | None] = []
    for spec in specs:
        run_dir, label = _maybe_parse_run_spec(spec)
        run_dirs.append(run_dir)
        labels.append(label)

    if any(lbl is not None for lbl in labels):
        finalized = [
            lbl if lbl is not None else rd.name for rd, lbl in zip(run_dirs, labels)
        ]
        return run_dirs, finalized
    return run_dirs, None


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions.csv in {run_dir}")
    df = pd.read_csv(predictions_path)
    df["run"] = run_dir.name
    return df


def _order_with_steps(
    df: pd.DataFrame, splits: Iterable[str]
) -> tuple[pd.DataFrame, dict]:
    ordered = [df[df["split"] == split] for split in splits]
    df_ordered = pd.concat(ordered, ignore_index=True)
    df_ordered["step"] = range(len(df_ordered))

    boundaries = {}
    cursor = 0
    for split_df, split in zip(ordered, splits):
        cursor += len(split_df)
        boundaries[split] = cursor
    return df_ordered, boundaries


def _assert_compatibility(base: pd.DataFrame, other: pd.DataFrame) -> None:
    if len(base) != len(other):
        raise ValueError(
            "Runs are incompatible: different number of rows in predictions.csv"
        )
    if not np.allclose(base["target"].values, other["target"].values):
        raise ValueError("Runs are incompatible: targets differ between runs")


def plot_runs(
    run_dirs: list[Path],
    out_path: Path | None = None,
    *,
    labels: list[str] | None = None,
) -> Path:
    run_dirs = [rd.resolve() for rd in run_dirs]
    if labels is not None and len(labels) != len(run_dirs):
        raise ValueError("labels must match run_dirs length")
    cfg = _load_config(run_dirs[0])
    y_label = cfg.get("dataset", {}).get("target_column", "target")

    maybe_use_agg_backend(show=False)
    import matplotlib.pyplot as plt

    splits = ["train", "val", "test"]
    dfs = [_load_predictions(rd) for rd in run_dirs]

    base_ordered, boundaries = _order_with_steps(dfs[0], splits)
    ordered_dfs = [base_ordered]
    for df in dfs[1:]:
        ordered, _ = _order_with_steps(df, splits)
        _assert_compatibility(base_ordered, ordered)
        ordered_dfs.append(ordered)

    fig, ax = plt.subplots(figsize=(12, 5))

    def _timestamp_for(run_dir: Path) -> str | None:
        for cand in (run_dir, *list(run_dir.parents)[:4]):
            name = cand.name
            match = re.match(r"^run_(\d{8}-\d{6})$", name)
            if match:
                return match.group(1)
        return None

    def _legend_label(run_dir: Path, base_label: str) -> str:
        stamp = _timestamp_for(run_dir)
        if stamp is None:
            return base_label
        if stamp in base_label:
            return base_label
        return f"{base_label} ({stamp})"

    combined = pd.concat(ordered_dfs, ignore_index=True)
    ymin, ymax = (
        combined[["target", "prediction"]].min().min(),
        combined[["target", "prediction"]].max().max(),
    )
    ypad = 0.05 * (ymax - ymin if ymax != ymin else 1.0)
    ymin -= ypad
    ymax += ypad

    colors = {"train": "#ace5b1", "val": "#e0c69c", "test": "#82bae2"}
    start = 0
    for split in splits:
        end = boundaries.get(split, len(base_ordered))
        ax.axvspan(
            start, end, color=colors.get(split, "#f5f5f5"), alpha=0.4, label=None
        )
        ax.text(
            (start + end) / 2,
            ymax - ypad,
            split,
            ha="center",
            va="top",
            fontsize=10,
            alpha=0.8,
        )
        start = end

    ax.plot(
        base_ordered["step"],
        base_ordered["target"],
        label="reference",
        linewidth=2,
        color="black",
    )
    for idx, ordered in enumerate(ordered_dfs):
        if labels is not None:
            base_label = labels[idx]
        else:
            base_label = ordered["run"].iloc[0] if "run" in ordered else "prediction"
        label = _legend_label(run_dirs[idx], base_label)
        ax.plot(ordered["step"], ordered["prediction"], label=label, linewidth=1.5)

    ax.set_title(_build_title(cfg, run_dirs))
    ax.set_xlabel("sequence index (train → val → test)")
    ax.set_ylabel(y_label)
    ax.set_ylim([ymin, ymax])

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ncol = 1
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol,
            fontsize=9,
            frameon=False,
        )

    bottom = 0.12 + 0.02 * min(len(handles), 10)
    fig.tight_layout(rect=[0, bottom, 1, 1])

    if out_path is None:
        suffix = (
            "predictions_overlay.png" if len(run_dirs) > 1 else "predictions_plot.png"
        )
        out_path = run_dirs[0] / suffix

    out_path = save_figure(plt, out_path, dpi=150)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot predictions for one or more run directories."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=str,
        help=(
            "One or more run directories containing predictions.csv. "
            "You can also pass 'path:label' to customize legend labels."
        ),
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Explicit output path for the saved plot (default: saved in the first run directory)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help=(
            "Comma-separated labels (same count/order as run_dirs) to use in the legend. "
            'Example: --labels "RNN,Photonic"'
        ),
    )
    args = parser.parse_args(argv)

    run_dirs, spec_labels = _parse_run_specs(list(args.run_dirs))
    flag_labels = _parse_labels(args.labels, expected=len(run_dirs))
    if spec_labels is not None and flag_labels is not None:
        raise ValueError(
            "Provide labels either via 'path:label' or via --labels, not both"
        )
    labels = spec_labels if spec_labels is not None else flag_labels

    out_path = plot_runs(run_dirs, out_path=args.out_path, labels=labels)
    print(f"Saved plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
