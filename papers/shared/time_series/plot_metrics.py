#!/usr/bin/env python3
"""Plot training curves (test_acc and loss) for time-series runs.

Currently targets the QRNN-style `metrics.json` format:
- top-level `history`: list of per-epoch dicts with keys:
  `epoch`, `train_loss`, `val_loss`, `val_accuracy`, `test_loss`, `test_accuracy`

Examples:
  python -m papers.shared.time_series.plot_metrics /tmp/run_*/
  python -m papers.shared.time_series.plot_metrics /tmp/run_2026... --include-val
    python -m papers.shared.time_series.plot_metrics \
        /tmp/run_a:RNN /tmp/run_b:Photonic --include-val

If you want the QRNN convenience wrapper, use:
  python papers/QRNN/utils/plot_metrics.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


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


def _resolve_run_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name != "metrics.json":
            raise ValueError(
                f"Expected a run directory or a metrics.json file, got file: {path}"
            )
        return path.parent
    return path


def _load_history(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {run_dir}")

    metrics = json.loads(metrics_path.read_text())
    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(
            f"Invalid metrics.json (missing non-empty 'history') in {run_dir}"
        )
    return history


def _series(history: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for row in history:
        if key not in row:
            raise KeyError(f"Missing '{key}' in history row keys={list(row.keys())}")
        out.append(float(row[key]))
    return out


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
    - /path/to/run/metrics.json
    - /path/to/run:My label
    - /path/to/run/metrics.json:My label

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
    paths: list[Path] = []
    labels: list[str | None] = []
    for spec in specs:
        path, label = _maybe_parse_run_spec(spec)
        paths.append(path)
        labels.append(label)

    if any(lbl is not None for lbl in labels):
        resolved = [_resolve_run_dir(p) for p in paths]
        finalized = [
            lbl if lbl is not None else rd.name for rd, lbl in zip(resolved, labels)
        ]
        return paths, finalized

    return paths, None


def plot_runs(
    run_dirs: list[Path],
    *,
    out_path: Path | None = None,
    show: bool = False,
    include_val: bool = False,
    labels: list[str] | None = None,
) -> Path:
    run_dirs = [_resolve_run_dir(p) for p in run_dirs]
    if labels is not None and len(labels) != len(run_dirs):
        raise ValueError("labels must match run_dirs length")

    maybe_use_agg_backend(show=show)
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

    for idx, run_dir in enumerate(run_dirs):
        history = _load_history(run_dir)
        epochs = _series(history, "epoch")

        test_acc = _series(history, "test_accuracy")
        test_loss = _series(history, "test_loss")

        base_label = labels[idx] if labels is not None else run_dir.name
        label = _legend_label(run_dir, base_label)

        ax1.plot(epochs, test_acc, label=label, linewidth=1.8)
        ax2.plot(epochs, test_loss, label=label, linewidth=1.8)

        if include_val:
            val_acc = _series(history, "val_accuracy")
            val_loss = _series(history, "val_loss")
            ax1.plot(epochs, val_acc, linestyle="--", linewidth=1.2, alpha=0.8)
            ax2.plot(epochs, val_loss, linestyle="--", linewidth=1.2, alpha=0.8)

    ax1.set_title("Accuracy")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("test_acc (%)")
    ax1.grid(True, alpha=0.25)

    ax2.set_title("Loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test_loss")
    ax2.grid(True, alpha=0.25)

    base_title = "Time-series learning curves"
    if include_val:
        base_title += " (solid=test, dashed=val)"

    fig.suptitle(base_title)

    handles, legend_labels = ax1.get_legend_handles_labels()
    if handles:
        # Put the legend under both subplots so it doesn't cover curves.
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

    # Leave room at the bottom for the legend.
    bottom = 0.12 + 0.02 * min(len(handles), 10)
    fig.tight_layout(rect=[0, bottom, 1, 0.95])

    if out_path is None:
        suffix = "metrics_overlay.png" if len(run_dirs) > 1 else "metrics_plot.png"
        out_path = run_dirs[0] / suffix

    out_path = save_figure(plt, out_path, dpi=150)

    if show:
        plt.show()

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot time-series training curves from one or more run directories."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=str,
        help=(
            "One or more run directories (or metrics.json files). "
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
        "--show",
        action="store_true",
        help="Display the figure in a window after saving",
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Also plot validation curves (dashed)",
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

    run_paths, spec_labels = _parse_run_specs(list(args.run_dirs))
    flag_labels = _parse_labels(args.labels, expected=len(run_paths))
    if spec_labels is not None and flag_labels is not None:
        raise ValueError(
            "Provide labels either via 'path:label' or via --labels, not both"
        )
    labels = spec_labels if spec_labels is not None else flag_labels

    out_path = plot_runs(
        [Path(p) for p in run_paths],
        out_path=args.out_path,
        show=args.show,
        include_val=args.include_val,
        labels=labels,
    )
    print(f"Saved plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
