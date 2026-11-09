"""Plot a QPE_GNN sweep from its summary and run histories.

Reads ``sweep_summary.json`` (written by ``utils/summarise_sweep.py``) and
each included run's ``metrics.json`` and draws two figures:

* ``<name>_test_metric.png``: the test metric per candidate, mean and sample
  standard deviation over seeds, every seed as a point, and optional paper
  reference values as horizontal marks;
* ``<name>_validation_curves.png``: the validation metric per epoch, mean
  over seeds with the per-seed range shaded.

The figures go to ``--output-dir`` and to the folder holding the summary.
Every figure title carries the result-scope label given by ``--scope``.

Usage (from ``papers/QPE_GNN``)::

    python utils/plot_sweep.py --summary results/synthetic_d2/sweep_summary.json \
        --sweep-dir outdir/sweep_synthetic --name synthetic_d2 \
        --scope "paper protocol, 4 seeds" --output-dir results/synthetic_d2 \
        --paper-value synthetic_quantum=1.0 --paper-value synthetic_rrwp=0.45
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_folder(sweep_dirs: list[Path], run_dir: str) -> Path:
    """Locate a summary ``run_dirs`` entry, prefixed by its folder when combined."""
    if len(sweep_dirs) == 1:
        return sweep_dirs[0] / run_dir
    folder_name, _, inner = run_dir.partition("/")
    for folder in sweep_dirs:
        if folder.name == folder_name:
            return folder / inner
    raise ValueError(f"{run_dir} is in none of the sweep folders")


def plot_sweep(
    summary_path: Path,
    sweep_dirs: list[Path],
    name: str,
    scope: str,
    output_dir: Path,
    paper_values: dict[str, float] | None = None,
) -> list[Path]:
    """Draw the test-metric and validation-curve figures of one sweep.

    Parameters
    ----------
    summary_path : pathlib.Path
        ``sweep_summary.json`` of the sweep (or of combined launches).
    sweep_dirs : list[pathlib.Path]
        Sweep folder(s) the summary's ``run_dirs`` refer to.
    name : str
        Figure file-name prefix.
    scope : str
        Result-scope label shown in every title (e.g. "reduced budget,
        partial").
    output_dir : pathlib.Path
        Curated output folder (normally under ``results/``).
    paper_values : dict[str, float] | None
        Paper-reported value per candidate, drawn as reference marks.
        Default value is None (no marks).

    Returns
    -------
    list[pathlib.Path]
        Every figure file written.
    """
    summary = _read_json(summary_path)
    metric = summary["metric"]
    metric_label = metric
    validation_key = "val_" + metric.removeprefix("test_")
    candidates = summary["candidates"]
    names = [record["candidate"] for record in candidates]
    positions = np.arange(len(names))
    wrapped_scope = textwrap.fill(f"[{scope}]", 70)

    test_figure, test_axis = plt.subplots(figsize=(1.6 * len(names) + 2, 4.2))
    for position, record in zip(positions, candidates, strict=True):
        if not record["values"]:
            continue
        test_axis.bar(
            position,
            record["mean"],
            yerr=record["std"] or 0.0,
            capsize=4,
            color="#4C72B0",
            alpha=0.75,
        )
        jitter = np.linspace(-0.12, 0.12, len(record["values"]))
        test_axis.scatter(
            position + jitter, record["values"], color="black", s=14, zorder=3
        )
    if paper_values:
        for position, candidate in zip(positions, names, strict=True):
            if candidate in paper_values:
                test_axis.hlines(
                    paper_values[candidate],
                    position - 0.4,
                    position + 0.4,
                    colors="#C44E52",
                    linestyles="--",
                    label="paper" if position == positions[0] else None,
                )
        handles, labels = test_axis.get_legend_handles_labels()
        if "paper" not in labels:
            test_axis.plot([], [], color="#C44E52", linestyle="--", label="paper")
        test_axis.legend(loc="best")
    test_axis.set_xticks(positions, names, rotation=20, ha="right")
    test_axis.set_ylabel(metric_label)
    test_axis.set_title(
        f"{summary['sweep_id']}: {metric_label} "
        f"({summary['observed_runs']}/{summary['expected_runs']} runs)\n"
        f"{wrapped_scope}",
        fontsize=9,
    )
    test_figure.tight_layout()

    curve_figure, curve_axis = plt.subplots(figsize=(7, 4.2))
    for record in candidates:
        histories = [
            _read_json(_run_folder(sweep_dirs, run_dir) / "metrics.json")["history"][
                validation_key
            ]
            for run_dir in record["run_dirs"]
        ]
        if not histories:
            continue
        length = min(len(history) for history in histories)
        stacked = np.asarray([history[:length] for history in histories])
        epochs = np.arange(1, length + 1)
        curve_axis.plot(epochs, stacked.mean(axis=0), label=record["candidate"])
        curve_axis.fill_between(
            epochs, stacked.min(axis=0), stacked.max(axis=0), alpha=0.15
        )
    curve_axis.set_xlabel("epoch")
    curve_axis.set_ylabel(validation_key)
    # Errors shrink by an order of magnitude over training; a log axis keeps
    # the late epochs, where candidates separate, readable.
    if summary["direction"] == "minimize":
        curve_axis.set_yscale("log")
    curve_axis.set_title(
        f"{summary['sweep_id']}: validation curves (mean, seed range)\n{wrapped_scope}",
        fontsize=9,
    )
    curve_axis.legend(fontsize=8)
    curve_figure.tight_layout()

    written = []
    for folder in dict.fromkeys([output_dir, summary_path.parent]):
        folder.mkdir(parents=True, exist_ok=True)
        for suffix, figure in (
            ("test_metric", test_figure),
            ("validation_curves", curve_figure),
        ):
            path = folder / f"{name}_{suffix}.png"
            figure.savefig(path, dpi=150)
            written.append(path)
    plt.close(test_figure)
    plt.close(curve_figure)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--sweep-dir", type=Path, action="append", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--paper-value",
        action="append",
        default=[],
        metavar="CANDIDATE=VALUE",
        help="Paper-reported value for one candidate; repeat as needed",
    )
    args = parser.parse_args(argv)

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_DIR / path

    paper_values = {}
    for item in args.paper_value:
        candidate, _, value = item.partition("=")
        paper_values[candidate] = float(value)
    for path in plot_sweep(
        resolve(args.summary),
        [resolve(folder) for folder in args.sweep_dir],
        args.name,
        args.scope,
        resolve(args.output_dir),
        paper_values,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
