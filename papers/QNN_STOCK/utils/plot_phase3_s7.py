"""Sweeps S7 (E12) and S8 (E12b): QBGU vs per-sample SGD at equal shots.

``--design s7``: paper HEA, QBGU vs per-sample, two shot allocations.
``--design s8``: context-preserving order-1 circuit, QBGU vs per-sample
mini-batches of 1, 100 and 10,000 windows. Output names carry the prefix.

Reads ``sweep_status.json``, every run's ``metrics.json``, ``losses.csv`` and
``distributions.json``, and writes to ``--out`` and ``<sweep_dir>/figures``:

- ``<design>_summary.csv``: per (data, asset, arm) mean and std over seeds of the
  final QBGU fidelity, the final expected per-sample fidelity, the test
  log-loss and accuracy of the paper's inference readout, the test log-loss
  of the output readout, and the mean confidence ``max_o q(o | c)`` of the
  inference conditional (1 = deterministic predictions).
- ``<design>_objective_optimum.csv``: the best reachable value of each objective per
  asset (QBGU fidelity 1; per-sample fidelity ``sum_c max_o P(c, o)``).
- ``fig_<design>_objectives.png``: both objectives against cumulative shots.

Usage::

    python papers/QNN_STOCK/utils/plot_phase3_s7.py <sweep_dir> --design s7         --out results/phase3_s7
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR / "utils"))

from plot_phase3_s1 import INK_2, SERIES, _save, _style, plt  # noqa: E402

# Per design: scope line, arms (label, override suffix or "" for the plain
# config, color index, line style) and datasets (label, config stem).
DESIGNS = {
    "s7": {
        "scope": "substitute-data · seeds 42–46 · QSTL paper HEA "
        "(3 context + 1 prediction qubit)",
        "arms": (
            ("QBGU 3000 × 10⁴ shots", "", 0, "-"),
            ("Per-sample 3000 × 10⁴ shots", "ps_e3000", 1, "-"),
            ("QBGU 3×10⁵ × 100 shots", "qbgu_e300k_s100", 0, "--"),
            ("Per-sample 3×10⁵ × 100 shots", "ps_e300k_s100", 1, "--"),
        ),
        "datasets": (("raw", "e3_qstl_original"), ("MA5", "e3_qstl_ma5")),
    },
    "s8": {
        "scope": "substitute-data · seeds 42–46 · context-preserving order-1 "
        "circuit (3 context + 1 prediction qubit) · 3000 × 10⁴ shots",
        "arms": (
            ("QBGU", "", 0, "-"),
            ("Per-sample, batch 1", "ps_b1", 1, ":"),
            ("Per-sample, batch 100", "ps_b100", 1, "--"),
            ("Per-sample, batch 10⁴", "ps_b10000", 1, "-"),
        ),
        "datasets": (("raw", "e11_qstl_cc1"), ("MA5", "e11_qstl_cc1_ma5")),
    },
}


def _runs(sweep_dir: Path) -> dict[tuple[str, int], Path]:
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    runs = {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            raise ValueError(f"run {run['run_id']} is {run['status']}")
        runs[(run["candidate"]["config"], run["seed"])] = (
            sweep_dir / run["metrics_path"]
        ).parent
    return runs


def _config(stem: str, suffix: str) -> str:
    return f"{stem}__{suffix}" if suffix else stem


def _history(run_dir: Path, asset: str) -> list[dict]:
    with (run_dir / "losses.csv").open(encoding="utf-8") as handle:
        return [
            row for row in csv.DictReader(handle) if row["stage"] == f"qstl_{asset}"
        ]


def _shots_per_iteration(run_dir: Path) -> int:
    snapshot = json.loads(
        (run_dir / "config_snapshot.json").read_text(encoding="utf-8")
    )
    return 2 * int(snapshot["training"]["shots"])


def collect(sweep_dir: Path, design: dict):
    runs = _runs(sweep_dir)
    seeds = sorted({seed for _, seed in runs})
    values = defaultdict(list)
    curves = defaultdict(list)
    optimum = {}
    for data, stem in design["datasets"]:
        for label, suffix, _, _ in design["arms"]:
            config = _config(stem, suffix)
            for seed in seeds:
                run_dir = runs[(config, seed)]
                metrics = json.loads(
                    (run_dir / "metrics.json").read_text(encoding="utf-8")
                )
                tables = json.loads(
                    (run_dir / "distributions.json").read_text(encoding="utf-8")
                )
                shots = _shots_per_iteration(run_dir)
                for asset, block in metrics["assets"].items():
                    model = block["models"]["QSTL"]
                    history = _history(run_dir, asset)
                    last = history[-1]
                    conditional = np.asarray(
                        tables[asset]["QSTL_inference_conditional"]
                    )
                    joint = np.asarray(tables[asset]["train_joint"])
                    optimum[(data, asset)] = float(joint.max(axis=1).sum())
                    key = (data, asset, label)
                    values[key + ("final_fidelity",)].append(model["final_fidelity"])
                    values[key + ("per_sample_fidelity",)].append(
                        float(last["per_sample_fidelity"])
                    )
                    values[key + ("inference_test_log_loss",)].append(
                        model["inference"]["test"]["log_loss"]
                    )
                    values[key + ("inference_test_accuracy",)].append(
                        model["inference"]["test"]["accuracy"]
                    )
                    values[key + ("output_test_log_loss",)].append(
                        model["output"]["test"]["log_loss"]
                    )
                    values[key + ("inference_confidence",)].append(
                        float(conditional.max(axis=1).mean())
                    )
                    curves[key].append(
                        (
                            np.array([int(r["stage_epoch"]) * shots for r in history]),
                            np.array([float(r["fidelity"]) for r in history]),
                            np.array(
                                [float(r["per_sample_fidelity"]) for r in history]
                            ),
                        )
                    )
    return values, curves, optimum


def write_tables(values, optimum, outputs: list[Path], prefix: str) -> None:
    rows = []
    for (data, asset, label, metric), series in sorted(values.items()):
        rows.append(
            {
                "data": data,
                "asset": asset,
                "arm": label,
                "metric": metric,
                "n": len(series),
                "mean": statistics.fmean(series),
                "std": statistics.stdev(series) if len(series) > 1 else 0.0,
            }
        )
    optimum_rows = [
        {
            "data": data,
            "asset": asset,
            "qbgu_fidelity_max": 1.0,
            "per_sample_fidelity_max": value,
        }
        for (data, asset), value in sorted(optimum.items())
    ]
    for out in outputs:
        out.mkdir(parents=True, exist_ok=True)
        for name, table in (
            (f"{prefix}_summary.csv", rows),
            (f"{prefix}_objective_optimum.csv", optimum_rows),
        ):
            with (out / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)


def plot_objectives(
    curves, optimum, outputs: list[Path], design: dict, prefix: str
) -> None:
    """Rows: the two objectives; columns: (data, asset). Mean over seeds."""

    _style()
    panels = sorted({(data, asset) for data, asset, _ in curves})
    fig, axes = plt.subplots(
        2, len(panels), figsize=(4.2 * len(panels), 7.0), sharex=True, squeeze=False
    )
    for column, (data, asset) in enumerate(panels):
        for row, (index, title) in enumerate(
            ((1, "QBGU objective: fidelity F"), (2, "Per-sample objective"))
        ):
            ax = axes[row][column]
            for label, _, color, style in design["arms"]:
                runs = curves[(data, asset, label)]
                shots = runs[0][0]
                mean = np.mean([run[index] for run in runs], axis=0)
                ax.plot(shots, mean, color=SERIES[color], linestyle=style, label=label)
            ceiling = 1.0 if index == 1 else optimum[(data, asset)]
            ax.axhline(ceiling, color=INK_2, linewidth=0.8, linestyle=":")
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.02)
            if row == 0:
                ax.set_title(f"{asset} · {data}")
            if column == 0:
                ax.set_ylabel(title)
            if row == 1:
                ax.set_xlabel("Cumulative shots")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "QBGU vs per-sample SGD at equal total shots (dotted: best reachable value)",
        fontsize=11,
    )
    fig.text(0.01, -0.01, design["scope"], fontsize=8, color=INK_2)
    fig.tight_layout()
    _save(fig, f"fig_{prefix}_objectives", outputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--design", choices=sorted(DESIGNS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    design = DESIGNS[args.design]
    outputs = [args.sweep_dir / "figures", args.out]
    values, curves, optimum = collect(args.sweep_dir, design)
    write_tables(values, optimum, outputs, args.design)
    plot_objectives(curves, optimum, outputs, design, args.design)
    print(f"{args.design} tables and figure written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
