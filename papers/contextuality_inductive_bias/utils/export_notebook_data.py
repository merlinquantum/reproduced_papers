"""Export compact per-run metrics and learning curves into ``results/``.

Usage (from the repository root)::

    python3 papers/contextuality_inductive_bias/utils/export_notebook_data.py \\
        --sweeps-root papers/contextuality_inductive_bias/outdir/sweeps \\
        --out-dir papers/contextuality_inductive_bias/results

``outdir/`` is disposable and git-ignored, so the notebook reads the stored
full runs from two curated files written here:

- ``per_run_metrics.csv``: one row per COMPLETED run of every non-tuning sweep (sweep,
  run directory name, system name, protocol, training data, seed, lr, L2,
  final and early-stopped test KL, final train / validation NLL, parameter
  count, training-loop seconds).
- ``learning_curves.csv``: per-step test KL and train NLL of the Fig. 8 and
  fair-comparison sweeps, aggregated over seeds (mean, sample std, n) every
  ``--curve-stride`` steps (plus the last step).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_io import describe, load_sweep  # noqa: E402

# Tuning sweeps are excluded: their complete candidate tables live in
# results/fair_tuning_selection.md and results/sweep_index.md, and fair_tuning
# carries the coordinator status FAILED (see LOG.md, Amendment 22:37Z).
TUNING_SWEEPS = ("fair_tuning", "fair_tuning_ext", "fair_tuning_ext2")
CURVE_SWEEPS = (
    "fig8_quantum_paper",
    "fig8_quantum_code_fork",
    "fig8_surrogates",
    "fair_final",
)
RUN_FIELDS = (
    "sweep_id",
    "run_dir",
    "system",
    "protocol",
    "train_source",
    "train_seed",
    "seed",
    "lr",
    "l2",
    "final_test_kl",
    "early_stop_test_kl",
    "final_train_nll",
    "final_validation_nll",
    "n_parameters",
    "train_seconds",
)


def _optional(metrics: dict, section: str, key: str) -> float | str:
    # Runs without a validation split store min_validation_nll = null.
    block = metrics.get(section)
    value = None if block is None else block.get(key)
    return "" if value is None else float(value)


def export(sweeps_root: Path, out_dir: Path, curve_stride: int) -> tuple[int, int]:
    """Write both CSV files and return (run rows, curve rows)."""
    sweep_dirs = sorted(
        path.parent
        for path in sweeps_root.glob("*/sweep_status.json")
        if path.parent.name not in TUNING_SWEEPS
    )
    if not sweep_dirs:
        raise FileNotFoundError(f"No sweep_status.json under {sweeps_root}")
    run_rows = []
    curves: dict[tuple, list[dict]] = defaultdict(list)
    for sweep_dir in sweep_dirs:
        for record in load_sweep(sweep_dir):
            candidate, metrics = record.candidate, record.metrics
            system = describe(candidate)
            run_rows.append(
                {
                    "sweep_id": record.sweep_id,
                    "run_dir": record.run_dir.name,
                    "system": system,
                    "protocol": candidate["protocol"],
                    "train_source": candidate["train_source"],
                    "train_seed": ""
                    if candidate["train_seed"] is None
                    else candidate["train_seed"],
                    "seed": record.seed,
                    "lr": candidate["lr"],
                    "l2": candidate["l2"],
                    "final_test_kl": float(metrics["final"]["test_kl"]),
                    "early_stop_test_kl": _optional(
                        metrics, "min_validation_nll", "test_kl"
                    ),
                    "final_train_nll": float(metrics["final"]["train_nll"]),
                    "final_validation_nll": _optional(
                        metrics, "final", "validation_nll"
                    ),
                    "n_parameters": int(metrics["n_parameters"]),
                    "train_seconds": float(metrics["train_seconds"]),
                }
            )
            if (
                record.sweep_id in CURVE_SWEEPS
                and candidate["train_source"] == "author"
            ):
                curves[(record.sweep_id, system, candidate["protocol"])].append(
                    record.history()
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_run_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(run_rows)
    curve_rows = 0
    with (out_dir / "learning_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sweep_id",
                "system",
                "protocol",
                "step",
                "n",
                "test_kl_mean",
                "test_kl_std",
                "train_nll_mean",
                "train_nll_std",
            ]
        )
        for (sweep_id, system, protocol), histories in sorted(curves.items()):
            steps = np.array(histories[0]["step"])
            if any(not np.array_equal(np.array(h["step"]), steps) for h in histories):
                raise ValueError(
                    f"{sweep_id} {system}: runs were evaluated at different steps"
                )
            test_kl = np.array([h["test_kl"] for h in histories])
            train_nll = np.array([h["train_nll"] for h in histories])
            keep = [
                i
                for i, step in enumerate(steps)
                if step % curve_stride == 0 or i == len(steps) - 1
            ]
            n = len(histories)
            for i in keep:
                writer.writerow(
                    [
                        sweep_id,
                        system,
                        protocol,
                        int(steps[i]),
                        n,
                        test_kl[:, i].mean(),
                        test_kl[:, i].std(ddof=1) if n > 1 else math.nan,
                        train_nll[:, i].mean(),
                        train_nll[:, i].std(ddof=1) if n > 1 else math.nan,
                    ]
                )
                curve_rows += 1
    return len(run_rows), curve_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweeps-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--curve-stride", type=int, default=10)
    args = parser.parse_args(argv)
    runs, curve_rows = export(args.sweeps_root, args.out_dir, args.curve_stride)
    print(
        f"{runs} runs -> per_run_metrics.csv; {curve_rows} curve rows -> learning_curves.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
