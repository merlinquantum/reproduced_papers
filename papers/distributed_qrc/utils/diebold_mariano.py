"""Diebold-Mariano tests on the saved test predictions of seed sweeps (headline comparisons).

For each comparison ``(sweep, cell A, cell B)`` and each seed, the loss
differential ``d_t = L(e_A,t) - L(e_B,t)`` on the test split is tested for zero
mean. The statistic uses the Harvey-Leybourne-Newbold small-sample correction
and a Bartlett (Newey-West) long-run variance with ``lag`` autocovariances:
``lag = 0`` is the textbook one-step DM test; ``lag = 24`` allows for the daily
autocorrelation of hourly load errors. Two-sided p-values use Student-t with
T - 1 degrees of freedom. Negative statistics mean cell A has the lower loss.

Comparisons (declared before the runs):
the best paper-rule cell vs the classical + linear baseline at the smallest
configuration of each architecture (where the peak reductions occur), and the
quantum vs classical reservoir with the same ideal kernel readout.

Writes ``results/data/diebold_mariano.json`` (``-R2`` for the evidence sweeps) and prints a table.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/diebold_mariano.py papers/distributed_qrc/outdir
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PAPER_DIR = Path(__file__).resolve().parents[1]
COMPARISONS = [
    ("E1", "srsr:10", "quantum_ideal|quantum_kernel_ideal", "classical|linear"),
    ("E2", "mrsr:2x5", "quantum_ideal|quantum_kernel_ideal", "classical|linear"),
    ("E3", "srmr:10/2", "quantum_ideal|quantum_kernel_ideal", "classical|linear"),
    ("E4", "mrmr:2x5", "quantum_ideal|quantum_kernel_ideal", "classical|linear"),
    (
        "E1",
        "srsr:60",
        "quantum_ideal|quantum_kernel_ideal",
        "classical|quantum_kernel_ideal",
    ),
    (
        "E2",
        "mrsr:3x20",
        "quantum_ideal|quantum_kernel_ideal",
        "classical|quantum_kernel_ideal",
    ),
    ("E1", "srsr:60", "quantum_ideal|quantum_kernel_ideal", "quantum_ideal|linear"),
]


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lag: int) -> tuple[float, float]:
    d = loss_a - loss_b
    t = d.size
    centered = d - d.mean()
    variance = centered @ centered / t
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        variance += 2 * weight * (centered[k:] @ centered[:-k]) / t
    statistic = d.mean() / np.sqrt(variance / t)
    # Harvey-Leybourne-Newbold correction for a one-step-ahead forecast (h = 1).
    statistic *= np.sqrt((t - 1) / t)
    p_value = 2 * stats.t.sf(abs(statistic), df=t - 1)
    return float(statistic), float(p_value)


def main() -> int:
    outdir = Path(sys.argv[1])
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    results = []
    for sweep, config, cell_a, cell_b in COMPARISONS:
        sweep_dir = outdir / f"sweep_{sweep}{suffix}"
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        for run in status["runs"]:
            predictions = np.load(sweep_dir / run["run_dir"] / "test_predictions.npz")
            target = predictions["test_targets"].astype(np.float64)
            pred_a = predictions[f"{config}|{cell_a}"].astype(np.float64)
            pred_b = predictions[f"{config}|{cell_b}"].astype(np.float64)
            for loss_name, loss in (("abs", np.abs), ("sq", np.square)):
                for lag in (0, 24):
                    statistic, p_value = dm_test(
                        loss(target - pred_a), loss(target - pred_b), lag
                    )
                    results.append(
                        {
                            "sweep": sweep,
                            "configuration": config,
                            "cell_a": cell_a,
                            "cell_b": cell_b,
                            "seed": run["seed"],
                            "loss": loss_name,
                            "lag": lag,
                            "dm": statistic,
                            "p_value": p_value,
                        }
                    )
    out = PAPER_DIR / "results" / "data" / "diebold_mariano.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"{'comparison':78s} {'loss':4s} {'lag':>3s} | DM (p) per seed")
    keys = dict.fromkeys(
        (r["sweep"], r["configuration"], r["cell_a"], r["cell_b"], r["loss"], r["lag"])
        for r in results
    )
    for key in keys:
        rows = [
            r
            for r in results
            if (
                r["sweep"],
                r["configuration"],
                r["cell_a"],
                r["cell_b"],
                r["loss"],
                r["lag"],
            )
            == key
        ]
        label = f"{key[0]} {key[1]}: {key[2]} vs {key[3]}"
        cells = "  ".join(f"{r['dm']:+7.2f} ({r['p_value']:.1e})" for r in rows)
        print(f"{label:78s} {key[4]:4s} {key[5]:3d} | {cells}")
    print(f"wrote {os.path.relpath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
