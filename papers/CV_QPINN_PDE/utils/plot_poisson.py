"""Plot the predicted u(x) for the Poisson 1D experiment against the analytic
solution. Reads `predictions.json` from a run directory and writes a PNG.

Usage:

    python utils/plot_poisson.py <run_dir> [<run_dir> ...] --out results/poisson_compare.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(run_dir: Path) -> tuple[str, dict]:
    summary = json.loads((run_dir / "summary.json").read_text())
    predictions = json.loads((run_dir / "predictions.json").read_text())
    return summary["experiment"], {**summary, **predictions}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in args.run_dirs:
        label_root, data = _load(d)
        x = np.array(data["x"])
        u_pred = np.array(data["u_pred"])
        u_ref = np.array(data["u_ref"])
        ax.plot(x, u_pred, label=f"{label_root} (RMSE={data['metrics']['rmse']:.2e})",
                linewidth=1.6)
    ax.plot(x, u_ref, label="Analytic", color="k", linestyle="--", linewidth=1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("Poisson 1D — QPINN / PINN / MerLin predictions vs analytic")
    ax.legend(fontsize="small")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
