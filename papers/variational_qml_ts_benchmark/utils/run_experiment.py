"""Run a single benchmark cell and write metrics.json to a target directory.

Thin CLI wrapper around ``lib.runner.train_and_evaluate`` used by the sweep
launcher.  Kept separate from the shared-runtime CLI so a sweep can spawn many
isolated subprocesses cheaply.

Example
-------
    python utils/run_experiment.py --model qrnn --ansatz paper_no_reset \
        --num-qubits 4 --hidden-size 2 --dataset henon_1000 \
        --sequence-length 4 --prediction-step 1 --seed 0 --epochs 250 \
        --bugfix 1 --out results/sweep/qrnn_henon_p1_s0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Pin to a single thread per process so a concurrent sweep does not oversubscribe
# the CPU (torch/openblas otherwise each spawn one thread per core).
for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")
import torch  # noqa: E402

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(PROJECT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.runner import train_and_evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ansatz", required=True)
    ap.add_argument("--num-qubits", type=int, default=None)
    ap.add_argument("--hidden-size", type=int, default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sequence-length", type=int, required=True)
    ap.add_argument("--prediction-step", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--bugfix", type=int, default=1)
    ap.add_argument("--use-convergence", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = {
        "seed": args.seed,
        "dtype": "float32",
        "dataset": {
            "name": args.dataset,
            "root": str(
                PROJECT_DIR.parent.parent / "data" / "variational_qml_ts_benchmark"
            ),
            "sequence_length": args.sequence_length,
            "prediction_step": args.prediction_step,
            "batch_size": args.batch_size,
        },
        "model": {
            "name": args.model,
            "params": {
                "ansatz": args.ansatz,
                "num_qubits": args.num_qubits,
                "hidden_size": args.hidden_size,
                "bugfix": bool(args.bugfix),
            },
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "use_convergence": bool(args.use_convergence),
            "min_epochs": 400,
            "window": 200,
        },
    }
    train_and_evaluate(cfg, Path(args.out))


if __name__ == "__main__":
    main()
