"""Concurrent reduced-compute benchmark sweep.

Launches one subprocess per (model, learning-problem, seed) cell of a reduced
grid, with a bounded number of concurrent workers, then aggregates every
``metrics.json`` into ``results/sweep_summary.csv``.

This is the REDUCED (V2) reproduction: a single representative hyperparameter
configuration per model (not the paper's full grid search), a fixed epoch
budget shared by all models, and 3 seeds.  The full grid-optimised claim is
reproduced separately from the authors' released result CSVs
(see utils/plot_paper_figures.py).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RUN_ONE = PROJECT_DIR / "utils" / "run_experiment.py"
PY = sys.executable

# A representative single configuration per model family.  Hyperparameters are
# chosen mid-range and to keep trainable-parameter counts comparable, matching
# the paper's fairness intent.
RU_HENON = "ruexp_EYX_EZY_CX_CY_X_CZ_X_CZ_EXY_EXX_EZZ_X_EYZ_EXY_EZX_Y_EYX_CY_X_CY"
MODELS = [
    # name,      display,     ansatz,                                    nq,   hs
    ("mlp", "MLP", "relu_16_16", None, None),
    ("rnn", "RNN", "layers_1", None, 16),
    ("lstm", "LSTM", "layers_1", None, 16),
    ("vqc", "d-QNN", "paper_rivera-ruiz_with_inputlayer_2", 6, None),
    ("vqc", "ru-QNN", RU_HENON, 4, None),
    ("qrnn", "QRNN", "paper_no_reset", 4, 2),
    ("qlstm", "QLSTM", "original_2", 4, None),
    ("le_qlstm", "le-QLSTM", "original_2", 6, 8),
]

# Reduced learning problems (dataset, sequence_length, prediction_step, tag).
PROBLEMS = [
    ("henon_1000", 4, 1, "henon_p1"),  # near-linear / easy
    ("henon_1000", 4, 4, "henon_p4"),  # full Lyapunov / hard
]


def run_cell(spec: dict) -> dict:
    out = PROJECT_DIR / "results" / "sweep" / spec["runid"]
    if (out / "metrics.json").exists():
        spec["ok"] = True
        spec["wall_s"] = 0.0
        spec["skipped"] = True
        return spec
    cmd = [
        PY,
        str(RUN_ONE),
        "--model",
        spec["model"],
        "--ansatz",
        spec["ansatz"],
        "--dataset",
        spec["dataset"],
        "--sequence-length",
        str(spec["seq"]),
        "--prediction-step",
        str(spec["pred"]),
        "--seed",
        str(spec["seed"]),
        "--epochs",
        str(spec["epochs"]),
        "--bugfix",
        str(spec["bugfix"]),
        "--out",
        str(out),
    ]
    if spec["nq"] is not None:
        cmd += ["--num-qubits", str(spec["nq"])]
    if spec["hs"] is not None:
        cmd += ["--hidden-size", str(spec["hs"])]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    ok = proc.returncode == 0 and (out / "metrics.json").exists()
    spec["ok"] = ok
    spec["wall_s"] = round(time.time() - t, 1)
    if not ok:
        spec["error"] = proc.stderr[-800:]
    return spec


def build_specs(epochs: int, seeds: list[int], bugfix: int) -> list[dict]:
    specs = []
    for dataset, seq, pred, tag in PROBLEMS:
        for model, disp, ansatz, nq, hs in MODELS:
            for seed in seeds:
                specs.append(
                    {
                        "model": model,
                        "display": disp,
                        "ansatz": ansatz,
                        "nq": nq,
                        "hs": hs,
                        "dataset": dataset,
                        "seq": seq,
                        "pred": pred,
                        "tag": tag,
                        "seed": seed,
                        "epochs": epochs,
                        "bugfix": bugfix,
                        "runid": f"{disp}_{tag}_s{seed}_bf{bugfix}",
                    }
                )
    return specs


def aggregate() -> pd.DataFrame:
    rows = []
    for mj in (PROJECT_DIR / "results" / "sweep").rglob("metrics.json"):
        d = json.loads(mj.read_text())
        d["display"] = mj.parent.name.split("_")[0]
        rows.append(d)
    df = pd.DataFrame(rows)
    out = PROJECT_DIR / "results" / "sweep_summary.csv"
    df.to_csv(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bugfix", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if args.aggregate_only:
        df = aggregate()
        print(f"Aggregated {len(df)} runs -> results/sweep_summary.csv")
        return

    specs = build_specs(args.epochs, args.seeds, args.bugfix)
    print(
        f"Launching {len(specs)} runs on {args.workers} workers "
        f"(epochs={args.epochs}, seeds={args.seeds}, bugfix={args.bugfix})",
        flush=True,
    )
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_cell, s): s for s in specs}
        for fut in as_completed(futs):
            s = fut.result()
            done += 1
            status = "OK " if s["ok"] else "FAIL"
            print(
                f"[{done}/{len(specs)}] {status} {s['runid']} ({s['wall_s']}s)",
                flush=True,
            )
            if not s["ok"]:
                print("   ", s.get("error", "").replace("\n", " ")[-300:], flush=True)
    df = aggregate()
    print(f"\nDone. Aggregated {len(df)} runs -> results/sweep_summary.csv")


if __name__ == "__main__":
    main()
