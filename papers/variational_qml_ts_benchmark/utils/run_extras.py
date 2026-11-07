"""Extra runs: (a) ru-QNN bugfix A/B, (b) photonic (MerLin) dressed QNN.

Run AFTER the main sweep to avoid CPU oversubscription.

(a) trains the ru-QNN with bugfix=0 (original 1/3-scaled exponential encoding)
    and bugfix=1 (paper-faithful) on the two Henon problems x 3 seeds, to measure
    the impact of BUGS.md #2.  Writes results/ruqnn_ab/... and a markdown A/B.
(b) trains the photonic dressed QNN on the same two problems x 3 seeds and writes
    the runs into results/sweep/ so plot_sweep.py includes the photonic bar.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RUN_ONE = PROJECT_DIR / "utils" / "run_experiment.py"
PY = sys.executable
RU = "ruexp_EYX_EZY_CX_CY_X_CZ_X_CZ_EXY_EXX_EZZ_X_EYZ_EXY_EZX_Y_EYX_CY_X_CY"
PROBLEMS = [("henon_1000", 4, 1, "henon_p1"), ("henon_1000", 4, 4, "henon_p4")]
SEEDS = [0, 1, 2]
EPOCHS = 400
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def _run(cmd):
    t = time.time()
    p = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    return p.returncode == 0, round(time.time() - t, 1), p.stderr[-400:]


def _mad(x):
    return float(np.median(np.abs(np.array(x) - np.median(x))))


def _ruqnn_cmd(dataset, seq, pred, tag, bf, seed):
    out = PROJECT_DIR / "results" / "ruqnn_ab" / f"ruqnn_{tag}_bf{bf}_s{seed}"
    cmd = [
        PY,
        str(RUN_ONE),
        "--model",
        "vqc",
        "--ansatz",
        RU,
        "--num-qubits",
        "4",
        "--dataset",
        dataset,
        "--sequence-length",
        str(seq),
        "--prediction-step",
        str(pred),
        "--seed",
        str(seed),
        "--epochs",
        str(EPOCHS),
        "--bugfix",
        str(bf),
        "--out",
        str(out),
    ]
    return cmd, out, tag, bf, seed


def ruqnn_ab(pool):
    from concurrent.futures import as_completed

    jobs = [
        _ruqnn_cmd(dataset, seq, pred, tag, bf, seed)
        for dataset, seq, pred, tag in PROBLEMS
        for bf in (0, 1)
        for seed in SEEDS
    ]
    futs = {pool.submit(_run, c[0]): c for c in jobs}
    rows = []
    for fut in as_completed(futs):
        cmd, out, tag, bf, seed = futs[fut]
        ok, dt, err = fut.result()
        print(f"ruqnn {tag} bf{bf} s{seed}: {'OK' if ok else 'FAIL'} {dt}s", flush=True)
        if ok:
            d = json.loads((out / "metrics.json").read_text())
            rows.append(
                {"problem": tag, "bugfix": bf, "seed": seed, "mse_test": d["mse_test"]}
            )
    df = pd.DataFrame(rows).sort_values(["problem", "bugfix", "seed"])
    lines = [
        "# ru-QNN bugfix A/B (BUGS.md #2: exponential encoding prefactor)\n",
        "Original encoding uses prefactor 3^(j-n) (top qubit 1/3); the fix uses "
        "3^(j-n+1) (top qubit 1, matching the paper). 4 qubits, 400 epochs, 3 seeds.\n",
    ]
    for tag, g in df.groupby("problem"):
        lines.append(f"\n## {tag}\n")
        tbl = g.groupby("bugfix")["mse_test"].agg(["median", "min", "max"])
        tbl.index = ["original (buggy)", "fixed"]
        lines.append(tbl.to_markdown(floatfmt=".5g"))
    (PROJECT_DIR / "results" / "ruqnn_bugfix_ab.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    df.to_csv(PROJECT_DIR / "results" / "ruqnn_bugfix_ab.csv", index=False)
    print("\n".join(lines))


def photonic(pool):
    from concurrent.futures import as_completed

    jobs = []
    for dataset, seq, pred, tag in PROBLEMS:
        for seed in SEEDS:
            out = PROJECT_DIR / "results" / "sweep" / f"photonic_{tag}_s{seed}_bf1"
            cmd = [
                PY,
                str(RUN_ONE),
                "--model",
                "photonic",
                "--ansatz",
                "photonic",
                "--num-qubits",
                "6",
                "--hidden-size",
                "3",
                "--dataset",
                dataset,
                "--sequence-length",
                str(seq),
                "--prediction-step",
                str(pred),
                "--seed",
                str(seed),
                "--epochs",
                str(EPOCHS),
                "--bugfix",
                "1",
                "--out",
                str(out),
            ]
            jobs.append((cmd, tag, seed))
    futs = {pool.submit(_run, c[0]): c for c in jobs}
    for fut in as_completed(futs):
        cmd, tag, seed = futs[fut]
        ok, dt, err = fut.result()
        print(f"photonic {tag} s{seed}: {'OK' if ok else 'FAIL'} {dt}s", flush=True)
        if not ok:
            print("   ", err.replace("\n", " ")[-300:], flush=True)


def main():
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=6) as pool:
        print("=== ru-QNN bugfix A/B + photonic dressed QNN (parallel) ===", flush=True)
        # submit photonic first (cheap), then run A/B which collects results
        photonic(pool)
        ruqnn_ab(pool)
    subprocess.run(
        [PY, str(PROJECT_DIR / "utils" / "sweep.py"), "--aggregate-only"], env=ENV
    )
    print("extras done", flush=True)


if __name__ == "__main__":
    main()
