"""Photonic-reservoir extension sweep (NOT part of the paper reproduction).

Runs the two non-variational photonic reservoirs of ``lib/reservoir.py``
alongside the variational photonic d-QNN and the paper's classical baselines,
inside the paper's own protocol (same datasets, splits, windowing, metric and
"median over seeds of the best-validation model" reporting rule).

Two budget arms are run for every cell:

``fixed``
    400 epochs for every model -- directly comparable to ``utils/sweep.py`` and
    to ``results/sweep_table.md``.

``conv``
    The paper's validation-plateau convergence criterion (min 400 epochs, two
    200-epoch windows), capped at ``--max-epochs``.  The reservoirs train only a
    linear readout and the classical baselines are small, so this arm is cheap
    -- and it removes the "the classical baselines were under-trained" confound
    that makes the fixed-budget arm unreliable on the hard tasks.

Task coverage is widened from the 2 Henon problems of the reduced sweep to 6
problems spanning all three dynamical systems at an easy and a hard horizon.

Outputs ``results/reservoir_summary.csv``.
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

# model, display, ansatz, num_qubits/n_modes, hidden_size/n_photons
MODELS = [
    ("mlp", "MLP", "relu_16_16", None, None),
    ("rnn", "RNN", "layers_1", None, 16),
    ("lstm", "LSTM", "layers_1", None, 16),
    ("photonic", "photonic-dQNN", "photonic", 6, 3),
    ("photonic_reservoir", "photonic-RC", "reservoir", 6, 3),
    ("photonic_memristor", "photonic-memRC", "reservoir", 6, 3),
    # Capacity control: same sequential readout as photonic-memRC, no optical
    # memory. Isolates the memristor's contribution from readout size.
    ("photonic_seqreservoir", "photonic-seqRC", "reservoir", 6, 3),
]

# Six of the paper's 27 learning problems: every dynamical system at its
# easiest (k=1) and hardest (full-Lyapunov) horizon, at sequence length 4.
PROBLEMS = [
    ("henon_1000", 4, 1, "henon_p1"),
    ("henon_1000", 4, 4, "henon_p4"),
    ("mackey_1000", 4, 1, "mackey_p1"),
    ("mackey_1000", 4, 140, "mackey_p140"),
    ("lorenz_1000", 4, 1, "lorenz_p1"),
    ("lorenz_1000", 4, 25, "lorenz_p25"),
]

RESULT_DIR = PROJECT_DIR / "results" / "reservoir"


def run_cell(spec: dict) -> dict:
    out = RESULT_DIR / spec["runid"]
    if (out / "metrics.json").exists():
        spec.update(ok=True, wall_s=0.0, skipped=True)
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
        "--use-convergence",
        str(spec["use_convergence"]),
        "--bugfix",
        "1",
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


def build_specs(seeds: list[int], arms: list[str], max_epochs: int) -> list[dict]:
    specs = []
    for arm in arms:
        use_conv = 1 if arm == "conv" else 0
        epochs = max_epochs if arm == "conv" else 400
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
                            "arm": arm,
                            "epochs": epochs,
                            "use_convergence": use_conv,
                            "runid": f"{arm}__{disp}__{tag}__s{seed}",
                        }
                    )
    return specs


def aggregate() -> pd.DataFrame:
    rows = []
    for mj in RESULT_DIR.rglob("metrics.json"):
        d = json.loads(mj.read_text())
        arm, disp, tag, _ = mj.parent.name.split("__")
        d.update(arm=arm, display=disp, tag=tag)
        rows.append(d)
    df = pd.DataFrame(rows)
    out = PROJECT_DIR / "results" / "reservoir_summary.csv"
    df.to_csv(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=["fixed", "conv"])
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=3000,
        help="Hard cap for the convergence arm (documented as a limitation).",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if args.aggregate_only:
        df = aggregate()
        print(f"Aggregated {len(df)} runs -> results/reservoir_summary.csv")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args.seeds, args.arms, args.max_epochs)
    print(
        f"Launching {len(specs)} runs on {args.workers} workers "
        f"(arms={args.arms}, seeds={args.seeds}, cap={args.max_epochs})",
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
    print(f"\nDone. Aggregated {len(df)} runs -> results/reservoir_summary.csv")


if __name__ == "__main__":
    main()
