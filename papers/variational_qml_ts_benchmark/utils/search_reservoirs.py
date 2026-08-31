"""Hyperparameter search for the photonic reservoirs.

The reservoir results in ``results/reservoir_table.md`` used a single a-priori
configuration (6 modes, 3 photons, scale=pi, leak=0.5, 2 memristors), so the
reservoirs' last-place ranking there is only a *lower bound* on what the topology
can do.  This script searches the configuration space to find out how much of
that gap is the topology and how much was an unlucky guess.

Methodology
-----------
* **Selection is on validation MSE, never test.**  The pipeline records both;
  every "best" decision here uses ``mse_val``, and test numbers are only read
  out afterwards.  Otherwise the search would simply overfit the test set.
* **Search on Hénon, evaluate on all six tasks.**  Stages 1-2 tune on
  ``henon_p1`` (easy) and ``henon_p4`` (hard) only.  Stage 3 then evaluates the
  winning configuration on all six learning problems, so Mackey-Glass and Lorenz
  are a genuine held-out test of whether the tuning generalises.
* **Staged, not full-factorial.**  Stage 1 tunes the shared optical geometry on
  the cheap static reservoir; stage 2 tunes the memristor-specific parameters at
  the winning geometry.  This assumes geometry and memristor parameters are
  roughly separable — cheaper than a full grid, and the assumption is stated
  rather than hidden.

Stages (all resumable — a cell with a ``metrics.json`` is skipped):

1. optical geometry for ``photonic_reservoir``: modes x photons x encoding scale
2. memristor parameters for ``photonic_memristor``: leak x number of memristors
3. final evaluation of the tuned configs (+ the matched capacity control) on all
   six tasks, 3 seeds, against the a-priori baseline
4. convergence-budget evaluation, capped at ``--conv-epochs``
5. retry only stage-4 cells that reached ``--previous-conv-cap``, writing them
   under a cap-specific namespace so the original stage-4 results are preserved

Outputs ``results/reservoir_search_summary.csv`` and ``results/reservoir_search.md``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RUN_ONE = PROJECT_DIR / "utils" / "run_experiment.py"
RESULT_DIR = PROJECT_DIR / "results" / "reservoir_search"
PY = sys.executable

# Tuning tasks (Hénon only) and the full evaluation set.
TUNE_TASKS = [("henon_1000", 4, 1, "henon_p1"), ("henon_1000", 4, 4, "henon_p4")]
EVAL_TASKS = TUNE_TASKS + [
    ("mackey_1000", 4, 1, "mackey_p1"),
    ("mackey_1000", 4, 140, "mackey_p140"),
    ("lorenz_1000", 4, 1, "lorenz_p1"),
    ("lorenz_1000", 4, 25, "lorenz_p25"),
]

# Stage 1 grid: optical geometry. Combinations are filtered so the Fock readout
# has a usable dimension -- C(modes, photons) >= 4 -- which drops e.g. (4 modes,
# 4 photons) where the UNBUNCHED space collapses to a single outcome.
MODES_PHOTONS = [(4, 2), (4, 3), (6, 2), (6, 3), (6, 4), (8, 2), (8, 3), (8, 4)]
SCALES = [1.0, math.pi / 2, math.pi, 2 * math.pi]

# Stage 1b: the coarse grid above turned out to be *monotone* in scale, with the
# winner pinned at its 2*pi edge -- an edge optimum is not an optimum, so scale is
# refined along a 1-D sweep at the winning geometry (coordinate descent) until the
# curve turns over.
SCALES_REFINE = [
    2 * math.pi,
    3 * math.pi,
    4 * math.pi,
    6 * math.pi,
    8 * math.pi,
    12 * math.pi,
    16 * math.pi,
    24 * math.pi,
    32 * math.pi,
    48 * math.pi,
]

# Stage 2 grid: memristor dynamics.
LEAKS = [0.0, 0.3, 0.5, 0.7, 0.9]
N_MEMRISTORS = [1, 2, 3]

APRIORI = {"modes": 6, "photons": 3, "scale": math.pi, "leak": 0.5, "mem": 2}


def ansatz_for(scale=None, leak=None, mem=None) -> str:
    s = "reservoir"
    if scale is not None:
        s += f"_scale{scale:.4f}"
    if leak is not None:
        s += f"_leak{leak:.2f}"
    if mem is not None:
        s += f"_mem{mem}"
    return s


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
        "--num-qubits",
        str(spec["modes"]),
        "--hidden-size",
        str(spec["photons"]),
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
        str(spec.get("use_conv", 0)),
        "--bugfix",
        "1",
        "--out",
        str(out),
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    ok = proc.returncode == 0 and (out / "metrics.json").exists()
    spec.update(ok=ok, wall_s=round(time.time() - t, 1))
    if not ok:
        spec["error"] = proc.stderr[-500:]
    return spec


def launch(specs: list[dict], workers: int, label: str) -> None:
    if not specs:
        return
    print(f"\n=== {label}: {len(specs)} runs on {workers} workers ===", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_cell, s): s for s in specs}
        for fut in as_completed(futs):
            s = fut.result()
            done += 1
            if not s["ok"]:
                print(f"  FAIL {s['runid']}: {s.get('error', '')[-200:]}", flush=True)
            elif done % 10 == 0 or done == len(specs):
                print(f"  [{done}/{len(specs)}]", flush=True)


def collect() -> pd.DataFrame:
    rows = []
    for mj in RESULT_DIR.rglob("metrics.json"):
        d = json.loads(mj.read_text())
        parts = mj.parent.name.split("__")
        d["stage"], d["cfg"], d["tag"] = parts[0], parts[1], parts[2]
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_DIR / "results" / "reservoir_search_summary.csv", index=False)
    return df


def best_by_validation(df: pd.DataFrame, stage: str) -> tuple[str, pd.DataFrame]:
    """Rank configs by mean rank of median *validation* MSE across tuning tasks."""
    sub = df[df.stage == stage]
    piv = sub.pivot_table(
        index="cfg", columns="tag", values="mse_val", aggfunc="median"
    )
    ranked = piv.rank().mean(axis=1).sort_values()
    return ranked.index[0], piv.assign(mean_val_rank=ranked).sort_values(
        "mean_val_rank"
    )


def stage1_specs(seeds, epochs) -> list[dict]:
    out = []
    for (m, p), sc in itertools.product(MODES_PHOTONS, SCALES):
        cfg = f"m{m}p{p}s{sc:.4f}"
        for ds, seq, pred, tag in TUNE_TASKS:
            for sd in seeds:
                out.append(
                    {
                        "model": "photonic_reservoir",
                        "ansatz": ansatz_for(scale=sc),
                        "modes": m,
                        "photons": p,
                        "dataset": ds,
                        "seq": seq,
                        "pred": pred,
                        "tag": tag,
                        "seed": sd,
                        "epochs": epochs,
                        "runid": f"s1__{cfg}__{tag}__seed{sd}",
                    }
                )
    return out


def stage1b_specs(modes, photons, seeds, epochs) -> list[dict]:
    """1-D refinement of the encoding scale at the winning geometry."""
    out = []
    for sc in SCALES_REFINE:
        cfg = f"m{modes}p{photons}s{sc:.4f}"
        for ds, seq, pred, tag in TUNE_TASKS:
            for sd in seeds:
                out.append(
                    {
                        "model": "photonic_reservoir",
                        "ansatz": ansatz_for(scale=sc),
                        "modes": modes,
                        "photons": photons,
                        "dataset": ds,
                        "seq": seq,
                        "pred": pred,
                        "tag": tag,
                        "seed": sd,
                        "epochs": epochs,
                        "runid": f"s1b__{cfg}__{tag}__seed{sd}",
                    }
                )
    return out


def stage2_specs(modes, photons, scale, seeds, epochs) -> list[dict]:
    out = []
    for leak, mem in itertools.product(LEAKS, N_MEMRISTORS):
        cfg = f"leak{leak:.2f}mem{mem}"
        for ds, seq, pred, tag in TUNE_TASKS:
            for sd in seeds:
                out.append(
                    {
                        "model": "photonic_memristor",
                        "ansatz": ansatz_for(scale=scale, leak=leak, mem=mem),
                        "modes": modes,
                        "photons": photons,
                        "dataset": ds,
                        "seq": seq,
                        "pred": pred,
                        "tag": tag,
                        "seed": sd,
                        "epochs": epochs,
                        "runid": f"s2__{cfg}__{tag}__seed{sd}",
                    }
                )
    return out


def stage3_specs(tuned, seeds, epochs, arm="fixed") -> list[dict]:
    """Evaluate tuned configs on all six tasks.

    ``arm='fixed'`` uses a flat epoch budget (comparable to the a-priori numbers
    in ``results/reservoir_table.md``); ``arm='conv'`` uses the paper's
    validation-plateau rule.  The convergence arm is the one that supports fair
    claims against the classical baselines -- at a flat budget those baselines are
    under-trained, which is precisely the artifact documented in README §4.
    """
    out = []
    variants = [
        (
            "tunedRC",
            "photonic_reservoir",
            ansatz_for(scale=tuned["scale"]),
            tuned["modes"],
            tuned["photons"],
        ),
        (
            "tunedMemRC",
            "photonic_memristor",
            ansatz_for(scale=tuned["scale"], leak=tuned["leak"], mem=tuned["mem"]),
            tuned["modes"],
            tuned["photons"],
        ),
        (
            "tunedSeqRC",
            "photonic_seqreservoir",
            ansatz_for(scale=tuned["scale"]),
            tuned["modes"],
            tuned["photons"],
        ),
    ]
    for cfg, model, ansatz, m, p in variants:
        for ds, seq, pred, tag in EVAL_TASKS:
            for sd in seeds:
                out.append(
                    {
                        "model": model,
                        "cfg": cfg,
                        "ansatz": ansatz,
                        "modes": m,
                        "photons": p,
                        "dataset": ds,
                        "seq": seq,
                        "pred": pred,
                        "tag": tag,
                        "seed": sd,
                        "epochs": epochs,
                        "use_conv": 1 if arm == "conv" else 0,
                        "runid": (
                            f"{'s4' if arm == 'conv' else 's3'}__{cfg}__{tag}__seed{sd}"
                        ),
                    }
                )
    return out


def _extended_convergence_specs(
    tuned: dict,
    seeds: list[int],
    extended_cap: int,
    previous_results: pd.DataFrame,
    previous_cap: int,
) -> list[dict]:
    """Build retries for stage-4 cells that reached the previous epoch cap.

    Parameters
    ----------
    tuned : dict
        Selected reservoir hyperparameters.
    seeds : list[int]
        Evaluation seeds whose stage-4 cells must be inspected.
    extended_cap : int
        New maximum number of convergence-training epochs.
    previous_results : pandas.DataFrame
        Collected search results containing a complete stage-4 evaluation.
    previous_cap : int
        Epoch cap used for the stage-4 evaluation.

    Returns
    -------
    list[dict]
        Specifications for only the cells that reached the previous cap.

    Raises
    ------
    ValueError
        If the new cap is not larger or the requested stage-4 matrix is incomplete.
    """
    if extended_cap <= previous_cap:
        raise ValueError(
            f"Extended cap {extended_cap} must be greater than previous cap {previous_cap}."
        )
    required_columns = {"stage", "cfg", "tag", "seed", "epochs"}
    missing_columns = required_columns - set(previous_results.columns)
    if missing_columns:
        raise ValueError(
            "Stage 5 requires collected stage-4 results with columns: "
            + ", ".join(sorted(required_columns))
        )

    stage4_specs = stage3_specs(tuned, seeds, previous_cap, arm="conv")
    expected_keys = {(spec["cfg"], spec["tag"], spec["seed"]) for spec in stage4_specs}
    stage4_results = previous_results[previous_results["stage"] == "s4"].copy()
    available_keys = {
        (row.cfg, row.tag, int(row.seed)) for row in stage4_results.itertuples()
    }
    missing_keys = expected_keys - available_keys
    if missing_keys:
        missing_preview = ", ".join(
            f"{cfg}/{tag}/seed{seed}" for cfg, tag, seed in sorted(missing_keys)[:5]
        )
        raise ValueError(
            "Stage 5 requires a complete stage-4 matrix. "
            f"Missing {len(missing_keys)} cells: {missing_preview}"
        )

    capped_keys = {
        (row.cfg, row.tag, int(row.seed))
        for row in stage4_results.itertuples()
        if int(row.epochs) >= previous_cap
    }
    extended_specs = []
    for spec in stage4_specs:
        key = (spec["cfg"], spec["tag"], spec["seed"])
        if key not in capped_keys:
            continue
        extended_spec = dict(spec)
        extended_spec["epochs"] = extended_cap
        extended_spec["runid"] = spec["runid"].replace(
            "s4__", f"s4_cap{extended_cap}__", 1
        )
        extended_specs.append(extended_spec)
    return extended_specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tune-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument(
        "--conv-epochs",
        type=int,
        default=3000,
        help="Cap for the stage-4 or stage-5 convergence arm.",
    )
    ap.add_argument(
        "--previous-conv-cap",
        type=int,
        default=3000,
        help="Stage-4 cap whose capped cells stage 5 retries.",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tuned = dict(APRIORI)

    if 1 in args.stage:
        launch(
            stage1_specs(args.tune_seeds, args.epochs),
            args.workers,
            "Stage 1 — optical geometry (static reservoir)",
        )
    df = collect()
    if not df.empty and (df.stage == "s1").any():
        cfg, table = best_by_validation(df, "s1")
        m, rest = cfg[1:].split("p", 1)
        p, sc = rest.split("s", 1)
        tuned.update(modes=int(m), photons=int(p), scale=float(sc))
        print(
            f"\nStage 1 winner (by validation): {cfg} -> "
            f"{tuned['modes']} modes, {tuned['photons']} photons, scale={tuned['scale']:.4f}"
        )

    if 12 in args.stage:  # stage "1b"
        launch(
            stage1b_specs(
                tuned["modes"], tuned["photons"], args.tune_seeds, args.epochs
            ),
            args.workers,
            "Stage 1b — refine encoding scale at the winning geometry",
        )
    df = collect()
    if not df.empty and (df.stage == "s1b").any():
        cfg, _ = best_by_validation(df, "s1b")
        tuned["scale"] = float(cfg.split("s")[-1])
        print(
            f"Stage 1b winner (by validation): scale={tuned['scale']:.4f} "
            f"({tuned['scale'] / math.pi:.3g} pi)"
        )

    if 2 in args.stage:
        launch(
            stage2_specs(
                tuned["modes"],
                tuned["photons"],
                tuned["scale"],
                args.tune_seeds,
                args.epochs,
            ),
            args.workers,
            "Stage 2 — memristor dynamics",
        )
    df = collect()
    if not df.empty and (df.stage == "s2").any():
        cfg, table = best_by_validation(df, "s2")
        leak, mem = cfg.replace("leak", "").split("mem")
        tuned.update(leak=float(leak), mem=int(mem))
        print(
            f"Stage 2 winner (by validation): {cfg} -> leak={tuned['leak']}, mem={tuned['mem']}"
        )

    if 3 in args.stage:
        launch(
            stage3_specs(tuned, args.eval_seeds, args.epochs),
            args.workers,
            "Stage 3 — evaluate tuned configs on all 6 tasks (fixed budget)",
        )

    if 4 in args.stage:
        launch(
            stage3_specs(tuned, args.eval_seeds, args.conv_epochs, arm="conv"),
            args.workers,
            "Stage 4 — evaluate tuned configs at the CONVERGENCE budget",
        )

    if 5 in args.stage:
        df = collect()
        extended_specs = _extended_convergence_specs(
            tuned,
            args.eval_seeds,
            args.conv_epochs,
            df,
            args.previous_conv_cap,
        )
        launch(
            extended_specs,
            args.workers,
            f"Stage 5 — retry {len(extended_specs)} capped convergence cells "
            f"at {args.conv_epochs} epochs",
        )

    df = collect()
    (PROJECT_DIR / "results" / "reservoir_search_best.json").write_text(
        json.dumps(tuned, indent=2)
    )
    print(f"\nTuned configuration: {tuned}")
    print(f"Collected {len(df)} runs -> results/reservoir_search_summary.csv")


if __name__ == "__main__":
    main()
