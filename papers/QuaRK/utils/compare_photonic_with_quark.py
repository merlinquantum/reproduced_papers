"""Photonic adaptation against QuaRK and the classical baselines, from the sweeps' artifacts.

Usage (from the repository root)::

    python papers/QuaRK/utils/compare_photonic_with_quark.py \\
        --photonic papers/QuaRK/outdir/sweep_photonic_adaptation \\
        --main papers/QuaRK/outdir/sweep_main_comparison \\
        --shot-noise papers/QuaRK/outdir/sweep_shot_noise_ablation

Writes ``results/photonic_comparison.json`` (full precision) and
``results/photonic_comparison.md``.

Rules (frozen in the Phase 4 adaptation plan)
---------------------------------------------
* Metric: test MSE at the ridge selected on validation data; normalized test MSE
  divides it by the variance of the test labels.
* Paired comparison per task over the 5 shared seeds, with the 2 SE rule and the sign
  count of ``report_statistics.paired``.
* Measurement penalty: each seed's test MSE divided by the same seed's test MSE with
  exact photonic features (the rule of ``report_statistics.shot_budget``).
* Setup table: read from each run's ``config_snapshot.json``, ``metrics.json`` and
  ``run.log``; the detection fields are fixed by ``lib/quark_merlin.py``.
* Tolerance, on each of the three paper tasks: (i) training MSE at the smallest ridge
  (10^-12) <= 10^-10 in 5/5 seeds; (ii) test MSE at N = 8000 below the one at N = 100
  in 5/5 seeds; (iii) mean normalized test MSE within a factor 2 of its QuaRK
  counterpart (photonic exact <-> QuaRK exact; photonic 1000 shots <-> random-Pauli
  shadows at 1000 shots).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from report_statistics import (
    PAPER_TASKS,
    RESULTS_DIR,
    TASK_LABELS,
    TASKS,
    _jsonable,
    cell,
    describe,
    fmt,
    load_sweep,
    paired,
)

INTERPOLATION_MAX = 1e-10
FACTOR = 2.0
PHOTONIC = {
    "quark_merlin_shots_100": "Photonic, 100 shots",
    "quark_merlin": "Photonic, 1000 shots",
    "quark_merlin_shots_10000": "Photonic, 10000 shots",
    "quark_exact_merlin": "Photonic, exact",
    "quark_distinguishable_merlin": "Photonic, exact, distinguishable photons",
}
QUARK = {
    "quark_shadows_shots_100": "QuaRK, shadows, 100 shots",
    "quark_shadows": "QuaRK, shadows, 1000 shots",
    "quark_shadows_shots_10000": "QuaRK, shadows, 10000 shots",
    "quark_shots_100": "QuaRK, per-observable noise, 100 shots",
    "quark": "QuaRK, per-observable noise, 1000 shots (paper)",
    "quark_shots_10000": "QuaRK, per-observable noise, 10000 shots",
    "quark_exact": "QuaRK, exact",
}
CLASSICAL = {
    "raw_window_kernel": "Matérn kernel on the raw window",
    "raw_window_linear": "Linear ridge on the raw window",
    "esn_default": "Echo state network, default",
    "esn_tuned": "Echo state network, tuned",
}
# Tolerance counterparts (criterion iii).
TOLERANCE_PAIRS = [
    ("quark_exact_merlin", "quark_exact"),
    ("quark_merlin", "quark_shadows"),
]
# Every declared paired comparison: (photonic candidate, other candidate).
COMPARISONS = [
    *TOLERANCE_PAIRS,
    ("quark_merlin_shots_100", "quark_shadows_shots_100"),
    ("quark_merlin_shots_10000", "quark_shadows_shots_10000"),
    ("quark_merlin_shots_100", "quark_shots_100"),
    ("quark_merlin", "quark"),
    ("quark_merlin_shots_10000", "quark_shots_10000"),
    ("quark_exact_merlin", "quark_distinguishable_merlin"),
    *[(p, c) for p in PHOTONIC for c in CLASSICAL],
]
LABELS = {**PHOTONIC, **QUARK, **CLASSICAL}
SHOT_CANDIDATES = ("quark_merlin_shots_100", "quark_merlin", "quark_merlin_shots_10000")
# Fixed by lib/quark_merlin.py (truncated_layer), the same for every configuration.
DETECTION = {
    "computation_space": "FOCK, occupancy_readout=True",
    "detector": "ideal threshold detectors (click / no click), no dark counts, no loss",
    "postselection": "none",
    "encoding": "phases pi (1 - 1e-6) tanh(Pi x_t) on all modes at each step, "
    "between copies of one fixed random MZI mesh",
    "measurement": "click patterns, reduced to 1- and 2-mode click marginals",
}


def per_seed(runs: dict, candidate: str, task: str) -> dict:
    """``{seed: task metrics}`` for one candidate."""
    return {
        seed: run["metrics"]["tasks"][task]
        for (name, seed), run in runs.items()
        if name == candidate
    }


def behaviour(runs: dict, candidate: str) -> list:
    """Criteria (i) and (ii) for every task."""
    rows = []
    for task in TASKS:
        seeds = per_seed(runs, candidate, task)
        train = {s: t["mse_train_at_smallest_reg"] for s, t in seeds.items()}
        decays = {}
        for s, t in seeds.items():
            sel = t["selections"]["validation"]
            sizes, mse = sel["training_sizes"], sel["training_size_mse_test"]
            if sizes[0] != 100 or sizes[-1] != 8000:
                raise SystemExit(f"{candidate} seed {s}: unexpected sizes {sizes}")
            decays[s] = mse[-1] < mse[0]
        rows.append(
            {
                "task": task,
                "paper_task": task in PAPER_TASKS,
                "seeds": len(seeds),
                "max_train_mse_at_smallest_reg": max(train.values()),
                "interpolating_seeds": sum(
                    v <= INTERPOLATION_MAX for v in train.values()
                ),
                "decaying_seeds": sum(decays.values()),
            }
        )
    return rows


def versions(run_log: Path) -> dict:
    """Package versions printed by the shared runtime at the top of ``run.log``."""
    out = {}
    for line in run_log.read_text().splitlines()[:20]:
        for package in ("Python", "Perceval", "MerLin"):
            if f"| {package} version " in line:
                out[package] = line.rsplit(" ", 1)[-1]
    return out


def setup(runs: dict, run_dirs: dict) -> list:
    """Hardware-aware fields of every photonic configuration, from its runs."""
    rows = []
    for cand in PHOTONIC:
        keys = sorted(k for k in runs if k[0] == cand)
        merlin, info, logged = set(), set(), set()
        for key in keys:
            cfg = json.loads((run_dirs[key] / "config_snapshot.json").read_text())
            merlin.add(json.dumps(cfg["model"]["merlin"], sort_keys=True))
            model_info = runs[key]["metrics"]["tasks"][TASKS[0]]["model_info"]
            info.add(
                json.dumps(
                    {
                        k: model_info[k]
                        for k in (
                            "input_state",
                            "num_click_patterns",
                            "num_observables",
                        )
                    },
                    sort_keys=True,
                )
            )
            logged.add(json.dumps(versions(run_dirs[key] / "run.log"), sort_keys=True))
            device = (cfg["device"], cfg["dtype"])
        if len(merlin) != 1 or len(info) != 1 or len(logged) != 1:
            raise SystemExit(f"{cand}: settings differ between seeds")
        seconds = [runs[k]["metrics"]["seconds"] for k in keys]
        rows.append(
            {
                "candidate": cand,
                "merlin": json.loads(merlin.pop()),
                **json.loads(info.pop()),
                "versions": json.loads(logged.pop()),
                "device": device,
                "seeds": [k[1] for k in keys],
                "seconds": {
                    "min": min(seconds),
                    "mean": float(np.mean(seconds)),
                    "max": max(seconds),
                },
            }
        )
    return rows


def penalty(runs: dict) -> list:
    """Photonic test MSE over the same seed's exact photonic test MSE."""
    rows = []
    for task in TASKS:
        exact = cell(runs, "quark_exact_merlin", task, "mse_test")
        for cand in SHOT_CANDIDATES:
            values = cell(runs, cand, task, "mse_test")
            rows.append(
                {
                    "candidate": cand,
                    "task": task,
                    "ratio_to_exact": describe(
                        values[s] / exact[s] for s in sorted(exact)
                    ),
                }
            )
    return rows


def compare(runs: dict, a: str, b: str) -> list:
    rows = []
    for task in TASKS:
        mse_a = cell(runs, a, task, "mse_test")
        mse_b = cell(runs, b, task, "mse_test")
        norm_a = describe(cell(runs, a, task, "normalized_mse_test").values())
        norm_b = describe(cell(runs, b, task, "normalized_mse_test").values())
        rows.append(
            {
                "task": task,
                "a": a,
                "b": b,
                "normalized_a": norm_a,
                "normalized_b": norm_b,
                "ratio_of_means": norm_a["mean"] / norm_b["mean"],
                "paired": paired(mse_a, mse_b, a, b),
            }
        )
    return rows


def verdict(behaviours: dict, comparisons: list) -> list:
    out = []
    for a, b in TOLERANCE_PAIRS:
        for task in PAPER_TASKS:
            beh = next(r for r in behaviours[a] if r["task"] == task)
            comp = next(
                r
                for r in comparisons
                if r["a"] == a and r["b"] == b and r["task"] == task
            )
            ratio = comp["ratio_of_means"]
            checks = {
                "interpolation": beh["interpolating_seeds"] == beh["seeds"] == 5,
                "decay": beh["decaying_seeds"] == beh["seeds"] == 5,
                "factor_2": 1 / FACTOR <= ratio <= FACTOR,
            }
            out.append(
                {
                    "a": a,
                    "b": b,
                    "task": task,
                    "ratio": ratio,
                    **checks,
                    "pass": all(checks.values()),
                }
            )
    return out


def to_markdown(report: dict) -> str:
    lines = [
        "# Photonic adaptation: comparison with QuaRK",
        "",
        "Simulated photonic reservoir (MerLin, 3 photons, 14 modes, threshold "
        "detection), same data, seeds and readout as QuaRK. Metric: test MSE at the "
        "validation-selected ridge; normalized = MSE / variance of the test labels; "
        "mean ± std over 5 seeds.",
        "",
        f"**Tolerance (frozen plan): {'PASS' if report['tolerance_pass'] else 'FAIL'}**",
        "",
        "| Photonic | QuaRK | Task | Ratio of normalized MSE | Interpolation | Decay | "
        "Within ×2 | Pass |",
        "|---|---|---|---:|:-:|:-:|:-:|:-:|",
    ]
    for r in report["tolerance"]:
        lines.append(
            f"| {LABELS[r['a']]} | {LABELS[r['b']]} | {TASK_LABELS[r['task']]} "
            f"| {r['ratio']:.3f} | {'yes' if r['interpolation'] else 'no'} "
            f"| {'yes' if r['decay'] else 'no'} | {'yes' if r['factor_2'] else 'no'} "
            f"| {'yes' if r['pass'] else 'no'} |"
        )
    lines += [
        "",
        "## Setup of every photonic configuration",
        "",
        "Simulated, not hardware-executed; no photonic parameter is trained, so no "
        "gradient is used. Common to all: "
        + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in DETECTION.items())
        + ".",
        "",
        "| Configuration | Photons | Modes | Sub-reservoirs | Input state "
        "| Click patterns | Marginals | Shots | Simulator | Seeds | Wall-clock per run "
        "(min / mean / max) |",
        "|---|---:|---:|---:|---|---:|---:|---|---|---|---|",
    ]
    for r in report["setup"]:
        m, v, sec = r["merlin"], r["versions"], r["seconds"]
        shots = m["shots"] if m["estimator"] == "click_sampling" else "exact"
        photons = m["num_photons"]
        if m["distinguishable"]:
            photons = f"{photons}, distinguishable"
        lines.append(
            f"| {LABELS[r['candidate']]} | {photons} | {m['num_modes']} "
            f"| {m['num_reservoirs']} | `{r['input_state']}` "
            f"| {r['num_click_patterns']} | {r['num_observables']} per sub-reservoir "
            f"| {shots} | MerLin {v['MerLin']}, Perceval {v['Perceval']}, "
            f"{r['device'][0]}, {r['device'][1]} "
            f"| {', '.join(str(s) for s in r['seeds'])} "
            f"| {sec['min']:.0f} / {sec['mean']:.0f} / {sec['max']:.0f} s |"
        )
    lines += [
        "",
        "The version string is the one MerLin prints; the installed distribution is "
        "merlinquantum 0.4.2 (`requirements.txt`).",
        "",
        "## Measurement penalty",
        "",
        "Each seed's test MSE divided by the same seed's test MSE with exact photonic "
        "features; mean ± std over 5 seeds.",
        "",
        "| Configuration | Task | Ratio to exact |",
        "|---|---|---:|",
    ]
    for r in report["penalty"]:
        d = r["ratio_to_exact"]
        lines.append(
            f"| {LABELS[r['candidate']]} | {TASK_LABELS[r['task']]} "
            f"| {fmt(d['mean'])} ± {fmt(d['std'])} |"
        )
    lines += [
        "",
        "## Interpolation and decay, every photonic configuration",
        "",
        "Seeds (of 5) with training MSE ≤ 1e-10 at ridge 1e-12, and with a lower test "
        "MSE at N = 8000 than at N = 100.",
        "",
        "| Configuration | Task | Interpolating | Decaying | Max training MSE |",
        "|---|---|---:|---:|---:|",
    ]
    for cand, rows in report["behaviour"].items():
        for r in rows:
            lines.append(
                f"| {LABELS[cand]} | {TASK_LABELS[r['task']]} "
                f"| {r['interpolating_seeds']} | {r['decaying_seeds']} "
                f"| {fmt(r['max_train_mse_at_smallest_reg'])} |"
            )
    lines += [
        "",
        "## Paired comparisons",
        "",
        'Delta = MSE(photonic) − MSE(other), 2 SE rule; "lower" names the model with '
        "the lower test MSE; seeds = seeds where the photonic model is lower.",
        "",
        "| Photonic | Other | Task | Normalized (photonic) | Normalized (other) "
        "| Mean delta | SE | Seeds | Label |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in report["comparisons"]:
        p = r["paired"]
        label = p["label"].replace(r["a"], "photonic").replace(r["b"], "other")
        if p["flag_interval_contains_zero"]:
            label += " (interval contains 0)"
        lines.append(
            f"| {LABELS[r['a']]} | {LABELS[r['b']]} | {TASK_LABELS[r['task']]} "
            f"| {fmt(r['normalized_a']['mean'])} ± {fmt(r['normalized_a']['std'])} "
            f"| {fmt(r['normalized_b']['mean'])} ± {fmt(r['normalized_b']['std'])} "
            f"| {fmt(p['mean_delta'])} | {fmt(p['se_delta'])} "
            f"| {p['a_lower_seeds']}/5 | {label} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photonic", type=Path, required=True)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--shot-noise", type=Path, required=True)
    args = parser.parse_args()

    sweeps, runs = [], {}
    for path in (args.photonic, args.main, args.shot_noise):
        sweep_id, sweep_runs = load_sweep(path)
        sweeps.append(sweep_id)
        if set(sweep_runs) & set(runs):
            raise SystemExit(f"{sweep_id}: duplicate (candidate, seed)")
        runs.update(sweep_runs)
    status = json.loads((args.photonic / "sweep_status.json").read_text())
    run_dirs = {
        (run["candidate"]["name"], int(run["seed"])): (
            args.photonic / run["metrics_path"]
        ).parent
        for run in status["runs"]
    }
    seeds = {name: sorted(s for n, s in runs if n == name) for name in LABELS}
    if len({tuple(v) for v in seeds.values()}) != 1 or len(seeds["quark"]) != 5:
        raise SystemExit(f"seed sets differ: {seeds}")
    # Same data per seed: the dataset metadata of every model must agree.
    data_per_seed = {
        seed: {
            json.dumps(run["metrics"]["dataset_meta"], sort_keys=True)
            for (name, s), run in runs.items()
            if name in LABELS and s == seed
        }
        for seed in seeds["quark"]
    }
    if any(len(v) != 1 for v in data_per_seed.values()):
        raise SystemExit("dataset metadata differ between models at one seed")
    behaviours = {cand: behaviour(runs, cand) for cand in PHOTONIC}
    comparisons = [row for a, b in COMPARISONS for row in compare(runs, a, b)]
    tolerance = verdict(behaviours, comparisons)
    report = {
        "sweeps": sweeps,
        "seeds": seeds["quark"],
        "same_data_per_seed": True,
        "tolerance_pass": all(r["pass"] for r in tolerance),
        "tolerance": tolerance,
        "setup": setup(runs, run_dirs),
        "penalty": penalty(runs),
        "behaviour": behaviours,
        "comparisons": comparisons,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "photonic_comparison.json").write_text(
        json.dumps(_jsonable(report), indent=2)
    )
    (RESULTS_DIR / "photonic_comparison.md").write_text(to_markdown(report))
    print(RESULTS_DIR / "photonic_comparison.json")
    print(RESULTS_DIR / "photonic_comparison.md")
    print(f"tolerance: {'PASS' if report['tolerance_pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    np.seterr(all="raise")
    raise SystemExit(main())
