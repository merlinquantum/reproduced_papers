"""Fair comparison of QuaRK, its photonic adaptation and the classical baselines.

Usage (from the repository root)::

    python papers/QuaRK/utils/compare_fair_baselines.py \\
        --main papers/QuaRK/outdir/sweep_main_comparison \\
        --photonic papers/QuaRK/outdir/sweep_photonic_adaptation \\
        --shot-noise papers/QuaRK/outdir/sweep_shot_noise_ablation \\
        --scaling papers/QuaRK/outdir/sweep_scaling \\
        --release-comparison papers/QuaRK/results/release_comparison.json

Writes ``results/fair_comparison.json`` (full precision) and
``results/fair_comparison.md``. Read-only on the run artifacts.

Rules (frozen in the "Phase 5 Comparison Plan" of the reproduction log)
---------------------------------------------------------------------
* Metric: test MSE at the ridge selected on validation data, minimized; normalized
  test MSE divides it by the variance of the test labels.
* Aggregation: mean and sample standard deviation over the 5 seeds.
* Paired comparison per task over the shared seeds, with the 2 SE rule and the sign
  count of ``report_statistics.paired``.
* Commensurable: same seeds, the same dataset metadata at each seed, and identical
  ``data`` and ``readout`` settings in every run's ``config_snapshot.json``.
* Wall-clock per run: ``metrics.json`` ``seconds``. Feature construction: from the
  ``run.log`` timestamps, from ``TRAINING_STARTED`` (or the previous task's end) to
  each ``FEATURES_READY``, at 1 s resolution.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
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

# (candidate, label, role, scope), in display order.
SYSTEMS = (
    (
        "quark",
        "QuaRK, paper settings",
        "method, reproduced",
        "paper-accurate settings, 5 seeds",
    ),
    (
        "quark_exact",
        "QuaRK, exact expectations",
        "method variant: no shot noise",
        "paper-accurate settings, 5 seeds",
    ),
    (
        "quark_shadows",
        "QuaRK, random-Pauli shadows",
        "method variant: the paper text's estimator",
        "paper-accurate settings, 5 seeds",
    ),
    (
        "quark_merlin",
        "Photonic, 1000 shots",
        "MerLin adaptation",
        "adaptation, simulated, 5 seeds",
    ),
    (
        "quark_exact_merlin",
        "Photonic, exact",
        "MerLin adaptation, no shot noise",
        "adaptation, simulated, 5 seeds",
    ),
    (
        "quark_distinguishable_merlin",
        "Photonic, exact, distinguishable photons",
        "control: no interference",
        "adaptation, simulated, 5 seeds",
    ),
    (
        "raw_window_kernel",
        "Matérn kernel on the raw window",
        "fair baseline (ours)",
        "5 seeds",
    ),
    (
        "raw_window_linear",
        "Linear ridge on the raw window",
        "fair baseline (ours)",
        "5 seeds",
    ),
    (
        "esn_default",
        "Echo state network, default",
        "fair baseline (ours)",
        "5 seeds",
    ),
    (
        "esn_tuned",
        "Echo state network, tuned",
        "fair baseline (ours)",
        "5 seeds",
    ),
)
LABELS = {s[0]: s[1] for s in SYSTEMS}
BASELINES = ("raw_window_kernel", "raw_window_linear", "esn_default", "esn_tuned")
# Paired comparisons of the plan: (A, B, origin).
PAIRS = (
    *[
        (q, b, "frozen in the Fair Baseline Plan")
        for q in ("quark",)
        for b in BASELINES
    ],
    *[
        (q, b, "added after the main sweep (phase 03)")
        for q in ("quark_exact",)
        for b in BASELINES
    ],
    *[
        (p, b, "Phase 4 plan")
        for p in ("quark_merlin", "quark_exact_merlin", "quark_distinguishable_merlin")
        for b in BASELINES
    ],
    ("quark_exact_merlin", "quark_exact", "Phase 4 plan"),
    ("quark_merlin", "quark_shadows", "Phase 4 plan"),
    ("quark_merlin", "quark", "Phase 4 plan"),
)
EXTENSIONS = (
    (
        "shot_noise",
        "QuaRK at 100 and 10000 shots, both estimators",
        "characterization",
        "results/statistics.md (shot budget)",
    ),
    (
        "scaling",
        "QuaRK at n = 3, 7 (R = 3) and R = 1, 6 (n = 5)",
        "characterization",
        "results/statistics.md (n and R)",
    ),
    (
        "photonic_shots",
        "Photonic at 100 and 10000 shots",
        "characterization; compared only with QuaRK at the same budget",
        "results/photonic_comparison.md",
    ),
)
ESTIMATOR_LABELS = {
    "per_observable_shots": "per observable (authors' model)",
    "random_pauli_shadows": "snapshots, shared (random-Pauli shadows)",
}
COMMON_SETTINGS = ("data.", "readout.")
EVENTS = ("TRAINING_STARTED", "TRAIN_TASK_COMPLETED", "FEATURES_READY")


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: json.dumps(value, sort_keys=True)}


def run_dirs(sweep_dir: Path) -> dict:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    return {
        (run["candidate"]["name"], int(run["seed"])): (
            sweep_dir / run["metrics_path"]
        ).parent
        for run in status["runs"]
    }


def expected_runs(sweep_dir: Path) -> dict:
    """``{candidate: planned run count}`` from the sweep's frozen spec."""
    spec = json.loads((sweep_dir / "sweep_spec.json").read_text())
    return {c["name"]: len(spec["seeds"]) for c in spec["candidates"]}


def feature_seconds(run_log: Path) -> float:
    """Seconds spent building features, from the run log's event timestamps."""
    previous, total = None, 0.0
    for line in run_log.read_text().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4 or parts[3] not in EVENTS:
            continue
        stamp = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
        if parts[3] == "FEATURES_READY":
            total += (stamp - previous).total_seconds()
        previous = stamp
    return total


def spread(values) -> dict:
    """Minimum, mean and maximum of a list of timings."""
    values = [float(v) for v in values]
    return {"min": min(values), "mean": float(np.mean(values)), "max": max(values)}


def resources(run_dir: Path, metrics: dict) -> dict:
    cfg = json.loads((run_dir / "config_snapshot.json").read_text())
    model = cfg["model"]
    kind = model["kind"]
    if kind == "quark":
        q = model["quark"]
        shots = (
            "exact"
            if q["estimator"] == "exact"
            else f"{q['shots']} {ESTIMATOR_LABELS[q['estimator']]}"
        )
    elif kind == "quark_merlin":
        m = model["merlin"]
        shots = (
            "exact"
            if m["estimator"] == "exact"
            else f"{m['shots']} click patterns, shared"
        )
    else:
        shots = "none (classical)"
    searched = 0
    if kind == "esn_kernel" and model["esn"]["mode"] == "tuned":
        esn = model["esn"]
        searched = (
            len(esn["spectral_radius"]) * len(esn["leak"]) * len(esn["input_scale"])
        )
    return {
        "features": metrics["tasks"][TASKS[0]]["feature_dim"],
        "measurement": shots,
        "readout": "linear ridge"
        if kind == "raw_window_linear"
        else "Matérn kernel ridge",
        "reservoir_settings_searched": searched,
    }


def system_rows(runs: dict, dirs: dict, expected: dict) -> list:
    rows = []
    for candidate, label, role, scope in SYSTEMS:
        keys = sorted(k for k in runs if k[0] == candidate)
        seconds = [runs[k]["metrics"]["seconds"] for k in keys]
        features = [feature_seconds(dirs[k] / "run.log") for k in keys]
        info = {
            json.dumps(resources(dirs[k], runs[k]["metrics"]), sort_keys=True)
            for k in keys
        }
        if len(info) != 1:
            raise SystemExit(f"{candidate}: resources differ between seeds")
        rows.append(
            {
                "candidate": candidate,
                "label": label,
                "role": role,
                "scope": scope,
                "provenance": "local run",
                "runs": len(keys),
                "expected_runs": expected[candidate],
                "normalized": {
                    t: describe(
                        cell(runs, candidate, t, "normalized_mse_test").values()
                    )
                    for t in TASKS
                },
                "mse": {
                    t: describe(cell(runs, candidate, t, "mse_test").values())
                    for t in TASKS
                },
                "resources": json.loads(info.pop()),
                "seconds": spread(seconds),
                "feature_seconds": spread(features),
            }
        )
    return rows


def commensurability(runs: dict, dirs: dict) -> dict:
    """Check that every system ran on the same data and readout settings."""
    seeds = {c: sorted(s for n, s in runs if n == c) for c in LABELS}
    same_seeds = len({tuple(v) for v in seeds.values()}) == 1
    meta_ok, settings_ok, differing = True, True, set()
    for seed in seeds["quark"]:
        keys = [(c, seed) for c in LABELS]
        metas = {
            json.dumps(runs[k]["metrics"]["dataset_meta"], sort_keys=True) for k in keys
        }
        meta_ok &= len(metas) == 1
        flat = [
            {
                key: value
                for key, value in flatten(
                    json.loads((dirs[k] / "config_snapshot.json").read_text())
                ).items()
                if key.startswith(COMMON_SETTINGS)
            }
            for k in keys
        ]
        all_keys = set().union(*flat)
        bad = {key for key in all_keys if len({f.get(key) for f in flat}) > 1}
        differing |= bad
        settings_ok &= not bad
    finite = all(
        math.isfinite(runs[k]["metrics"]["tasks"][t]["selections"]["validation"][m])
        for k in runs
        if k[0] in LABELS
        for t in TASKS
        for m in ("mse_test", "normalized_mse_test")
    )
    status = (
        "COMMENSURABLE"
        if same_seeds and meta_ok and settings_ok and finite
        else "PARTIALLY_COMMENSURABLE"
    )
    return {
        "seeds": seeds["quark"],
        "same_seeds": same_seeds,
        "same_dataset_meta_per_seed": meta_ok,
        "same_data_and_readout_settings": settings_ok,
        "differing_settings": sorted(differing),
        "finite_metrics": finite,
        "status": status,
    }


def pair_rows(runs: dict, status: str) -> list:
    rows = []
    for a, b, origin in PAIRS:
        for task in TASKS:
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "origin": origin,
                    "task": task,
                    "status": status,
                    "paired": paired(
                        cell(runs, a, task, "mse_test"),
                        cell(runs, b, task, "mse_test"),
                        a,
                        b,
                    ),
                }
            )
    return rows


def claim_evidence(runs: dict) -> dict:
    """Per system: interpolation, decay over N and the task ranking, 5 seeds."""
    out = {}
    for candidate in LABELS:
        keys = sorted(k for k in runs if k[0] == candidate)
        tasks = [runs[k]["metrics"]["tasks"] for k in keys]
        raw_order = sum(
            t["current_point_readout"]["selections"]["validation"]["mse_test"]
            < t["exp_fading"]["selections"]["validation"]["mse_test"]
            < t["volterra"]["selections"]["validation"]["mse_test"]
            for t in tasks
        )
        normalized_order = sum(
            t["current_point_readout"]["selections"]["validation"][
                "normalized_mse_test"
            ]
            < t["exp_fading"]["selections"]["validation"]["normalized_mse_test"]
            < t["volterra"]["selections"]["validation"]["normalized_mse_test"]
            for t in tasks
        )
        out[candidate] = {
            "max_train_mse_at_smallest_ridge": max(
                t[task]["mse_train_at_smallest_reg"]
                for t in tasks
                for task in PAPER_TASKS
            ),
            "decay_slope": {
                task: describe(
                    t[task]["selections"]["validation"]["decay_slope"] for t in tasks
                )
                for task in PAPER_TASKS
            },
            "seeds_with_lower_mse_at_8000_than_100": {
                task: sum(
                    t[task]["selections"]["validation"]["training_size_mse_test"][-1]
                    < t[task]["selections"]["validation"]["training_size_mse_test"][0]
                    for t in tasks
                )
                for task in PAPER_TASKS
            },
            "paper_order_raw": raw_order,
            "paper_order_normalized": normalized_order,
            "seeds": len(keys),
        }
    return out


def extension_rows(args) -> list:
    rows = []
    for key, what, label, table in EXTENSIONS:
        if key == "photonic_shots":
            sweep, names = (
                args.photonic,
                ("quark_merlin_shots_100", "quark_merlin_shots_10000"),
            )
        else:
            sweep = args.shot_noise if key == "shot_noise" else args.scaling
            names = None
        _, runs = load_sweep(sweep)
        expected = expected_runs(sweep)
        names = names or tuple(expected)
        rows.append(
            {
                "extension": key,
                "what": what,
                "label": label,
                "table": table,
                "runs": sum(1 for k in runs if k[0] in names),
                "expected_runs": sum(expected[n] for n in names),
            }
        )
    return rows


def to_markdown(report: dict) -> str:
    c = report["commensurability"]
    lines = [
        "# Fair comparison: QuaRK, its photonic adaptation and classical baselines",
        "",
        "All values are local runs on the same windows, split and seeds "
        f"({', '.join(str(s) for s in c['seeds'])}); the paper reports no baseline. "
        "Metric: test MSE on 2,000 held-out windows at the ridge chosen on validation "
        "data; normalized = MSE / variance of the test labels; mean ± std over 5 seeds. "
        "Matching: the same Matérn kernel ridge readout and tuning for every kernel "
        "model; 315 features for QuaRK, its photonic version and the echo state "
        "networks; 75 raw inputs for the reservoir-free baselines.",
        "",
        f"**Coverage:** {report['coverage']}. **Commensurability:** {c['status']} "
        f"(same seeds: {c['same_seeds']}; same dataset per seed: "
        f"{c['same_dataset_meta_per_seed']}; identical data and readout settings: "
        f"{c['same_data_and_readout_settings']}; finite metrics: {c['finite_metrics']}).",
        "",
        "## Systems",
        "",
        "| System | Role | Scope | Runs | F1 | F2 | F3 | Forecast | Features "
        "| Measurement | Settings searched | Wall-clock per run |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in report["systems"]:
        norm = [
            f"{fmt(r['normalized'][t]['mean'])} ± {fmt(r['normalized'][t]['std'])}"
            for t in TASKS
        ]
        res = r["resources"]
        lines.append(
            f"| {r['label']} | {r['role']} | {r['scope']} | {r['runs']}/{r['expected_runs']} "
            f"| {' | '.join(norm)} | {res['features']} | {res['measurement']} "
            f"| {res['reservoir_settings_searched']} | {r['seconds']['mean']:.0f} s |"
        )
    lines += [
        "",
        "F1–F3: the paper's tasks (F1 as run: current-point readout); Forecast: F1 as "
        "the text defines it. Values: normalized test MSE.",
        "",
        "## Paper-reported values",
        "",
        "The paper's only numbers are QuaRK's, from one run (the authors' seed 12345, "
        "ridge chosen on the test set). Test MSE at N = 8000, from the authors' release "
        "and from our run at the same seed and rule. They are PARTIALLY_COMMENSURABLE "
        "with the tables above: one seed, a ridge chosen on the test set, raw MSE; the "
        "paper has no value for any other system (N/A).",
        "",
        "| Task | Paper (authors' release) | This reproduction |",
        "|---|---:|---:|",
    ]
    for task, v in report["paper_reported"].items():
        lines.append(
            f"| {TASK_LABELS[task]} | {fmt(v['release'], 4)} | {fmt(v['ours'], 4)} |"
        )
    lines += [
        "",
        "## Paired comparisons",
        "",
        "Delta = MSE(A) − MSE(B) over the 5 seeds; frozen 2 SE rule; seeds = seeds "
        "where A is lower; 95 % interval of the mean delta (descriptive).",
        "",
        "| A | B | Task | Mean delta | SE | Seeds | 95 % interval | Label | Origin |",
        "|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for r in report["pairs"]:
        p = r["paired"]
        label = p["label"].replace(r["a"], "A").replace(r["b"], "B")
        flag = " (interval contains 0)" if p["flag_interval_contains_zero"] else ""
        lines.append(
            f"| {LABELS[r['a']]} | {LABELS[r['b']]} | {TASK_LABELS[r['task']]} "
            f"| {fmt(p['mean_delta'])} | {fmt(p['se_delta'])} | {p['a_lower_seeds']}/5 "
            f"| [{fmt(p['ci95_delta'][0])}, {fmt(p['ci95_delta'][1])}] | {label}{flag} "
            f"| {r['origin']} |"
        )
    lines += [
        "",
        "## Evidence for the paper's claims, every system",
        "",
        "Interpolation: largest training MSE at ridge 10⁻¹² over the three paper tasks "
        "and 5 seeds. Decay: log-log slope of the test MSE over N = 500–8000 (−0.5 is "
        "1/√N). Lower at 8000: seeds with a lower test MSE at N = 8000 than at N = 100. "
        "Order: seeds where F1 < F2 < F3, on raw and on normalized test MSE. The linear "
        "ridge cannot interpolate (76 parameters for 8,000 windows).",
        "",
        "| System | Interpolation | Slope F1 | Slope F2 | Slope F3 | Lower at 8000 "
        "(F1/F2/F3) | Order, raw | Order, normalized |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for cand, e in report["claim_evidence"].items():
        slopes = [
            f"{fmt(e['decay_slope'][t]['mean'])} ± {fmt(e['decay_slope'][t]['std'])}"
            for t in PAPER_TASKS
        ]
        lower = "/".join(
            str(e["seeds_with_lower_mse_at_8000_than_100"][t]) for t in PAPER_TASKS
        )
        lines.append(
            f"| {LABELS[cand]} | {fmt(e['max_train_mse_at_smallest_ridge'])} "
            f"| {' | '.join(slopes)} | {lower} | {e['paper_order_raw']}/5 "
            f"| {e['paper_order_normalized']}/5 |"
        )
    lines += [
        "",
        "## Wall-clock",
        "",
        "One machine (Apple M5, 10 cores, local CPU, recorded in the reproduction log), "
        "one run at a time. Workload per run: 10,000 windows, features once, then for "
        "each of the 4 tasks the Matérn tuning (4 ν × up to 80 ξ evaluations), the "
        "26-ridge sweep and the 10-size training sweep. Quantum features are simulated; "
        "these times say nothing about hardware execution.",
        "",
        "| System | Per run (min / mean / max) | Feature construction (mean) | Share of the run |",
        "|---|---|---:|---:|",
    ]
    for r in report["systems"]:
        s, f = r["seconds"], r["feature_seconds"]
        share = f["mean"] / s["mean"]
        lines.append(
            f"| {r['label']} | {s['min']:.0f} / {s['mean']:.0f} / {s['max']:.0f} s "
            f"| {f['mean']:.0f} s | {share:.0%} |"
        )
    lines += [
        "",
        "## Extensions",
        "",
        "Conditions the paper makes no claim about. Only QuaRK or its photonic version "
        "ran under them, so they characterize those models and make no advantage "
        "statement over the classical baselines.",
        "",
        "| Extension | What ran | Runs | Label | Table |",
        "|---|---|---:|---|---|",
    ]
    for r in report["extensions"]:
        lines.append(
            f"| {r['extension'].replace('_', ' ')} | {r['what']} "
            f"| {r['runs']}/{r['expected_runs']} | {r['label']} | `{r['table']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--photonic", type=Path, required=True)
    parser.add_argument("--shot-noise", type=Path, required=True)
    parser.add_argument("--scaling", type=Path, required=True)
    parser.add_argument("--release-comparison", type=Path, required=True)
    args = parser.parse_args()

    sweeps, runs, dirs, expected = [], {}, {}, {}
    for path in (args.main, args.photonic):
        sweep_id, sweep_runs = load_sweep(path)
        if set(sweep_runs) & set(runs):
            raise SystemExit(f"{sweep_id}: duplicate (candidate, seed)")
        sweeps.append(sweep_id)
        runs.update(sweep_runs)
        dirs.update(run_dirs(path))
        expected.update(expected_runs(path))
    missing = [c for c in LABELS if sum(1 for k in runs if k[0] == c) != expected[c]]
    if missing:
        raise SystemExit(f"incomplete systems: {missing}")
    commensurable = commensurability(runs, dirs)
    release = json.loads(args.release_comparison.read_text())
    report = {
        "sweeps": sweeps,
        "coverage": (
            f"complete: {sum(expected[c] for c in LABELS)}/"
            f"{sum(expected[c] for c in LABELS)} planned runs of the compared systems "
            "completed and validated"
        ),
        "commensurability": commensurable,
        "systems": system_rows(runs, dirs, expected),
        "paper_reported": {
            task: {
                "release": release["tasks"][task]["fig4_test_mse_first_last"][
                    "release"
                ][-1],
                "ours": release["tasks"][task]["fig4_test_mse_first_last"]["ours"][-1],
            }
            for task in PAPER_TASKS
        },
        "pairs": pair_rows(runs, commensurable["status"]),
        "claim_evidence": claim_evidence(runs),
        "extensions": extension_rows(args),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "fair_comparison.json").write_text(
        json.dumps(_jsonable(report), indent=2)
    )
    (RESULTS_DIR / "fair_comparison.md").write_text(to_markdown(report))
    print(RESULTS_DIR / "fair_comparison.json")
    print(RESULTS_DIR / "fair_comparison.md")
    print(f"commensurability: {commensurable['status']}")
    return 0


if __name__ == "__main__":
    np.seterr(all="raise")
    raise SystemExit(main())
