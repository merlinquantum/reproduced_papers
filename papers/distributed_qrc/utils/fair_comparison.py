"""Comparison with the additional classical baselines, applying the decision rules declared before the runs.

Reads the raw ``sweep_status.json`` / ``metrics.json`` / ``test_predictions.npz``
of the gate-model reproduction sweeps (``E{1..4}-R2``), the classical-baseline sweeps
(``E{1..4}-CLASSICAL``, ``B1B3-RAW``, ``C6-{SRSR,MRSR,MRMR}``) and the
MerLin sweeps (``E{1..4}-MERLIN``), and writes ``results/data/fair_comparison.json``:

* integrity: coverage (seeds 0-2 completed), pinned code hash per sweep family,
  in-run anchors equal to R2, no-reservoir seed invariance;
* per configuration: every system's test MAE (mean, sample std over seeds);
* eval-selected cells: per seed and configuration, the cell with the minimum
  eval RMSE within the quantum/hybrid pool, the fair classical pool and the
  MerLin photonic pool;
* claim outcomes C1, C2, C4, C5 (decision rules of the plan; C4 = noisy cells, C5 = best
  model), the best architecture per size and the noise side check of Sec. III-B5; the headline reduction vs the best fair classical cell;
* Diebold-Mariano tests at SRSR 10, MRSR 3x20, SRMR 10/2 and MRMR 2x5;
* wall-clock: per sweep and run, and mean readout seconds per cell type.

Per-configuration decision: A "better" if mean(A) - mean(B) < -2 sigma_p, "worse"
if > 2 sigma_p, else "tie", with sigma_p = sqrt((s_A^2 + s_B^2) / 2).

Usage (from the repository root)::

    python papers/distributed_qrc/utils/fair_comparison.py papers/distributed_qrc/outdir
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from diebold_mariano import dm_test

PAPER_DIR = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2]
ARCHS = {"srsr": "E1", "mrsr": "E2", "srmr": "E3", "mrmr": "E4"}
LIB = {
    "R2": "19c36e44",
    "MERLIN": "764ee0c7",
    "CLASSICAL": "086ec349",
}
BASELINE = ("classical", "linear")
QUANTUM_POOL_EXCLUDE = {BASELINE}
FAIR_CLASSICAL = [
    ("classical", "linear"),  # paper baseline
    (
        "classical",
        "rbf_kernel",
    ),  # classical-kernel control on the paper baseline's features
    ("esn_tuned", "linear"),  # tuned ESN (ReservoirPy, tuned on eval)
    ("esn_tuned", "rbf_kernel"),  # tuned ESN + RBF kernel readout
    ("raw_lags", "persistence"),  # no reservoir, next hour = newest lag
    ("raw_lags", "linear"),  # no reservoir: AR(4), linear ridge on the 4 raw lags
    (
        "raw_lags",
        "quantum_kernel_ideal",
    ),  # no reservoir: the paper's kernel, evaluated classically
    ("raw_lags", "rbf_kernel"),  # no reservoir: RBF kernel
]
MERLIN_POOL = [
    ("photonic", "linear"),
    ("photonic", "photonic_kernel"),
    ("classical", "photonic_kernel"),
]
DM_CONFIGS = {"srsr": "10", "mrsr": "3x20", "srmr": "10/2", "mrmr": "2x5"}
C6_PAIRS = {
    "10": "2x5",
    "15": "3x5",
    "20": "2x10",
    "25": "5x5",
    "30": "3x10",
    "40": "2x20",
    "50": "5x10",
    "60": "3x20",
}
C6_MRMR_TOTALS = ["40", "50", "60"]
FMT = "%Y-%m-%dT%H:%M:%SZ"


def load_sweep(
    outdir: Path, sweep_id: str, lib_prefix: str, problems: list[str]
) -> dict:
    """{candidate_json: {seed: {"cells": {...}, "predictions": path, "seconds": ...}}}."""

    sweep_dir = outdir / f"sweep_{sweep_id}"
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        problems.append(f"{sweep_id}: status {status['status']}")
    if not status["code"]["lib_sha256"].startswith(lib_prefix):
        problems.append(
            f"{sweep_id}: lib {status['code']['lib_sha256'][:12]} != {lib_prefix}"
        )
    out: dict = {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        run_dir = sweep_dir / run["run_dir"]
        rs = json.loads((run_dir / "run_status.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        cells = {}
        for c in metrics["cells"]:
            key = (c["configuration"], c["reservoir_type"], c["readout_type"])
            values = [
                c["metrics"][s][m]
                for s in ("eval", "test")
                for m in ("mae", "rmse", "r2")
            ]
            if not all(np.isfinite(values)):
                problems.append(f"{sweep_id}/{run['run_dir']}: non-finite {key}")
            cells[key] = {
                "eval_rmse": c["metrics"]["eval"]["rmse"],
                "test": c["metrics"]["test"],
                "test_mw_mae": c["metrics"]["test_mw"]["mae"],
                "seconds": sum(i["seconds"] for i in c["instances"]),
                "tag": f"{c['architecture']}:{c['configuration']}|{c['reservoir_type']}|{c['readout_type']}",
            }
        slot = out.setdefault(json.dumps(run["candidate"], sort_keys=True), {})
        if run["seed"] in slot:
            problems.append(f"{sweep_id}: duplicate seed {run['seed']}")
        slot[run["seed"]] = {
            "cells": cells,
            "predictions": run_dir / "test_predictions.npz",
            "seconds": (
                datetime.strptime(rs["completed_at"], FMT)
                - datetime.strptime(rs["started_at"], FMT)
            ).total_seconds(),
        }
    for candidate, seeds in out.items():
        if sorted(seeds) != SEEDS:
            problems.append(f"{sweep_id} {candidate}: seeds {sorted(seeds)}")
    return out


def decide(a: np.ndarray, b: np.ndarray) -> tuple[str, float, float]:
    delta = a.mean() - b.mean()
    pooled = float(np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2))
    if delta < -2 * pooled:
        return "better", float(delta), pooled
    if delta > 2 * pooled:
        return "worse", float(delta), pooled
    return "tie", float(delta), pooled


def outcome(n_better: int, n_total: int = 8) -> str:
    if n_total == 3:
        return (
            "SUPPORTED"
            if n_better >= 2
            else ("PARTIALLY_SUPPORTED" if n_better == 1 else "UNSUPPORTED")
        )
    return (
        "SUPPORTED"
        if n_better >= 6
        else ("PARTIALLY_SUPPORTED" if n_better >= 3 else "UNSUPPORTED")
    )


def mean_std(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1))}


def analyse_architecture(
    arch: str,
    exp: str,
    outdir: Path,
    raw: dict,
    raw_cells: dict,
    problems: list[str],
    wallclock: dict,
    readout_seconds: dict,
) -> dict:
    """Per-configuration tables, eval-selected cells, claim counts, headline and DM for one architecture."""

    r2 = load_sweep(outdir, f"{exp}-R2", LIB["R2"], problems)["{}"]
    cl = load_sweep(outdir, f"{exp}-CLASSICAL", LIB["CLASSICAL"], problems)["{}"]
    ml = load_sweep(outdir, f"{exp}-MERLIN", LIB["MERLIN"], problems)["{}"]
    for name, sweep in (("R2", r2), ("CLASSICAL", cl), ("MERLIN", ml)):
        wallclock[f"{exp}-{name}"] = [sweep[s]["seconds"] for s in SEEDS]
        for s in SEEDS:
            for key, cell in sweep[s]["cells"].items():
                readout_seconds.setdefault(
                    f"{exp}-{name}|{key[1]}|{key[2]}", []
                ).append(cell["seconds"])
    configs = list(dict.fromkeys(k[0] for k in r2[0]["cells"]))
    anchor = max(
        abs(cl[s]["cells"][(c, *a)]["test"][m] - r2[s]["cells"][(c, *a)]["test"][m])
        for s in SEEDS
        for c in configs
        for a in (("classical", "linear"), ("quantum_ideal", "linear"))
        for m in ("mae", "rmse", "r2")
    )
    if anchor > 1e-9:
        problems.append(f"{arch}: anchors differ from R2 by {anchor:.2e}")

    def cell(seed: int, config: str, system: tuple[str, str]) -> dict:
        if system[0] == "raw_lags":  # no-reservoir floors live in the B1B3-RAW sweep
            return raw[seed]["cells"][raw_cells[system]] | {
                "source": "B1B3-RAW",
                "seed_run": raw[seed],
            }
        for name, sweep in (("R2", r2), ("CLASSICAL", cl), ("MERLIN", ml)):
            if (config, *system) in sweep[seed]["cells"]:
                return sweep[seed]["cells"][(config, *system)] | {
                    "source": f"{exp}-{name}",
                    "seed_run": sweep[seed],
                }
        raise KeyError((arch, config, system))

    quantum_pool = [
        k[1:]
        for k in r2[0]["cells"]
        if k[0] == configs[0] and k[1:] not in QUANTUM_POOL_EXCLUDE
    ]
    systems = {
        **{"|".join(s): s for s in quantum_pool},
        **{"|".join(s): s for s in FAIR_CLASSICAL},
        "quantum_ideal|rbf_kernel": ("quantum_ideal", "rbf_kernel"),
        **{"|".join(s): s for s in MERLIN_POOL},
    }
    per_config: dict = {}
    selected: dict = {}
    for config in configs:
        rows = {}
        for label, system in systems.items():
            maes = [cell(s, config, system)["test"]["mae"] for s in SEEDS]
            rows[label] = mean_std(maes)
        sel = {}
        for pool_name, pool in (
            ("quantum", quantum_pool),
            ("fair_classical", FAIR_CLASSICAL),
            ("merlin", MERLIN_POOL),
        ):
            picks = []
            for s in SEEDS:
                best = min(pool, key=lambda p: cell(s, config, p)["eval_rmse"])
                c = cell(s, config, best)
                picks.append(
                    {
                        "seed": s,
                        "cell": "|".join(best),
                        "test_mae": c["test"]["mae"],
                        "test_rmse": c["test"]["rmse"],
                        "source": c["source"],
                    }
                )
            sel[pool_name] = {
                "picks": picks,
                "test_mae": mean_std([p["test_mae"] for p in picks]),
                "test_rmse": mean_std([p["test_rmse"] for p in picks]),
            }
        per_config[config] = rows
        selected[config] = sel

    def compare(get_a, get_b) -> dict:
        rows, counts = {}, {"better": 0, "tie": 0, "worse": 0}
        for config in configs:
            a, b = np.array(get_a(config)), np.array(get_b(config))
            verdict, delta, pooled = decide(a, b)
            counts[verdict] += 1
            rows[config] = {
                "a_mean": float(a.mean()),
                "b_mean": float(b.mean()),
                "delta": delta,
                "pooled_std": pooled,
                "verdict": verdict,
            }
        return {
            "configurations": rows,
            "counts": counts,
            "outcome": outcome(counts["better"]),
        }

    def sel_mae(pool: str):
        return lambda config: [p["test_mae"] for p in selected[config][pool]["picks"]]

    def sys_mae(system):
        return lambda config: [cell(s, config, system)["test"]["mae"] for s in SEEDS]

    claims = {
        "C1_vs_fair_classical": compare(sel_mae("quantum"), sel_mae("fair_classical")),
        "C1_vs_paper_baseline": compare(sel_mae("quantum"), sys_mae(BASELINE)),
        "C2_quantum_features": compare(
            sys_mae(("quantum_ideal", "quantum_kernel_ideal")),
            sys_mae(("quantum_ideal", "rbf_kernel")),
        ),
        "C2_classical_features": compare(
            sys_mae(("classical", "quantum_kernel_ideal")),
            sys_mae(("classical", "rbf_kernel")),
        ),
        "C4_noisy_fully_quantum_vs_fair_classical": compare(
            sys_mae(("quantum_noisy", "quantum_kernel_noisy")),
            sel_mae("fair_classical"),
        ),
        "MERLIN_vs_fair_classical": compare(
            sel_mae("merlin"), sel_mae("fair_classical")
        ),
    }
    # Headline reduction vs the best fair classical cell (eval-selected on both sides).
    reductions = {"mae": [], "rmse": []}
    for i in range(len(SEEDS)):
        for metric in reductions:
            per = [
                1
                - selected[c]["quantum"]["picks"][i][f"test_{metric}"]
                / selected[c]["fair_classical"]["picks"][i][f"test_{metric}"]
                for c in configs
            ]
            reductions[metric].append((np.mean(per), np.max(per)))
    headline = {
        m: {
            "average": mean_std([v[0] for v in vals]),
            "peak": mean_std([v[1] for v in vals]),
        }
        for m, vals in reductions.items()
    }
    # Diebold-Mariano at the declared headline configuration.
    dm_config = DM_CONFIGS[arch]
    dm_rows = []
    for i, s in enumerate(SEEDS):
        q = selected[dm_config]["quantum"]["picks"][i]
        c = selected[dm_config]["fair_classical"]["picks"][i]
        pred_q = np.load(
            cell(s, dm_config, tuple(q["cell"].split("|")))["seed_run"]["predictions"]
        )
        c_cell = cell(s, dm_config, tuple(c["cell"].split("|")))
        pred_c = np.load(c_cell["seed_run"]["predictions"])
        target = pred_q["test_targets"].astype(np.float64)
        if not np.array_equal(target, pred_c["test_targets"].astype(np.float64)):
            problems.append(f"{arch} DM: test targets differ")
        yq = pred_q[f"{arch}:{dm_config}|{q['cell']}"].astype(np.float64)
        yc = pred_c[c_cell["tag"]].astype(np.float64)
        for loss_name, loss in (("abs", np.abs), ("sq", np.square)):
            for lag in (0, 24):
                stat, p = dm_test(loss(target - yq), loss(target - yc), lag)
                dm_rows.append(
                    {
                        "seed": s,
                        "quantum_cell": q["cell"],
                        "classical_cell": c["cell"],
                        "loss": loss_name,
                        "lag": lag,
                        "dm": stat,
                        "p_value": p,
                    }
                )
    return {
        "anchor_max_abs_diff": anchor,
        "per_configuration": per_config,
        "eval_selected": selected,
        "claims": claims,
        "headline_vs_best_classical": headline,
        "diebold_mariano": {"configuration": dm_config, "rows": dm_rows},
    }


def main() -> int:
    outdir = Path(sys.argv[1])
    problems: list[str] = []
    raw = load_sweep(outdir, "B1B3-RAW", LIB["CLASSICAL"], problems)["{}"]
    raw_cells = {k[1:]: k for k in raw[0]["cells"]}
    for k in raw[0]["cells"]:
        maes = [raw[s]["cells"][k]["test"]["mae"] for s in SEEDS]
        if max(maes) != min(maes):
            problems.append(f"no-reservoir cell not seed-invariant: {k}")

    report: dict = {"problems": problems, "architectures": {}}
    wallclock: dict = {}
    readout_seconds: dict = {}
    for arch, exp in ARCHS.items():
        report["architectures"][arch] = analyse_architecture(
            arch, exp, outdir, raw, raw_cells, problems, wallclock, readout_seconds
        )

    # C5: MRSR 3x20 hybrid (the paper's best model) vs eval-selected fair classical.
    mrsr = report["architectures"]["mrsr"]
    hybrid = mrsr["per_configuration"]["3x20"]["classical|quantum_kernel_ideal"]
    fair = [
        p["test_mae"] for p in mrsr["eval_selected"]["3x20"]["fair_classical"]["picks"]
    ]
    r2_mrsr = load_sweep(outdir, "E2-R2", LIB["R2"], problems)["{}"]
    h = np.array(
        [
            r2_mrsr[s]["cells"][("3x20", "classical", "quantum_kernel_ideal")]["test"][
                "mae"
            ]
            for s in SEEDS
        ]
    )
    verdict, delta, pooled = decide(h, np.array(fair))
    report["C5_best_model"] = {
        "hybrid_3x20": hybrid,
        "fair_classical_3x20": mean_std(fair),
        "paper_value": 0.0113,
        "verdict": verdict,
        "delta": delta,
        "pooled_std": pooled,
        "outcome": "SUPPORTED" if verdict == "better" else "UNSUPPORTED",
    }

    # Best architecture per total N (lowest seed-mean test MAE).
    totals = {"srsr": {}, "mrsr": {}, "srmr": {}, "mrmr": {}}
    for arch, data in report["architectures"].items():
        for config, sel in data["eval_selected"].items():
            if arch == "srsr":
                n = int(config)
            elif arch == "srmr":
                n = int(config.split("/")[0])
            else:
                k, m = config.split("x")
                n = int(k) * int(m)
            totals[arch][n] = {
                "quantum": sel["quantum"]["test_mae"]["mean"],
                "fair_classical": sel["fair_classical"]["test_mae"]["mean"],
            }
    c3 = {}
    for n in sorted(totals["srsr"]):
        row = {a: totals[a].get(n) for a in totals if n in totals[a]}
        c3[n] = {
            "best_quantum_architecture": min(row, key=lambda a: row[a]["quantum"]),
            "best_fair_classical_architecture": min(
                row, key=lambda a: row[a]["fair_classical"]
            ),
            "values": row,
        }
    report["best_architecture_per_size"] = c3

    # Noise side check (Sec. III-B5 remark): same-device control at x10.
    c6 = {}
    sweeps = {
        a: load_sweep(outdir, f"C6-{a.upper()}", LIB["CLASSICAL"], problems)
        for a in ("srsr", "mrsr", "mrmr")
    }
    for candidate in sweeps["srsr"]:
        label = json.loads(candidate)["noise_mapping"]

        def noisy(arch: str, config: str, candidate: str = candidate) -> np.ndarray:
            return np.array(
                [
                    sweeps[arch][candidate][s]["cells"][
                        (config, "quantum_noisy", "quantum_kernel_noisy")
                    ]["test"]["mae"]
                    for s in SEEDS
                ]
            )

        mrsr_rows, mrmr_rows = {}, {}
        for n, kxn in C6_PAIRS.items():
            verdict, delta, pooled = decide(noisy("mrsr", kxn), noisy("srsr", n))
            mrsr_rows[n] = {
                "srsr": float(noisy("srsr", n).mean()),
                "mrsr": float(noisy("mrsr", kxn).mean()),
                "delta": delta,
                "pooled_std": pooled,
                "verdict": verdict,
            }
            if n in C6_MRMR_TOTALS:
                verdict, delta, pooled = decide(noisy("mrmr", kxn), noisy("srsr", n))
                mrmr_rows[n] = {
                    "srsr": float(noisy("srsr", n).mean()),
                    "mrmr": float(noisy("mrmr", kxn).mean()),
                    "delta": delta,
                    "pooled_std": pooled,
                    "verdict": verdict,
                }
        n_mrsr = sum(r["verdict"] == "better" for r in mrsr_rows.values())
        n_mrmr = sum(r["verdict"] == "better" for r in mrmr_rows.values())
        c6[label] = {
            "candidate": json.loads(candidate),
            "mrsr_vs_srsr": {
                "rows": mrsr_rows,
                "better": n_mrsr,
                "outcome": outcome(n_mrsr),
            },
            "mrmr_vs_srsr_from_40": {
                "rows": mrmr_rows,
                "better": n_mrmr,
                "outcome": outcome(n_mrmr, 3),
            },
        }
        for arch in sweeps:
            wallclock[f"C6-{arch.upper()} {label}"] = [
                sweeps[arch][candidate][s]["seconds"] for s in SEEDS
            ]
    report["noise_side_check_x10"] = c6

    wallclock["B1B3-RAW"] = [raw[s]["seconds"] for s in SEEDS]
    report["wallclock_run_seconds"] = {
        k: mean_std(v) | {"runs": len(v)} for k, v in wallclock.items()
    }
    report["readout_seconds_per_cell"] = {
        k: mean_std(v) for k, v in readout_seconds.items()
    }

    path = PAPER_DIR / "results" / "data" / "fair_comparison.json"
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for arch, data in report["architectures"].items():
        print(f"== {arch}: anchors {data['anchor_max_abs_diff']:.1e}")
        for name, claim in data["claims"].items():
            print(f"  {name:42s} {claim['outcome']:20s} {claim['counts']}")
        h = data["headline_vs_best_classical"]["mae"]
        print(
            f"  headline vs fair classical: MAE avg {h['average']['mean']:+.1%} ± {h['average']['std']:.1%}, peak {h['peak']['mean']:+.1%}"
        )
        for config, sel in data["eval_selected"].items():
            print(
                f"    {config:6s} Q {sel['quantum']['test_mae']['mean']:.5f}±{sel['quantum']['test_mae']['std']:.5f}"
                f"  C {sel['fair_classical']['test_mae']['mean']:.5f}±{sel['fair_classical']['test_mae']['std']:.5f}"
                f"  M {sel['merlin']['test_mae']['mean']:.5f}  C cells {[p['cell'] for p in sel['fair_classical']['picks']]}"
            )
        dm = [
            r
            for r in data["diebold_mariano"]["rows"]
            if r["loss"] == "abs" and r["lag"] == 24
        ]
        print(
            f"  DM {data['diebold_mariano']['configuration']} (abs, lag 24): "
            + "  ".join(f"{r['dm']:+.2f} (p {r['p_value']:.1e})" for r in dm)
        )
    print(
        f"C5: {report['C5_best_model']['outcome']} ({report['C5_best_model']['verdict']}), hybrid {report['C5_best_model']['hybrid_3x20']}, fair {report['C5_best_model']['fair_classical_3x20']}"
    )
    for label, v in report["noise_side_check_x10"].items():
        print(
            f"noise side check {label}: MRSR {v['mrsr_vs_srsr']['outcome']} ({v['mrsr_vs_srsr']['better']}/8), MRMR {v['mrmr_vs_srsr_from_40']['outcome']} ({v['mrmr_vs_srsr_from_40']['better']}/3)"
        )
    print(f"problems: {problems or 'none'}")
    print(f"wrote {os.path.relpath(path)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
