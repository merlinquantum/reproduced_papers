"""Independent verification of the reproduction sweep evidence.

Re-derives everything from the runs' raw ``metrics.json`` / ``run_status.json``
files, without importing the other analysis utilities, and compares with the
recorded results:

1. coverage: planned vs completed candidate-by-seed sets per sweep, duplicate
   detection, the GAP-SHOTS failed attempt and its retry;
2. integrity: every completed run's metrics are finite, its config snapshot
   matches its recorded SHA-256, and its dataset fingerprint is the paper window
   (or none for interpolated data);
3. recomputation with the predeclared rules: observable/update selection,
   headline statistics (paper rule: max test R2 per configuration), paper-baseline
   seed ranks, noisy degradation ratios, best architecture per total N;
4. equivalence of the default behaviour across code changes (seed-42 smoke runs).

Writes ``results/checks/sweep_verification.json`` (``-R2`` for the evidence sweeps) and prints a summary; exits 1 on any
mismatch.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/verify_sweeps.py papers/distributed_qrc/outdir
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
PAPER_FINGERPRINT = "eca5bc5197fd4ad0b004b5c8441f092d297213bdecdbf549fd248ba133a0fb99"
TOL = 1e-12
BASELINE = ("classical", "linear")

SUFFIX = ""  # set from the optional second CLI argument, e.g. "-R2"


def sweep_path(outdir: Path, sweep_id: str) -> Path:
    return outdir / f"sweep_{sweep_id}{SUFFIX}"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_key(candidate: dict) -> str:
    return json.dumps(candidate, sort_keys=True)


def completed_runs(sweep_dir: Path) -> list[dict]:
    status = load(sweep_dir / "sweep_status.json")
    out = []
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        run_status = load(sweep_dir / run["run_status"])
        metrics = load(sweep_dir / run["metrics_path"])
        out.append(
            {
                "entry": run,
                "status": run_status,
                "metrics": metrics,
                "dir": sweep_dir / run["run_dir"],
            }
        )
    return out


def check_sweep(sweep_dir: Path) -> dict:
    status = load(sweep_dir / "sweep_status.json")
    planned = [(candidate_key(r["candidate"]), r["seed"]) for r in status["runs"]]
    runs = completed_runs(sweep_dir)
    observed = [
        (candidate_key(r["entry"]["candidate"]), r["entry"]["seed"]) for r in runs
    ]
    problems = []
    if len(set(planned)) != len(planned):
        problems.append("duplicate planned candidate-seed pairs")
    if len(set(observed)) != len(observed):
        problems.append("duplicate completed candidate-seed pairs")
    for run in runs:
        if run["status"]["status"] != "COMPLETED":
            problems.append(f"{run['dir'].name}: run status {run['status']['status']}")
        if (
            sha256(run["dir"] / "config_snapshot.json")
            != run["status"]["config_sha256"]
        ):
            problems.append(f"{run['dir'].name}: config hash mismatch")
        if sha256(run["dir"] / "metrics.json") != run["status"]["metrics_sha256"]:
            problems.append(f"{run['dir'].name}: metrics hash mismatch")
        handling = run["metrics"]["dataset"]["preprocessing"]["artifact_handling"]
        fingerprint = run["metrics"]["dataset"]["fingerprint_sha256"]
        if (handling == "keep" and fingerprint != PAPER_FINGERPRINT) or (
            handling == "interpolate" and fingerprint is not None
        ):
            problems.append(f"{run['dir'].name}: dataset fingerprint {fingerprint}")
        for cell in run["metrics"]["cells"]:
            for split in ("eval", "test", "test_mw"):
                if not all(math.isfinite(v) for v in cell["metrics"][split].values()):
                    problems.append(f"{run['dir'].name}: non-finite {split} metric")
    missing = sorted(set(planned) - set(observed))
    return {
        "sweep_status": status["status"],
        "expected_runs": status["expected_runs"],
        "planned": len(planned),
        "completed": len(observed),
        "missing": [list(m) for m in missing],
        "lib_sha256": status["code"].get("lib_sha256"),
        "problems": problems,
    }


def cells_of(metrics: dict) -> dict:
    return {
        (c["configuration"], c["reservoir_type"], c["readout_type"]): {
            f"{s}.{k}": v for s in ("eval", "test") for k, v in c["metrics"][s].items()
        }
        for c in metrics["cells"]
    }


def recompute_s1(outdir: Path) -> dict:
    sweep = sweep_path(outdir, "P1-R6R7")
    status = load(sweep / "sweep_status.json")
    order = []
    scores: dict[str, list[float]] = {}
    for run in completed_runs(sweep):
        key = candidate_key(run["entry"]["candidate"])
        if key not in order:
            order.append(key)
        cells = cells_of(run["metrics"])
        for cell in (
            ("10", "quantum_ideal", "linear"),
            ("10", "quantum_ideal", "quantum_kernel_ideal"),
        ):
            scores.setdefault(key, []).append(cells[cell]["eval.rmse"])
    order = [
        candidate_key(r["candidate"])
        for r in status["runs"]
        if candidate_key(r["candidate"]) in scores
    ]
    order = list(dict.fromkeys(order))
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    best = min(means.values())
    tied = [k for k in order if means[k] <= best * 1.01]
    recorded = load(sweep / "summary.json")["selection"]

    def label(key: str) -> str:
        if key == "{}":
            return "default"
        return "_".join(f"{a}-{b}" for a, b in sorted(json.loads(key).items()))

    match = (
        all(abs(means[k] - recorded["scores"][label(k)]) <= TOL for k in means)
        and label(tied[0]) == recorded["selected"]
    )
    return {
        "scores": means,
        "tied": tied,
        "selected": tied[0],
        "matches_recorded": match,
    }


def recompute_headline(outdir: Path) -> dict:
    recorded = load(PAPER_DIR / "results" / "data" / "headline_statistics.json")["ours"]
    out = {}
    for sweep_id, architecture in (
        ("E1", "srsr"),
        ("E2", "mrsr"),
        ("E3", "srmr"),
        ("E4", "mrmr"),
    ):
        per_metric = {}
        for metric in ("mae", "rmse"):
            averages, peaks = [], []
            for run in completed_runs(sweep_path(outdir, sweep_id)):
                cells = cells_of(run["metrics"])
                reductions = []
                for config in dict.fromkeys(c for c, _, _ in cells):
                    base = cells[(config, *BASELINE)][f"test.{metric}"]
                    candidates = {
                        k: v
                        for k, v in cells.items()
                        if k[0] == config and k[1:] != BASELINE
                    }
                    best = max(candidates.values(), key=lambda v: v["test.r2"])
                    reductions.append(1 - best[f"test.{metric}"] / base)
                averages.append(np.mean(reductions))
                peaks.append(np.max(reductions))
            rec = recorded[architecture]["paper_rule"][metric]
            per_metric[metric] = {
                "average_mean": float(np.mean(averages)),
                "peak_mean": float(np.mean(peaks)),
                "matches_recorded": abs(np.mean(averages) - rec["average_mean"]) <= TOL
                and abs(np.mean(peaks) - rec["peak_mean"]) <= TOL,
            }
        out[architecture] = per_metric
    return out


def recompute_classical_ranks(outdir: Path) -> dict:
    paper = {
        (c["architecture"], c["configuration"]): c["mae"]
        for c in load(PAPER_DIR / "results" / "data" / "paper_tables.json")["cells"]
        if (c["reservoir_type"], c["readout_type"]) == BASELINE
    }
    out = {}
    for architecture in ("srsr", "mrsr", "srmr", "mrmr"):
        sweep_id = f"GAP-CLASSICAL-{architecture.upper()}"
        runs = completed_runs(sweep_path(outdir, sweep_id))
        recorded = load(
            PAPER_DIR
            / "results"
            / "data"
            / f"classical_seed_spread_{architecture}.json"
        )["configurations"]
        inside, matches = 0, True
        for config in recorded:
            values = np.array(
                [cells_of(r["metrics"])[(config, *BASELINE)]["test.mae"] for r in runs]
            )
            reference = paper[(architecture, config)]
            rank = float((values < reference).mean())
            inside += int(values.min() <= reference <= values.max())
            matches &= (
                abs(rank - recorded[config]["paper_rank"]) <= TOL and values.size == 20
            )
        out[architecture] = {
            "inside_range": inside,
            "configurations": len(recorded),
            "matches_recorded": bool(matches),
        }
    return out


def recompute_ratios(outdir: Path) -> dict:
    ratios = {
        "Q-noisy/linear vs Q-ideal/linear": (
            ("quantum_noisy", "linear"),
            ("quantum_ideal", "linear"),
        ),
        "Q-ideal/kernel-noisy vs Q-ideal/kernel-ideal": (
            ("quantum_ideal", "quantum_kernel_noisy"),
            ("quantum_ideal", "quantum_kernel_ideal"),
        ),
        "Q-noisy/kernel-noisy vs Q-ideal/kernel-ideal": (
            ("quantum_noisy", "quantum_kernel_noisy"),
            ("quantum_ideal", "quantum_kernel_ideal"),
        ),
        "classical/kernel-noisy vs classical/kernel-ideal": (
            ("classical", "quantum_kernel_noisy"),
            ("classical", "quantum_kernel_ideal"),
        ),
    }
    groups: dict[str, list[dict]] = {}
    shot_sweeps = (
        ("GAP-SHOTS", "GAP-NOISE-SCALE")
        if SUFFIX
        else ("GAP-SHOTS", "GAP-SHOTS-RETRY", "GAP-NOISE-SCALE")
    )
    for sweep_id in shot_sweeps:
        for run in completed_runs(sweep_path(outdir, sweep_id)):
            groups.setdefault(
                candidate_key(run["entry"]["candidate"])
                if run["entry"]["candidate"]
                else "exact",
                [],
            ).append(cells_of(run["metrics"]))
    recorded = load(PAPER_DIR / "results" / "data" / "shot_gap_analysis.json")[
        "candidates"
    ]
    out = {}
    for key, runs in groups.items():
        values = {
            name: float(
                np.mean(
                    [
                        r[("10", *a)]["test.mae"] / r[("10", *b)]["test.mae"]
                        for r in runs
                    ]
                )
            )
            for name, (a, b) in ratios.items()
        }
        match = (
            all(
                abs(values[n] - recorded[key]["ratios"][n]["mean"]) <= TOL
                for n in ratios
            )
            and len(runs) == 3
        )
        out[key] = {"n_seeds": len(runs), "ratios": values, "matches_recorded": match}
    return out


def recompute_cross_architecture(outdir: Path) -> dict:
    def total(label: str) -> int:
        if "x" in label:
            a, b = label.split("x")
            return int(a) * int(b)
        return int(label.split("/")[0])

    rows: dict[tuple, list[float]] = {}
    for sweep_id, architecture in (
        ("E1", "srsr"),
        ("E2", "mrsr"),
        ("E3", "srmr"),
        ("E4", "mrmr"),
    ):
        for run in completed_runs(sweep_path(outdir, sweep_id)):
            for (config, res, ro), v in cells_of(run["metrics"]).items():
                if (res, ro) != BASELINE:
                    rows.setdefault((architecture, config, f"{res}|{ro}"), []).append(
                        v["test.r2"]
                    )
    recorded = load(PAPER_DIR / "results" / "data" / "cross_architecture.json")["ours"]
    out = {}
    for n in sorted({total(c) for _, c, _ in rows}):
        group = {k: np.mean(v) for k, v in rows.items() if total(k[1]) == n}
        best = max(group, key=group.get)
        out[str(n)] = {
            "best": list(best),
            "matches_recorded": best[0] == recorded[str(n)]["best_architecture"]
            and best[2] == recorded[str(n)]["best_cell"],
        }
    return out


def smoke_equivalence(outdir: Path) -> dict:
    reference = load(outdir / "run_20261014-131710" / "metrics.json")
    out = {}
    for run in ("run_20261014-223030", "run_20261014-234827"):
        other = load(outdir / run / "metrics.json")
        diff = max(
            abs(a["metrics"][s][k] - b["metrics"][s][k])
            for a, b in zip(reference["cells"], other["cells"])
            for s in ("eval", "test")
            for k in ("mae", "rmse", "r2")
        )
        out[run] = diff
    return out


def main() -> int:
    global SUFFIX
    outdir = Path(sys.argv[1])
    SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
    report = {"suffix": SUFFIX, "sweeps": {}}
    for sweep_dir in sorted(outdir.glob(f"sweep_*{SUFFIX}")):
        if SUFFIX and not sweep_dir.name.endswith(SUFFIX):
            continue
        if sweep_dir.name in ("sweep_smoke_check",):
            continue
        report["sweeps"][sweep_dir.name] = check_sweep(sweep_dir)
    report["s1_selection"] = recompute_s1(outdir)
    report["headline"] = recompute_headline(outdir)
    report["classical_ranks"] = recompute_classical_ranks(outdir)
    report["degradation_ratios"] = recompute_ratios(outdir)
    report["cross_architecture"] = recompute_cross_architecture(outdir)
    report["smoke_equivalence_max_abs_diff"] = smoke_equivalence(outdir)

    failures = []
    for name, sweep in report["sweeps"].items():
        print(
            f"{name:28s} status={sweep['sweep_status']:9s} completed {sweep['completed']}/{sweep['planned']} missing={sweep['missing']} problems={len(sweep['problems'])}"
        )
        failures += [f"{name}: {p}" for p in sweep["problems"]]
    checks = {
        "S1 selection": report["s1_selection"]["matches_recorded"],
        "E7 headline": all(
            m["matches_recorded"]
            for a in report["headline"].values()
            for m in a.values()
        ),
        "GAP-CLASSICAL ranks": all(
            a["matches_recorded"] for a in report["classical_ranks"].values()
        ),
        "S3/S4 ratios": all(
            c["matches_recorded"] for c in report["degradation_ratios"].values()
        ),
        "E5 cross-architecture": all(
            c["matches_recorded"] for c in report["cross_architecture"].values()
        ),
        "smoke equivalence": all(
            v == 0.0 for v in report["smoke_equivalence_max_abs_diff"].values()
        ),
    }
    for name, ok in checks.items():
        print(f"recompute {name:24s}: {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(name)
    print(
        f"S1 selected {report['s1_selection']['selected']} (tied {report['s1_selection']['tied']})"
    )
    report["checks"] = checks
    report["failures"] = failures
    out = PAPER_DIR / "results" / "checks" / "sweep_verification.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"failures: {len(failures)}; wrote {os.path.relpath(out)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
