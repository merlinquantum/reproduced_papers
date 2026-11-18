"""Independent verification of the MerLin sweep evidence.

Re-derives from the raw ``run_status.json`` / ``metrics.json`` /
``config_snapshot.json`` files of ``P1-MERLIN-SCALES`` and ``E{1..4}-MERLIN``:

1. coverage: planned vs completed candidate-by-seed sets, duplicates, attempts;
2. integrity: every run is COMPLETED on the pinned ``lib_sha256``, its config and
   metrics files match their recorded SHA-256, its metrics are finite and its
   dataset fingerprint is the paper window;
3. configuration: the photonic section of every main-sweep snapshot holds the
   scales frozen by M0, and the device matches the plan (local CPU or cuda:0);
4. M0 selection recomputed with the declared rule (mean eval RMSE over
   ``photonic|linear`` and ``photonic|photonic_kernel``, minimized; 1 % ties ->
   first-listed candidate);
5. hardware-aware fields per sweep: wall-clock per run, photonic kernel modes,
   photons, input states and Fock dimensions observed in the metrics.

Writes ``results/checks/merlin_sweep_verification.json``; exits 1 on any problem.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/verify_merlin_sweeps.py papers/distributed_qrc/outdir
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
PAPER_FINGERPRINT = "eca5bc5197fd4ad0b004b5c8441f092d297213bdecdbf549fd248ba133a0fb99"
PINNED_LIB = "764ee0c7f1151e06f07f74f7eb72f44abaeae5b989cd079eac9943f4366f41e1"
SEEDS = [0, 1, 2]
M0_CANDIDATES = [
    {"neuron_scale": 1.0, "kernel_scale": 2.0},
    {"neuron_scale": 3.14159, "kernel_scale": 2.0},
    {"neuron_scale": 1.0, "kernel_scale": 3.14159},
    {"neuron_scale": 3.14159, "kernel_scale": 3.14159},
]
FROZEN = {"neuron_scale": 1.0, "kernel_scale": 2.0}
PLAN = {
    "P1-MERLIN-SCALES": (M0_CANDIDATES, None),
    "E1-MERLIN": ([{}], None),
    "E2-MERLIN": ([{}], None),
    "E3-MERLIN": ([{}], "cuda:0"),
    "E4-MERLIN": ([{}], "cuda:0"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(values) -> bool:
    if isinstance(values, dict):
        return all(finite(v) for v in values.values())
    if isinstance(values, list):
        return all(finite(v) for v in values)
    if isinstance(values, float):
        return math.isfinite(values)
    return True


def seconds(start: str, end: str) -> float:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (datetime.strptime(end, fmt) - datetime.strptime(start, fmt)).total_seconds()


def main() -> int:
    outdir = Path(sys.argv[1])
    report: dict = {"pinned_lib_sha256": PINNED_LIB, "sweeps": {}}
    problems: list[str] = []
    m0_scores: dict[str, list[float]] = {}
    for sweep_id, (candidates, device) in PLAN.items():
        sweep_dir = outdir / f"sweep_{sweep_id}"
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        planned = {
            (json.dumps(c, sort_keys=True), s) for c in candidates for s in SEEDS
        }
        observed = [
            (json.dumps(r["candidate"], sort_keys=True), r["seed"])
            for r in status["runs"]
            if r["status"] == "COMPLETED"
        ]
        if len(observed) != len(set(observed)):
            problems.append(f"{sweep_id}: duplicate completed runs")
        if set(observed) != planned or status["expected_runs"] != len(planned):
            problems.append(f"{sweep_id}: coverage {sorted(set(observed))} != plan")
        if status["status"] != "COMPLETED" or status["device"] != device:
            problems.append(
                f"{sweep_id}: status {status['status']} device {status['device']}"
            )
        wall, kernel_shapes = [], set()
        for r in status["runs"]:
            run_dir = sweep_dir / r["run_dir"]
            rs = json.loads((run_dir / "run_status.json").read_text())
            metrics = json.loads((run_dir / "metrics.json").read_text())
            snapshot = json.loads((run_dir / "config_snapshot.json").read_text())
            if rs["status"] != "COMPLETED" or rs["code"]["lib_sha256"] != PINNED_LIB:
                problems.append(
                    f"{sweep_id}/{r['run_dir']}: {rs['status']} {rs['code']['lib_sha256']}"
                )
            if sha256(run_dir / "config_snapshot.json") != rs["config_sha256"]:
                problems.append(f"{sweep_id}/{r['run_dir']}: config sha mismatch")
            if sha256(run_dir / "metrics.json") != rs["metrics_sha256"]:
                problems.append(f"{sweep_id}/{r['run_dir']}: metrics sha mismatch")
            if not finite([c["metrics"] for c in metrics["cells"]]):
                problems.append(f"{sweep_id}/{r['run_dir']}: non-finite metrics")
            if metrics["dataset"]["fingerprint_sha256"] != PAPER_FINGERPRINT:
                problems.append(f"{sweep_id}/{r['run_dir']}: dataset fingerprint")
            expected_scales = {**FROZEN, **r["candidate"]}
            if any(snapshot["photonic"][k] != v for k, v in expected_scales.items()):
                problems.append(
                    f"{sweep_id}/{r['run_dir']}: photonic {snapshot['photonic']}"
                )
            if snapshot.get("device") != (device or "cpu"):
                problems.append(
                    f"{sweep_id}/{r['run_dir']}: snapshot device {snapshot.get('device')}"
                )
            wall.append(seconds(rs["started_at"], rs["completed_at"]))
            for cell in metrics["cells"]:
                if cell["readout_type"] == "photonic_kernel":
                    for inst in cell["instances"]:
                        kernel_shapes.add(
                            (
                                inst["n_modes"],
                                inst["n_photons"],
                                tuple(inst["input_state"]),
                                inst["fock_dim"],
                            )
                        )
            if sweep_id == "P1-MERLIN-SCALES":
                cells = {
                    (c["reservoir_type"], c["readout_type"]): c["metrics"]["eval"][
                        "rmse"
                    ]
                    for c in metrics["cells"]
                }
                score = (
                    cells[("photonic", "linear")]
                    + cells[("photonic", "photonic_kernel")]
                ) / 2
                m0_scores.setdefault(
                    json.dumps(r["candidate"], sort_keys=True), []
                ).append(score)
        report["sweeps"][sweep_id] = {
            "status": status["status"],
            "machine_device": device or "local cpu",
            "planned_runs": len(planned),
            "completed_runs": len(observed),
            "attempts": len(status["runs"]),
            "started_at": status["started_at"],
            "completed_at": status["completed_at"],
            "run_wall_clock_s": wall,
            "photonic_kernel_shapes": [
                {"n_modes": m, "n_photons": p, "input_state": list(s), "fock_dim": f}
                for m, p, s, f in sorted(kernel_shapes)
            ],
        }
    means = {k: sum(v) / len(v) for k, v in m0_scores.items()}
    best = min(means.values())
    tied = [
        c for c in M0_CANDIDATES if means[json.dumps(c, sort_keys=True)] <= best * 1.01
    ]
    report["m0_selection"] = {
        "mean_eval_rmse": means,
        "tied_within_1pct": tied,
        "selected": tied[0],
    }
    if tied[0] != FROZEN:
        problems.append(f"M0 selection {tied[0]} != frozen {FROZEN}")
    report["problems"] = problems
    path = PAPER_DIR / "results" / "checks" / "merlin_sweep_verification.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for sweep_id, v in report["sweeps"].items():
        print(
            f"{sweep_id}: {v['status']} {v['completed_runs']}/{v['planned_runs']} on {v['machine_device']}, "
            f"run wall-clock {min(v['run_wall_clock_s']):.0f}-{max(v['run_wall_clock_s']):.0f} s, "
            f"kernel shapes {[(s['n_modes'], s['fock_dim']) for s in v['photonic_kernel_shapes']]}"
        )
    print(f"M0 selected {tied[0]} (tied: {tied})")
    print(f"problems: {problems or 'none'}")
    print(f"wrote {os.path.relpath(path)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
