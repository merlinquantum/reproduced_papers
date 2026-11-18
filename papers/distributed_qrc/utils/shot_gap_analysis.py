"""Do finite shots reproduce the paper's noisy-cell degradation?

Reads the completed runs of one or more sweeps over ``configs/diagnostics/gap_srsr10_original.json``
(sweep S3 ``GAP-SHOTS`` and, if any, its retry sweep), groups them by
candidate (shot setting), and computes the predeclared degradation ratios
(test MAE, mean over seeds):

* Q-noisy / linear            vs Q-ideal / linear
* Q-ideal / kernel-noisy      vs Q-ideal / kernel-ideal
* Q-noisy / kernel-noisy      vs Q-ideal / kernel-ideal
* classical / kernel-noisy    vs classical / kernel-ideal

and compares them with the same ratios in the paper's Tables C1-C3 (SRSR 10).
Writes ``results/data/shot_gap_analysis.json``.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/shot_gap_analysis.py papers/distributed_qrc/outdir/sweep_GAP-SHOTS [more sweep dirs]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
RATIOS = {
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
TOLERANCE = 0.15


def label(candidate: dict) -> str:
    return json.dumps(candidate, sort_keys=True) if candidate else "exact"


def main() -> int:
    per_candidate: dict[str, list[dict]] = {}
    attempts = []
    for arg in sys.argv[1:]:
        sweep_dir = Path(arg)
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        for run in status["runs"]:
            attempts.append(
                {
                    "sweep": status["sweep_id"],
                    "candidate": run["candidate"],
                    "seed": run["seed"],
                    "status": run["status"],
                }
            )
            if run["status"] != "COMPLETED":
                continue
            metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
            cells = {
                (c["reservoir_type"], c["readout_type"]): c["metrics"]["test"]["mae"]
                for c in metrics["cells"]
            }
            per_candidate.setdefault(label(run["candidate"]), []).append(
                {"seed": run["seed"], "cells": cells}
            )

    paper = {
        (c["reservoir_type"], c["readout_type"]): c["mae"]
        for c in json.loads(
            (PAPER_DIR / "results" / "data" / "paper_tables.json").read_text()
        )["cells"]
        if c["architecture"] == "srsr" and c["configuration"] == "10"
    }
    paper_ratios = {name: paper[a] / paper[b] for name, (a, b) in RATIOS.items()}

    result = {"paper_ratios": paper_ratios, "attempts": attempts, "candidates": {}}
    print(
        f"{'ratio':48s} {'paper':>6s} "
        + " ".join(f"{c[:30]:>32s}" for c in per_candidate)
    )
    for name, (a, b) in RATIOS.items():
        row = f"{name:48s} {paper_ratios[name]:6.3f} "
        for cand, runs in per_candidate.items():
            values = np.array([r["cells"][a] / r["cells"][b] for r in runs])
            entry = result["candidates"].setdefault(
                cand, {"seeds": sorted(r["seed"] for r in runs), "ratios": {}}
            )
            entry["ratios"][name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
            row += f"{values.mean():23.3f} ± {entry['ratios'][name]['std']:.3f} "
        print(row)
    for cand, entry in result["candidates"].items():
        within = all(
            abs(entry["ratios"][n]["mean"] - paper_ratios[n]) <= TOLERANCE
            for n in RATIOS
        )
        entry["all_within_tolerance"] = within
        print(
            f"{cand}: seeds {entry['seeds']}; all four ratios within ±{TOLERANCE} of the paper: {within}"
        )
    out = PAPER_DIR / "results" / "data" / "shot_gap_analysis.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {os.path.relpath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
