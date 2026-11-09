"""Apply the verdict rules declared before the robustness and gap sweeps were run.

Reads the sweeps' raw ``metrics.json`` files and ``results/data/paper_tables.json``
and writes ``results/data/robustness_verdicts.json`` with, exactly as declared before launch:

* ``preprocessing_robustness`` (sweep P1-ROBUST-R2): per cell, "material" if
  |delta mean| > the ``{}`` seed std of that cell; headline at N = 10 "unchanged"
  if |delta| < the ``{}`` headline seed std;
* ``finite_shots`` / ``noise_scale`` (sweeps GAP-SHOTS-R2, GAP-NOISE-SCALE-R2): per
  candidate, consistent / over-degrades / under-degrades / mixed from the 4
  degradation ratios (tolerance 0.15); closest noise scale by the minimum sum of
  |ratio - paper| (ties within 1 % -> smaller scale);
* ``missing_intercept`` (sweep GAP-INTERCEPT-R2): hypothesis supported if >= 13/24
  linear cells have |no-intercept mean - paper| <= no-intercept seed std;
* ``paper_baseline_seed_spread`` (sweeps GAP-CLASSICAL-*-R2): per architecture,
  "consistent with seed variance" if the paper value is within our 20-seed
  [min, max] for >= 5/8 configurations.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/robustness_verdicts.py papers/distributed_qrc/outdir
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
TOL = 0.15
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


def runs(sweep_dir: Path) -> list[tuple[dict, dict]]:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_dir} is {status['status']}")
    out = []
    for r in status["runs"]:
        m = json.loads((sweep_dir / r["metrics_path"]).read_text())
        cells = {
            (c["configuration"], c["reservoir_type"], c["readout_type"]): c["metrics"][
                "test"
            ]
            for c in m["cells"]
        }
        out.append((r["candidate"], cells))
    return out


def by_candidate(items: list[tuple[dict, dict]]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for cand, cells in items:
        groups.setdefault(json.dumps(cand, sort_keys=True), []).append(cells)
    return groups


def paper_rule_reduction(cells: dict, config: str) -> float:
    base = cells[(config, "classical", "linear")]["mae"]
    best = max(
        (
            v
            for k, v in cells.items()
            if k[0] == config and k[1:] != ("classical", "linear")
        ),
        key=lambda v: v["r2"],
    )
    return 1 - best["mae"] / base


def main() -> int:
    outdir = Path(sys.argv[1])
    paper = json.loads(
        (PAPER_DIR / "results" / "data" / "paper_tables.json").read_text()
    )["cells"]
    pmae = {
        (
            c["architecture"],
            c["configuration"],
            c["reservoir_type"],
            c["readout_type"],
        ): c["mae"]
        for c in paper
    }
    out = {}

    # Preprocessing robustness (sweep P1-ROBUST-R2).
    g = by_candidate(runs(outdir / "sweep_P1-ROBUST-R2"))
    base = g["{}"]
    s2 = {}
    for cand, rs in g.items():
        if cand == "{}":
            continue
        cells = {}
        for key in base[0]:
            b = np.array([r[key]["mae"] for r in base])
            c = np.array([r[key]["mae"] for r in rs])
            cells["|".join(key)] = {
                "delta": float(c.mean() - b.mean()),
                "base_std": float(b.std(ddof=1)),
                "material": bool(abs(c.mean() - b.mean()) > b.std(ddof=1)),
            }
        hb = np.array([paper_rule_reduction(r, "10") for r in base])
        hc = np.array([paper_rule_reduction(r, "10") for r in rs])
        s2[cand] = {
            "cells": cells,
            "headline_delta": float(hc.mean() - hb.mean()),
            "headline_base_std": float(hb.std(ddof=1)),
            "headline_unchanged": bool(abs(hc.mean() - hb.mean()) < hb.std(ddof=1)),
        }
    out["preprocessing_robustness"] = s2

    # Finite shots and noise scale (sweeps GAP-SHOTS-R2, GAP-NOISE-SCALE-R2).
    paper_ratio = {
        n: pmae[("srsr", "10", *a)] / pmae[("srsr", "10", *b)]
        for n, (a, b) in RATIOS.items()
    }
    shot = by_candidate(runs(outdir / "sweep_GAP-SHOTS-R2"))
    scale = by_candidate(runs(outdir / "sweep_GAP-NOISE-SCALE-R2"))
    categories = {}
    for cand, rs in {**shot, **scale}.items():
        ratios = {
            n: float(np.mean([r[("10", *a)]["mae"] / r[("10", *b)]["mae"] for r in rs]))
            for n, (a, b) in RATIOS.items()
        }
        dev = {n: ratios[n] - paper_ratio[n] for n in RATIOS}
        if all(abs(d) <= TOL for d in dev.values()):
            cat = "consistent"
        elif sum(d > TOL for d in dev.values()) >= 3:
            cat = "over-degrades"
        elif sum(d < -TOL for d in dev.values()) >= 3:
            cat = "under-degrades"
        else:
            cat = "mixed"
        categories[cand] = {
            "ratios": ratios,
            "category": cat,
            "sum_abs_dev": float(sum(abs(d) for d in dev.values())),
        }
    out["finite_shots"] = {k: v for k, v in categories.items() if k in shot}
    out["finite_shots_explain_noisy_gap"] = any(
        v["category"] == "consistent"
        for k, v in categories.items()
        if k in shot and k != "{}"
    )
    s4 = {k: v for k, v in categories.items() if k in scale}
    s4['{"noise_scale": 1} (= exact, shot-free)'] = categories["{}"]
    best = min(v["sum_abs_dev"] for v in s4.values())
    tied = sorted(
        (k for k, v in s4.items() if v["sum_abs_dev"] <= best * 1.01),
        key=lambda k: float(k.split(":")[1].split("}")[0]),
    )
    out["noise_scale"] = {
        "candidates": s4,
        "closest_scale": tied[0],
        "any_consistent": any(v["category"] == "consistent" for v in s4.values()),
    }
    out["paper_ratios"] = paper_ratio

    # Missing intercept (sweep GAP-INTERCEPT-R2).
    g5 = by_candidate(runs(outdir / "sweep_GAP-INTERCEPT-R2"))
    noint = g5[json.dumps({"linear_intercept": False})]
    within = 0
    for key in noint[0]:
        v = np.array([r[key]["mae"] for r in noint])
        within += int(abs(v.mean() - pmae[("srsr", *key)]) <= v.std(ddof=1))
    out["missing_intercept"] = {
        "cells_within_seed_std": within,
        "cells": len(noint[0]),
        "intercept_hypothesis": "supported" if within >= 13 else "rejected",
    }

    # GAP-CLASSICAL-*-R2
    gc = {}
    for arch in ("srsr", "mrsr", "srmr", "mrmr"):
        rs = [
            cells
            for _, cells in runs(outdir / f"sweep_GAP-CLASSICAL-{arch.upper()}-R2")
        ]
        configs = sorted({k[0] for k in rs[0]})
        inside = 0
        for config in configs:
            v = np.array([r[(config, "classical", "linear")]["mae"] for r in rs])
            ref = pmae[(arch, config, "classical", "linear")]
            inside += int(v.min() <= ref <= v.max())
        gc[arch] = {
            "inside": inside,
            "configurations": len(configs),
            "consistent_with_seed_variance": inside >= 5,
        }
    out["paper_baseline_seed_spread"] = gc

    path = PAPER_DIR / "results" / "data" / "robustness_verdicts.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                "preprocessing": {
                    k: {
                        "material_cells": [
                            c for c, x in v["cells"].items() if x["material"]
                        ],
                        "headline_unchanged": v["headline_unchanged"],
                    }
                    for k, v in s2.items()
                },
                "finite shots": {
                    k: v["category"] for k, v in out["finite_shots"].items()
                },
                "finite shots explain the noisy gap": out[
                    "finite_shots_explain_noisy_gap"
                ],
                "noise scale": {k: v["category"] for k, v in s4.items()},
                "closest noise scale": out["noise_scale"]["closest_scale"],
                "missing intercept": out["missing_intercept"],
                "classical": gc,
            },
            indent=1,
        )
    )
    print(f"wrote {os.path.relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
