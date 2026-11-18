"""Compare the MerLin adaptation with the gate-based reproduction.

Applies the rules declared before the runs (tolerance below).
Per configuration, the test MAE (normalized) is averaged over seeds 0-2:

* (a) photonic|photonic_kernel vs R2 quantum_ideal|quantum_kernel_ideal;
* (b) classical|photonic_kernel and quantum_ideal|photonic_kernel vs the same
  reservoir with R2 quantum_kernel_ideal (kernel-only effect);
* (c) photonic|linear vs R2 quantum_ideal|linear (reservoir-only effect);
* (d) paper-rule headline (max test R2 per configuration; MAE and RMSE
  reduction vs classical|linear), using the photonic analogues of the ideal
  quantum cells (classical|photonic_kernel, photonic|linear,
  photonic|photonic_kernel). The same rule is applied to the R2 ideal cells
  (classical|quantum_kernel_ideal, quantum_ideal|linear,
  quantum_ideal|quantum_kernel_ideal) as a like-for-like reference.

Tolerance: a comparison "preserves" the original behaviour in a configuration
if |delta mean| <= max(10 % of the R2 mean, 2 x pooled seed std), where pooled
std = sqrt((s_merlin^2 + s_r2^2) / 2). An architecture preserves it if this holds
in >= 6 of its 8 configurations; otherwise the direction is reported as
"better" (median delta < 0, lower MAE) or "worse".

The in-run anchors classical|linear and quantum_ideal|linear must equal the R2
values of the same seed (same code path); their max |difference| is reported.

Writes ``results/data/merlin_comparison.json``.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/merlin_comparison.py papers/distributed_qrc/outdir
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
ARCHS = {"srsr": "E1", "mrsr": "E2", "srmr": "E3", "mrmr": "E4"}
COMPARISONS = {
    "a_full_quantum": (
        ("photonic", "photonic_kernel"),
        ("quantum_ideal", "quantum_kernel_ideal"),
    ),
    "b_kernel_on_classical": (
        ("classical", "photonic_kernel"),
        ("classical", "quantum_kernel_ideal"),
    ),
    "b_kernel_on_quantum": (
        ("quantum_ideal", "photonic_kernel"),
        ("quantum_ideal", "quantum_kernel_ideal"),
    ),
    "c_reservoir_linear": (("photonic", "linear"), ("quantum_ideal", "linear")),
}
ANCHORS = (("classical", "linear"), ("quantum_ideal", "linear"))
BASELINE = ("classical", "linear")
HEADLINE_CELLS = {
    "merlin": [
        ("classical", "photonic_kernel"),
        ("photonic", "linear"),
        ("photonic", "photonic_kernel"),
    ],
    "r2_ideal": [
        ("classical", "quantum_kernel_ideal"),
        ("quantum_ideal", "linear"),
        ("quantum_ideal", "quantum_kernel_ideal"),
    ],
}


def load_sweep(sweep_dir: Path) -> dict[int, dict]:
    """{seed: {(config, reservoir, readout): test metrics}} for a COMPLETED sweep."""

    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_dir} is {status['status']}")
    out = {}
    for run in status["runs"]:
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        out[run["seed"]] = {
            (c["configuration"], c["reservoir_type"], c["readout_type"]): c["metrics"][
                "test"
            ]
            for c in metrics["cells"]
        }
    return out


def headline(runs: dict[int, dict], cells: list[tuple[str, str]]) -> dict:
    per_seed = {"mae": [], "rmse": []}
    best_cells = []
    for seed in sorted(runs):
        r = runs[seed]
        configs = list(dict.fromkeys(k[0] for k in r))
        red = {"mae": [], "rmse": []}
        for config in configs:
            best = max(cells, key=lambda c: r[(config, *c)]["r2"])
            best_cells.append("|".join(best))
            for m in red:
                red[m].append(1 - r[(config, *best)][m] / r[(config, *BASELINE)][m])
        for m in red:
            per_seed[m].append((np.mean(red[m]), np.max(red[m])))
    out = {}
    for m, v in per_seed.items():
        v = np.array(v)
        out[m] = {
            "average_mean": float(v[:, 0].mean()),
            "average_std": float(v[:, 0].std(ddof=1)),
            "peak_mean": float(v[:, 1].mean()),
            "peak_std": float(v[:, 1].std(ddof=1)),
        }
    out["best_cell_counts"] = {c: best_cells.count(c) for c in sorted(set(best_cells))}
    return out


def main() -> int:
    outdir = Path(sys.argv[1])
    result: dict = {
        "tolerance": "max(0.10 * R2 mean, 2 * pooled std); >= 6/8 configurations"
    }
    for arch, exp in ARCHS.items():
        merlin_dir, r2_dir = outdir / f"sweep_{exp}-MERLIN", outdir / f"sweep_{exp}-R2"
        try:
            merlin, r2 = load_sweep(merlin_dir), load_sweep(r2_dir)
        except (FileNotFoundError, ValueError) as error:
            result[arch] = {"status": f"MISSING: {error}"}
            continue
        seeds = sorted(merlin)
        if seeds != sorted(r2):
            raise ValueError(f"{arch}: seeds differ {seeds} vs {sorted(r2)}")
        configs = list(dict.fromkeys(k[0] for k in merlin[seeds[0]]))
        anchor_diff = max(
            abs(merlin[s][(c, *a)][m] - r2[s][(c, *a)][m])
            for s in seeds
            for c in configs
            for a in ANCHORS
            for m in ("mae", "rmse", "r2")
        )
        arch_out: dict = {
            "status": "COMPLETED",
            "seeds": seeds,
            "anchor_max_abs_diff": anchor_diff,
        }
        for name, (new, ref) in COMPARISONS.items():
            rows = {}
            for c in configs:
                x = np.array([merlin[s][(c, *new)]["mae"] for s in seeds])
                y = np.array([r2[s][(c, *ref)]["mae"] for s in seeds])
                delta = x.mean() - y.mean()
                pooled = float(np.sqrt((x.std(ddof=1) ** 2 + y.std(ddof=1) ** 2) / 2))
                tol = max(0.10 * y.mean(), 2 * pooled)
                rows[c] = {
                    "merlin_mean": float(x.mean()),
                    "merlin_std": float(x.std(ddof=1)),
                    "r2_mean": float(y.mean()),
                    "r2_std": float(y.std(ddof=1)),
                    "delta": float(delta),
                    "relative_delta": float(delta / y.mean()),
                    "tolerance": float(tol),
                    "preserved": bool(abs(delta) <= tol),
                }
            n_ok = sum(r["preserved"] for r in rows.values())
            median_delta = float(np.median([r["delta"] for r in rows.values()]))
            verdict = (
                "preserves"
                if n_ok >= 6
                else ("better" if median_delta < 0 else "worse")
            )
            arch_out[name] = {
                "merlin_cell": "|".join(new),
                "r2_cell": "|".join(ref),
                "configurations": rows,
                "preserved_count": n_ok,
                "median_relative_delta": float(
                    np.median([r["relative_delta"] for r in rows.values()])
                ),
                "verdict": verdict,
            }
        arch_out["d_headline"] = {
            "merlin": headline(merlin, HEADLINE_CELLS["merlin"]),
            "r2_ideal": headline(r2, HEADLINE_CELLS["r2_ideal"]),
        }
        result[arch] = arch_out

    path = PAPER_DIR / "results" / "data" / "merlin_comparison.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for arch in ARCHS:
        a = result[arch]
        if a["status"] != "COMPLETED":
            print(arch, a["status"])
            continue
        print(f"{arch}: anchors max|diff| {a['anchor_max_abs_diff']:.1e}")
        for name in COMPARISONS:
            v = a[name]
            print(
                f"  {name:24s} {v['verdict']:9s} preserved {v['preserved_count']}/8, "
                f"median rel. delta {v['median_relative_delta']:+.1%}"
            )
        for k, h in a["d_headline"].items():
            print(
                f"  d {k:9s} MAE avg {h['mae']['average_mean']:.1%}±{h['mae']['average_std']:.1%} "
                f"peak {h['mae']['peak_mean']:.1%}; RMSE avg {h['rmse']['average_mean']:.1%} "
                f"peak {h['rmse']['peak_mean']:.1%}; best {h['best_cell_counts']}"
            )
    print(f"wrote {os.path.relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
