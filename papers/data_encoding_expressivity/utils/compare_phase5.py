"""Generate the Phase 5 comparison table from a run manifest (frozen plan in LOG.md).

Usage, from the paper directory::

    python utils/compare_phase5.py --manifest results/phase5_manifest.json --output results/phase5_comparison

For every manifest run, the script checks that the run is COMPLETED and
reads its ``metrics.json``. It rejects duplicate (case, seed) results and
non-finite values, and compares the observed seeds with the expected ones.
It then applies the frozen rules:

- success: final MSE - B1 <= 1e-4;
- near-perfect: final MSE <= 1e-3;
- median, mean and sample std over seeds, at full precision.

Wall-clock per run comes from the ``run_status.json`` timestamps, and covers
all cases of that run. Writes ``<output>.json`` and ``<output>.md``, with one
table per comparison group.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import time
from pathlib import Path

import numpy as np

NEAR_PERFECT = 1e-3
GROUP_TITLES = {
    "G1": "G1 — Fig. 4 published target (s = 1), protocols P0–P3",
    "G2": "G2 — Fig. 4 published target (s = 1), paper protocol P0: degree sweep",
    "G3": "G3 — Fig. 3 targets, paper protocol P0",
    "G4": "G4 — Fig. 4 caption target (s = 2), paper protocol P0 (not comparable with G1/G2)",
}


def _timestamp(text: str) -> float:
    return calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ"))


def _as_recorded_at_run_time(kind: str, model: dict) -> dict:
    """Make the model config of runs that predate the Phase 4 amendment explicit.

    Before the amendment, the code had no ``readout``, ``phase_screens`` or
    ``mode_scales`` keys. Every such run used the fixed observable (``<Z_0>``,
    or the photonic mode-0 threshold readout), the bare mesh and unit scales.
    """
    model = dict(model)
    if kind == "quantum" and "readout" not in model:
        model["readout"] = "z0"
    if kind == "photonic":
        if "readout" not in model:
            model["readout"] = "threshold_mode0"
        if "phase_screens" not in model:
            model["phase_screens"] = False
        if "mode_scales" not in model:
            model["mode_scales"] = [1.0] * len(model["encoded_modes"])
    return model


def _system_fields(kind: str, model: dict) -> dict:
    """Describe the compared system: modality-specific capacity and readout fields."""
    model = _as_recorded_at_run_time(kind, model)
    if kind == "fourier_classical":
        return {
            "system": "B2 classical Fourier model",
            "readout": "linear (coefficients)",
            "hardware": "—",
        }
    if kind == "quantum":
        layout = f"qubit {model['layout']}"
        if model["layout"] == "parallel":
            layout += f" ({model['ansatz']} × {model['ansatz_layers']})"
        return {
            "system": layout,
            "readout": model["readout"],
            "hardware": "statevector (torch)",
        }
    screens = "universal mesh" if model["phase_screens"] else "bare mesh"
    return {
        "system": f"photonic n={model['n_photons']}, m={model['n_modes']}, L={model['encoding_layers']}, "
        f"scales {model['mode_scales']}, {screens}",
        "readout": model["readout"],
        "hardware": f"MerLin {model['computation_space'].upper()}, threshold (emulated), no shots",
    }


def collect(manifest: dict, base: Path) -> tuple[list[dict], list[str]]:
    """Return one row per (run, case, model kind) and the integrity log."""
    rows, checks = [], []
    for entry in manifest["runs"]:
        run_dir = base / entry["run"]
        status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
        if status["status"] != "COMPLETED":
            raise ValueError(f"{run_dir} is not COMPLETED")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if metrics["experiment"] != "fit":
            raise ValueError(f"{run_dir} is not a fit run")
        wall_clock = _timestamp(status["completed_at"]) - _timestamp(
            status["started_at"]
        )
        expected_seeds = list(metrics["seeds"])
        selected = entry["cases"]
        seen = set()
        for run in metrics["runs"]:
            key = (run["case"], run["model_kind"], run["seed"])
            if key in seen:
                raise ValueError(f"duplicate result {key} in {run_dir}")
            seen.add(key)
            if not math.isfinite(run["final_mse"]):
                raise ValueError(f"non-finite MSE {key} in {run_dir}")
        cases = {case["name"]: case for case in metrics["cases"]}
        names = selected if selected is not None else list(cases)
        missing = [name for name in names if name not in cases]
        if missing:
            raise ValueError(f"cases {missing} not in {run_dir}")
        for name in names:
            case = cases[name]
            for kind in metrics["model_kinds"]:
                case_runs = [
                    r
                    for r in metrics["runs"]
                    if r["case"] == name and r["model_kind"] == kind
                ]
                observed = sorted(r["seed"] for r in case_runs)
                if observed != sorted(expected_seeds):
                    checks.append(
                        f"INCOMPLETE {entry['run']} {name}: seeds {observed} vs {expected_seeds}"
                    )
                finals = np.array([r["final_mse"] for r in case_runs])
                rows.append(
                    {
                        "group": entry["group"],
                        "protocol": entry["protocol"],
                        "run": entry["run"],
                        "case": name,
                        "role": entry["role"],
                        "provenance": entry["provenance"],
                        "scope": entry["scope"],
                        "modality": entry["modality"],
                        **_system_fields(kind, case["model"]),
                        "degree": max(case["b1_frequencies"]),
                        "n_parameters": case_runs[0]["n_parameters"],
                        "b1_mse": case["b1_mse"],
                        "successful_runs": len(case_runs),
                        "expected_runs": len(expected_seeds),
                        "success_count": int(sum(r["success"] for r in case_runs)),
                        "near_perfect_count": int((finals <= NEAR_PERFECT).sum()),
                        "median_mse": float(np.median(finals)),
                        "mean_mse": float(finals.mean()),
                        "std_mse": float(finals.std(ddof=1))
                        if len(finals) > 1
                        else 0.0,
                        "min_mse": float(finals.min()),
                        "max_mse": float(finals.max()),
                        "run_wall_clock_s": wall_clock,
                        "training": metrics["training"],
                    }
                )
        checks.append(
            f"OK {entry['run']}: {len(names)} cases × {len(metrics['model_kinds'])} kinds, seeds {expected_seeds}"
        )
    return rows, checks


def to_markdown(rows: list[dict], manifest: dict, checks: list[str]) -> str:
    lines = [
        "# Phase 5 comparison (generated by `utils/compare_phase5.py`; do not edit by hand)",
        "",
        "Success = final MSE − B1 ≤ 1e-4; near-perfect = final MSE ≤ 1e-3; median and mean ± std over seeds.",
        "Rows within one group and one protocol are commensurable (same target, grid, optimiser, seeds; matched frequency set).",
        "Rows from different protocols are shown together but are not ranked against each other.",
        "Photonic rows are a MerLin adaptation (simulation), not part of the reproduction.",
        "",
        "## Paper-reported values",
        "",
        "| Group | Claim | Paper value |",
        "|---|---|---|",
    ]
    for item in manifest.get("paper_reported", []):
        lines.append(f"| {item['group']} | {item['claim']} | {item['value']} |")
    titles = {**GROUP_TITLES, **manifest.get("group_titles", {})}
    for group in dict.fromkeys(r["group"] for r in rows):
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            continue
        lines += [
            "",
            f"## {titles[group]}",
            "",
            "| Protocol | System | Case | Role / provenance | Scope | Degree | Params | Readout | Hardware | Runs | B1 | Success | ≤ 1e-3 | Median MSE | Mean ± std | Run wall-clock |",
            "|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in sorted(
            group_rows,
            key=lambda r: (r["protocol"], r["degree"], r["modality"], r["case"]),
        ):
            lines.append(
                f"| {r['protocol']} | {r['system']} | {r['case']} | {r['role']} ({r['provenance']}) | {r['scope']} | "
                f"{r['degree']} | {r['n_parameters']} | {r['readout']} | {r['hardware']} | "
                f"{r['successful_runs']}/{r['expected_runs']} | {r['b1_mse']:.2e} | {r['success_count']}/{r['successful_runs']} | "
                f"{r['near_perfect_count']}/{r['successful_runs']} | {r['median_mse']:.2e} | "
                f"{r['mean_mse']:.2e} ± {r['std_mse']:.1e} | {r['run_wall_clock_s']:.0f} s |"
            )
    lines += [
        "",
        "Wall-clock is per run (all cases of that run), on the local CPU (Apple M5, 10 cores), one run at a time.",
        "",
        "## Integrity checks",
        "",
        *[f"- {line}" for line in checks],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base = Path.cwd()
    rows, checks = collect(manifest, base)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "rows": rows,
                "checks": checks,
                "paper_reported": manifest.get("paper_reported", []),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(
        to_markdown(rows, manifest, checks), encoding="utf-8"
    )
    incomplete = [c for c in checks if c.startswith("INCOMPLETE")]
    print(
        f"{len(rows)} rows from {len(manifest['runs'])} runs; incomplete: {len(incomplete)}"
    )


if __name__ == "__main__":
    main()
