#!/usr/bin/env python3
"""Write the hardware-aware report of the MerLin adaptation sweeps (Phase 4).

Reads each sweep's ``sweep_status.json`` and every completed run's
``config_snapshot.json`` and ``metrics.json``; derives the photonic fields from
the configuration actually executed (input state from ``spread_input_state``,
computation space from the readout). Writes
``results/merlin_hardware_report.{md,json}``.

Example (from the paper folder)::

    python utils/merlin_hardware_report.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
for import_root in (PAPER_DIR.parents[1], PAPER_DIR):
    sys.path.insert(0, str(import_root))

from lib.qnn_merlin import spread_input_state  # noqa: E402

SWEEPS = (
    "m1_fig2a",
    "m2_fig2b",
    "m4_scaled",
    "m5_fig3",
    "m6_table1_tf",
    "m6_table1_ff",
    "m6_table1_ffs",
)
DETECTION = {
    "threshold_clicks": ("FOCK (simulated)", "threshold, no post-selection", "none"),
    "unbunched_clicks": (
        "UNBUNCHED",
        "threshold, collision-free post-selection",
        "collision-free outcomes, renormalised",
    ),
}
MEASUREMENT = "MeasurementStrategy.probs → fixed click observable (#clicks even modes − #clicks odd modes)"
WHERE = {
    "m6_table1_tf": "remote gpu-reproduced-papers-1 (NVIDIA L4), 4 workers",
    "m6_table1_ff": "remote gpu-reproduced-papers-1 (NVIDIA L4), 4 workers",
    "m6_table1_ffs": "remote gpu-reproduced-papers-1 (NVIDIA L4), 4 workers",
}


def main() -> int:
    rows = []
    for sweep in SWEEPS:
        sweep_dir = PAPER_DIR / "outdir" / "sweeps" / sweep
        status = json.loads(
            (sweep_dir / "sweep_status.json").read_text(encoding="utf-8")
        )
        by_candidate: dict[str, list[tuple[int, dict, dict]]] = {}
        for entry in status["runs"]:
            if entry["status"] != "COMPLETED":
                continue
            config = json.loads(
                (sweep_dir / entry["config_path"]).read_text(encoding="utf-8")
            )
            metrics = json.loads(
                (sweep_dir / entry["metrics_path"]).read_text(encoding="utf-8")
            )
            by_candidate.setdefault(entry["candidate"], []).append(
                (entry["seed"], config, metrics)
            )
        for candidate, runs in by_candidate.items():
            model = runs[0][1]["model"]
            space, detector, postselection = DETECTION[model["readout"]]
            encoding_modes = model["encoding_modes_per_input"] * (
                3 if model["layout"] == "ns_parallel" else 1
            )
            blocks = 2 if model["layout"] == "ns_parallel" else 1
            seconds = [m["training_seconds"] for _, _, m in runs]
            rows.append(
                {
                    "sweep": sweep,
                    "candidate": candidate,
                    "computation_space": space,
                    "detector_model": detector,
                    "photons": model["n_photons"],
                    "modes": model["n_modes"],
                    "input_state": spread_input_state(
                        model["n_modes"], model["n_photons"]
                    ),
                    "encoding": f"phase shifters on modes 0–{encoding_modes - 1}, φ = γ·w·x (γ: {model['feature_map']}; w: {model['frequency_mode']}, init {model['frequency_init']}); {blocks} encoding block(s)",
                    "measurement_strategy": MEASUREMENT,
                    "postselection": postselection,
                    "simulator_or_qpu": "MerLin analytic simulator (merlin 0.4.1), "
                    + WHERE.get(sweep, "local CPU (Apple M3), 6 workers"),
                    "shots": "n/a (exact probabilities, shots = 0)",
                    "gradients": "exact (autograd through the simulator)",
                    "wall_clock_training_s_mean": statistics.fmean(seconds),
                    "wall_clock_training_s_max": max(seconds),
                    "seeds": sorted(seed for seed, _, _ in runs),
                }
            )
    out = PAPER_DIR / "results"
    (out / "merlin_hardware_report.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# MerLin adaptation — hardware-aware report (simulation only; not hardware validation)",
        "",
        "Common to every row: measurement strategy = "
        + MEASUREMENT
        + "; simulator = MerLin analytic (merlin 0.4.1, perceval 1.3.1); shots = n/a (exact probabilities); gradients = exact autograd; lossless, noiseless, indistinguishable photons.",
        "",
        "| Sweep | Candidate | Computation space | Detector | Photons | Modes | Input state | Encoding | Post-selection | Where | Wall-clock (mean / max s) | Seeds |",
        "|---|---|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['sweep']} | {r['candidate']} | {r['computation_space']} | {r['detector_model']} | {r['photons']} | {r['modes']} | "
            f"{r['input_state']} | {r['encoding']} | {r['postselection']} | {r['simulator_or_qpu'].split(', ', 1)[1]} | "
            f"{r['wall_clock_training_s_mean']:.0f} / {r['wall_clock_training_s_max']:.0f} | {r['seeds'][0]}–{r['seeds'][-1]} ({len(r['seeds'])}) |"
        )
    (out / "merlin_hardware_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"{len(rows)} configurations reported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
