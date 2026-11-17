"""Render the moons_assignment_rules sweep ledger and result tables as Markdown.

Usage::

    python utils/render_moons_sweep_tables.py outdir/sweep_moons_assignment_rules

Reads only ``sweep_status.json`` and ``summary.csv`` written by
``utils/run_moons_assignment_sweep.py``; checks coverage against the plan
(21 candidates x 4 models) and fails if it does not match.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

N_CANDIDATES, N_MODELS = 21, 4
QUBIT, QUTRIT = "B2_qubit_experimental_fig4a", "B3_qutrit_experimental_fig4b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    sweep_dir = parser.parse_args().sweep_dir
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    rows = list(csv.DictReader((sweep_dir / "summary.csv").open()))
    completed = [run for run in status["runs"] if run["status"] == "COMPLETED"]
    if len(completed) != N_CANDIDATES or len(rows) != N_CANDIDATES * N_MODELS:
        raise ValueError(f"coverage {len(completed)} runs / {len(rows)} rows")

    print("| Candidate (rank_rule, noise) | Attempts | Status | Run dir |")
    print("|---|---:|---|---|")
    for run in status["runs"]:
        candidate = run["candidate"]
        print(f"| {candidate['rank_rule']}, {candidate['noise']} | "
              f"{len(run['attempts'])} | {run['status']} | outdir/{run['run_id']} |")  # fmt: skip

    table: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for row in rows:
        table.setdefault((row["noise"], row["rank_rule"]), {})[row["model"]] = (
            float(row["mean_test_accuracy"]),
            float(row["std_over_variants"]),
        )
    print()
    print("| rank_rule | noise | Qubit exp. | Qutrit exp. | Qutrit - qubit |")
    print("|---|---|---|---|---|")
    qutrit_better = 0
    for (noise, rule), models in table.items():
        qubit, qutrit = models[QUBIT], models[QUTRIT]
        qutrit_better += qutrit[0] > qubit[0]
        print(f"| {rule} | {noise} | {qubit[0]:.3f} ± {qubit[1]:.3f} | "
              f"{qutrit[0]:.3f} ± {qutrit[1]:.3f} | {qutrit[0] - qubit[0]:+.3f} |")  # fmt: skip
    print(f"\nCOMPLETE SWEEP: {len(completed)}/{N_CANDIDATES} runs, {len(rows)} rows. "
          f"Qutrit above qubit in {qutrit_better} of {len(table)} candidates.")  # fmt: skip


if __name__ == "__main__":
    main()
