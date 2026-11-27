"""Compare a completed reproduction sweep with the paper's Appendix C tables, cell by cell.

Reads ``<sweep_dir>/summary.json`` (written by ``utils/run_sweep.py`` from the
runs' ``metrics.json``) and ``results/data/paper_tables.json`` (written by
``utils/extract_paper_tables.py``). Writes ``results/data/<architecture>_vs_paper.csv``
and prints the table.

Columns: our test MAE / RMSE mean and sample std over seeds, the paper value,
the difference in units of our seed std (``z``) and the relative difference.
The paper reports one value per cell (no seeds or spread), so ``z`` measures how far the paper value
lies from our seed distribution, not a formal test.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/compare_with_paper.py papers/distributed_qrc/outdir/sweep_E1
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
RESERVOIRS = ("classical", "quantum_ideal", "quantum_noisy")
READOUTS = ("linear", "quantum_kernel_ideal", "quantum_kernel_noisy")


def compare(sweep_dir: Path) -> list[dict]:
    summary = json.loads((sweep_dir / "summary.json").read_text())
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_dir} is {status['status']}, not COMPLETED")
    paper = json.loads(
        (PAPER_DIR / "results" / "data" / "paper_tables.json").read_text()
    )["cells"]
    metrics = json.loads((sweep_dir / status["runs"][0]["metrics_path"]).read_text())
    architecture = metrics["architecture"]
    index = {
        (p["configuration"], p["reservoir_type"], p["readout_type"]): p
        for p in paper
        if p["architecture"] == architecture
    }
    rows = []
    for key, cell in summary["cells"]["default"].items():
        configuration, reservoir, readout = key.split("|")
        reference = index[(configuration, reservoir, readout)]
        row = {
            "sweep": summary["sweep_id"],
            "architecture": architecture,
            "configuration": configuration,
            "reservoir": reservoir,
            "readout": readout,
            "paper_table": reference["table"],
            "n_seeds": cell["n_seeds"],
        }
        for metric in ("mae", "rmse"):
            mean, std = cell[f"test.{metric}"]["mean"], cell[f"test.{metric}"]["std"]
            value = reference[metric]
            row.update(
                {
                    f"{metric}_ours": mean,
                    f"{metric}_ours_std": std,
                    f"{metric}_paper": value,
                    f"{metric}_z": (value - mean) / std if std > 0 else float("nan"),
                    f"{metric}_rel": (mean - value) / value,
                }
            )
        row["mae_mw_ours"] = cell["test_mw.mae"]["mean"]
        rows.append(row)
    order = {name: i for i, name in enumerate(RESERVOIRS + READOUTS)}
    rows.sort(
        key=lambda r: (r["configuration"], order[r["readout"]], order[r["reservoir"]])
    )
    return rows


def main() -> int:
    sweep_dir = Path(sys.argv[1])
    rows = compare(sweep_dir)
    out = PAPER_DIR / "results" / "data" / f"{rows[0]['architecture']}_vs_paper.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{'config':7s} {'reservoir':13s} {'readout':20s} {'MAE ours (mean±std)':>22s} {'paper':>7s} {'rel':>7s} {'z':>6s}"
    )
    for r in rows:
        print(
            f"{r['configuration']:7s} {r['reservoir']:13s} {r['readout']:20s} "
            f"{r['mae_ours']:.4f} ± {r['mae_ours_std']:.4f}    {r['mae_paper']:.4f} {100 * r['mae_rel']:+6.0f}% {r['mae_z']:+6.1f}"
        )
    print(f"wrote {os.path.relpath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
