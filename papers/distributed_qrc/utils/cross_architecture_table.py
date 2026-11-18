"""Cross-architecture comparison at equal total neuron count (paper Table VIII).

For each total neuron count (10, 15, 20, 25, 30, 40, 50, 60), the paper reports
the best model over all four architectures and observes that it always involves
some distribution (MRSR, SRMR or MRMR). The paper's selection is reproduced as
"highest test R2 among the quantum or hybrid cells" (the rule that reproduces
its headline percentages, see ``utils/headline_statistics.py``).

For our sweeps the table reports, per total neuron count:

* the best cell by mean test R2 over seeds, and its architecture;
* the best SRSR (single-node) cell, and the gap to the overall best in units of
  the pooled seed std, so that "distribution wins" can be told apart from
  seed noise.

Writes ``results/data/cross_architecture.json``.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/cross_architecture_table.py papers/distributed_qrc/outdir/sweep_E1 [...E2 E3 E4]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
BASELINE = ("classical", "linear")


def _total(architecture: str, label: str) -> int:
    if "x" in label:
        count, size = label.split("x")
        return int(count) * int(size)
    return int(label.split("/")[0])


def our_cells(sweep_dirs: list[Path]) -> list[dict]:
    rows = []
    for sweep_dir in sweep_dirs:
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        if status["status"] != "COMPLETED":
            raise ValueError(f"{sweep_dir} is {status['status']}")
        runs = [
            json.loads((sweep_dir / r["metrics_path"]).read_text())
            for r in status["runs"]
        ]
        architecture = runs[0]["architecture"]
        for index, cell in enumerate(runs[0]["cells"]):
            key = (cell["reservoir_type"], cell["readout_type"])
            if key == BASELINE:
                continue
            per_seed = [run["cells"][index]["metrics"]["test"] for run in runs]
            rows.append(
                {
                    "architecture": architecture,
                    "configuration": cell["configuration"],
                    "total": _total(architecture, cell["configuration"]),
                    "cell": "|".join(key),
                    "r2": [m["r2"] for m in per_seed],
                    "mae": [m["mae"] for m in per_seed],
                    "rmse": [m["rmse"] for m in per_seed],
                }
            )
    return rows


def paper_cells() -> list[dict]:
    cells = json.loads(
        (PAPER_DIR / "results" / "data" / "paper_tables.json").read_text()
    )["cells"]
    return [
        {
            "architecture": c["architecture"],
            "configuration": c["configuration"],
            "total": _total(c["architecture"], c["configuration"]),
            "cell": f"{c['reservoir_type']}|{c['readout_type']}",
            "r2": [c["r2"]],
            "mae": [c["mae"]],
            "rmse": [c["rmse"]],
        }
        for c in cells
        if (c["reservoir_type"], c["readout_type"]) != BASELINE
    ]


def best_per_total(rows: list[dict]) -> dict:
    table = {}
    for total in sorted({r["total"] for r in rows}):
        group = [r for r in rows if r["total"] == total]
        best = max(group, key=lambda r: np.mean(r["r2"]))
        single = [r for r in group if r["architecture"] == "srsr"]
        entry = {
            "best_architecture": best["architecture"],
            "best_configuration": best["configuration"],
            "best_cell": best["cell"],
            "best_mae_mean": float(np.mean(best["mae"])),
            "best_r2_mean": float(np.mean(best["r2"])),
            "architectures_available": sorted({r["architecture"] for r in group}),
        }
        if single and len(best["mae"]) > 1:
            srsr = max(single, key=lambda r: np.mean(r["r2"]))
            pooled = np.sqrt(
                (np.var(best["mae"], ddof=1) + np.var(srsr["mae"], ddof=1)) / 2
            )
            entry.update(
                {
                    "best_srsr_cell": srsr["cell"],
                    "best_srsr_mae_mean": float(np.mean(srsr["mae"])),
                    "mae_gap_srsr_minus_best": float(
                        np.mean(srsr["mae"]) - np.mean(best["mae"])
                    ),
                    "mae_gap_in_pooled_std": float(
                        (np.mean(srsr["mae"]) - np.mean(best["mae"])) / pooled
                    )
                    if pooled > 0
                    else float("nan"),
                }
            )
        table[str(total)] = entry
    return table


def main() -> int:
    sweep_dirs = [Path(a) for a in sys.argv[1:]]
    result = {
        "paper": best_per_total(paper_cells()),
        "ours": best_per_total(our_cells(sweep_dirs)),
        "sweeps": [str(s) for s in sweep_dirs],
    }
    out = PAPER_DIR / "results" / "data" / "cross_architecture.json"
    result = {"paper_reference": "Table VIII (best model per total neurons)", **result}
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(
        f"{'N':>3s} | {'paper best':28s} | {'ours best (mean over seeds)':40s} | {'best SRSR MAE':>13s} {'gap/std':>8s}"
    )
    for total, ours in result["ours"].items():
        paper = result["paper"].get(total, {})
        print(
            f"{total:>3s} | {paper.get('best_architecture', '-'):5s} {paper.get('best_cell', '-'):22s} | "
            f"{ours['best_architecture']:5s} {ours['best_configuration']:6s} {ours['best_cell']:28s} | "
            f"{ours.get('best_srsr_mae_mean', float('nan')):.5f} {ours.get('mae_gap_in_pooled_std', float('nan')):+8.2f}"
        )
    print(f"wrote {os.path.relpath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
