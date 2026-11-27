"""Extract the paper's Appendix C tables (C1-C12) from the cached arXiv LaTeX source.

Writes ``results/data/paper_tables.json``: one record per cell with the paper's MAE,
RMSE and R2, keyed like this reproduction's ``metrics.json`` cells
(architecture, configuration label, reservoir type, readout type).

Usage (from the repository root)::

    python papers/distributed_qrc/utils/extract_paper_tables.py \\
        data/distributed_qrc/sources/arxiv_source_v1/tqe_tran.tex
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]

# Appendix C: per architecture, the classical-ridge table, then the quantum
# ridge on the ideal simulator, then on the noisy simulator.
TABLES = {
    "tabC1": ("srsr", "linear"),
    "tabC2": ("srsr", "quantum_kernel_ideal"),
    "tabC3": ("srsr", "quantum_kernel_noisy"),
    "tabC4": ("mrsr", "linear"),
    "tabC5": ("mrsr", "quantum_kernel_ideal"),
    "tabC6": ("mrsr", "quantum_kernel_noisy"),
    "tabC7": ("srmr", "linear"),
    "tabC8": ("srmr", "quantum_kernel_ideal"),
    "tabC9": ("srmr", "quantum_kernel_noisy"),
    "tabC10": ("mrmr", "linear"),
    "tabC11": ("mrmr", "quantum_kernel_ideal"),
    "tabC12": ("mrmr", "quantum_kernel_noisy"),
}
RESERVOIRS = ("classical", "quantum_ideal", "quantum_noisy")
NUMBER = re.compile(r"^-?\d*\.\d+$|^-?\d+$")


def _clean(cell: str) -> str:
    return re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell).strip()


def table_rows(tex: str, label: str) -> list[list[str]]:
    start = tex.index(f"\\label{{table:{label}}}")
    end = tex.index("\\end{tabular}", start)
    body = tex[start:end]
    rows = []
    for line in body.split("\\\\"):
        cells = [_clean(c) for c in line.replace("\\hline", "").split("&")]
        numeric = [c for c in cells if NUMBER.match(c)]
        if len(numeric) >= 9:
            rows.append(cells)
    return rows


def configuration_label(architecture: str, cells: list[str]) -> str:
    first = cells[0]
    if architecture == "srsr":
        return first
    if architecture in ("mrsr", "mrmr"):
        return first.replace(" ", "")
    # SRMR: neurons, then "m" (C7) or "m x q" (C8, C9)
    instances = cells[1].replace(" ", "").split("x")[0]
    return f"{first}/{instances}"


def extract(tex: str) -> list[dict]:
    records = []
    for label, (architecture, readout) in TABLES.items():
        for cells in table_rows(tex, label):
            values = [float(c) for c in cells if NUMBER.match(c)][-9:]
            config = configuration_label(architecture, cells)
            for index, reservoir in enumerate(RESERVOIRS):
                mae, rmse, r2 = values[3 * index : 3 * index + 3]
                records.append(
                    {
                        "table": label.replace("tab", ""),
                        "architecture": architecture,
                        "configuration": config,
                        "reservoir_type": reservoir,
                        "readout_type": readout,
                        "mae": mae,
                        "rmse": rmse,
                        "r2": r2,
                    }
                )
    return records


def main() -> int:
    source = Path(sys.argv[1])
    records = extract(source.read_text(encoding="utf-8"))
    out = PAPER_DIR / "results" / "data" / "paper_tables.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"source": str(source), "cells": records}, indent=1),
        encoding="utf-8",
    )
    counts = {}
    for record in records:
        counts[record["architecture"]] = counts.get(record["architecture"], 0) + 1
    print(f"wrote {len(records)} cells to {out}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
