"""Where does the paper's value of the paper baseline fall in our 20-seed distribution?

Reads a completed ``GAP-CLASSICAL-<ARCH>`` sweep (classical reservoir + linear
ridge over the paper grid, many seeds) and ``results/data/paper_tables.json``. For
each configuration it reports our test-MAE distribution over seeds (min,
median, max) and the empirical rank of the paper value (fraction of our seeds
with a lower MAE). Writes ``results/data/classical_seed_spread_<architecture>.json``.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/classical_seed_spread.py papers/distributed_qrc/outdir/sweep_GAP-CLASSICAL-SRSR
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    sweep_dir = Path(sys.argv[1])
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_dir} is {status['status']}")
    runs = [
        json.loads((sweep_dir / r["metrics_path"]).read_text()) for r in status["runs"]
    ]
    architecture = runs[0]["architecture"]
    paper = {
        c["configuration"]: c["mae"]
        for c in json.loads(
            (PAPER_DIR / "results" / "data" / "paper_tables.json").read_text()
        )["cells"]
        if c["architecture"] == architecture
        and c["reservoir_type"] == "classical"
        and c["readout_type"] == "linear"
    }
    rows = {}
    for index, cell in enumerate(runs[0]["cells"]):
        values = np.array(
            [run["cells"][index]["metrics"]["test"]["mae"] for run in runs]
        )
        reference = paper[cell["configuration"]]
        rows[cell["configuration"]] = {
            "n_seeds": int(values.size),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "paper": reference,
            "paper_rank": float((values < reference).mean()),
            "paper_within_range": bool(values.min() <= reference <= values.max()),
        }
    out = PAPER_DIR / "results" / "data" / f"classical_seed_spread_{architecture}.json"
    out.write_text(
        json.dumps({"architecture": architecture, "configurations": rows}, indent=1),
        encoding="utf-8",
    )
    print(
        f"{architecture}: classical + linear test MAE over {len(runs)} seeds vs paper"
    )
    print(
        f"{'config':7s} {'min':>7s} {'median':>7s} {'max':>7s} {'paper':>7s} {'rank':>5s} in-range"
    )
    for config, r in rows.items():
        print(
            f"{config:7s} {r['min']:.4f} {r['median']:.4f} {r['max']:.4f} {r['paper']:.4f} {r['paper_rank']:5.2f} {r['paper_within_range']}"
        )
    inside = sum(r["paper_within_range"] for r in rows.values())
    print(
        f"paper value within our [min, max]: {inside}/{len(rows)} configurations; wrote {os.path.relpath(out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
