"""Export per-epoch test-accuracy curves of validated sweeps to ``results/curves.json``.

Reads ``metrics.json`` of every run listed in each sweep's status file (structured
artifacts only) and stores one curve per feature distribution, so notebooks and plots
do not depend on the disposable ``outdir/``.

Example
-------
    python utils/export_curves.py --sweep E1=outdir/sweep_e1_... --sweep E1h=...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sweep", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--out", type=Path, default=PROJECT_DIR / "results/curves.json")
    args = parser.parse_args()
    curves = {}
    for item in args.sweep:
        label, directory = item.split("=", 1)
        sweep = Path(directory)
        status = json.loads((sweep / "sweep_status.json").read_text())
        if status["status"] != "COMPLETED":
            raise ValueError(f"{label}: sweep is {status['status']}")
        curves[label] = {"sweep_id": status["sweep_id"], "runs": {}}
        for run in status["runs"]:
            metrics = json.loads((sweep / run["metrics_path"]).read_text())
            curves[label]["runs"][str(run["candidate"]["permutation_seed"])] = metrics[
                "history"
            ]["test_accuracy"]
    args.out.write_text(json.dumps(curves) + "\n")
    print(f"Wrote {args.out} ({len(curves)} systems)")


if __name__ == "__main__":
    main()
