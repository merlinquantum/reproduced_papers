"""Measure whether trained QSTL circuits keep the context register (Eq. 5 assumption).

For each QSTL run of a sweep and each basis context ``c``, apply the trained
circuit to ``|c>|0>`` and record the probability that the context register is
still ``c``. Eq. 5 and the QBGU argument (Sec. IV.A) assume this probability
is 1. Writes ``context_retention.csv`` into ``--out``.

Usage::

    python papers/QNN_STOCK/utils/context_retention.py <sweep_dir> --out results/phase3_s1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR.parents[1]))
sys.path.insert(0, str(PAPER_DIR))

from lib.statevector import basis_state, probabilities  # noqa: E402
from lib.training import qstl_circuit_factory  # noqa: E402


def retention_rows(sweep_dir: Path) -> list[dict]:
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    rows = []
    for run in status["runs"]:
        run_dir = sweep_dir / run["run_dir"]
        snapshot = json.loads(
            (run_dir / "config_snapshot.json").read_text(encoding="utf-8")
        )
        if snapshot["experiment"]["kind"] != "qstl":
            continue
        levels = snapshot["dataset"]["levels"]
        n_contexts = levels ** snapshot["dataset"]["context_length"]
        n_qubits = int(math.log2(n_contexts * levels))
        params = np.load(run_dir / "params.npz")
        # Runs made before model.ansatz existed used the paper HEA.
        model_cfg = {"ansatz": "hea", **snapshot["model"]}
        factory, _ = qstl_circuit_factory(model_cfg, n_qubits, levels)
        # ``loader_<ticker>`` entries are the 3-qubit HEA context loaders, not the QNN.
        for key in (k for k in params.files if k.startswith("qstl_")):
            asset = key.removeprefix("qstl_")
            for context in range(n_contexts):
                output = factory(params[key])(basis_state(context * levels, n_qubits))
                joint = probabilities(output).reshape(n_contexts, levels)
                rows.append(
                    {
                        "candidate": run["candidate"]["config"],
                        "seed": run["seed"],
                        "asset": asset,
                        "context": context,
                        "p_context_kept": float(joint[context].sum()),
                    }
                )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = retention_rows(args.sweep_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "context_retention.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} written to {args.out / 'context_retention.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
