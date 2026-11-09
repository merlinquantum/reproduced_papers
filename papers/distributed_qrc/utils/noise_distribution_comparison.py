"""Noise side check (Sec. III-B5 remark): does distributing the quantum components mitigate noise?

Reproduces the paper's own comparison (Sec. III-B, cross-architecture
paragraph) on our sweeps: the fully quantum noisy cell
(quantum_noisy reservoir + quantum_kernel_noisy readout) of MRSR, SRMR and
MRMR against SRSR at the same total neuron count. The paper reports that
noisy MRSR beats noisy SRSR at every N, MRMR beats SRSR from N = 40 up, and
SRMR does not.

For each pair the table gives our test-MAE difference (distributed - SRSR),
mean over seeds, the pooled seed std, and the paper's difference. Note the
confound: under the paper's device mapping (Table III) SRSR
runs on ibm_brisbane only, while distributed components also use the less
noisy ibm_marrakesh and IonQ Aria-1.

Writes ``results/data/noise_distribution_comparison.json`` (``-R2`` for the evidence sweeps).

Usage (from the repository root)::

    python papers/distributed_qrc/utils/noise_distribution_comparison.py papers/distributed_qrc/outdir
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
CELL = ("quantum_noisy", "quantum_kernel_noisy")
SWEEPS = {"srsr": "E1", "mrsr": "E2", "srmr": "E3", "mrmr": "E4"}


def total(label: str) -> int:
    if "x" in label:
        count, size = label.split("x")
        return int(count) * int(size)
    return int(label.split("/")[0])


def our_values(outdir: Path, architecture: str, suffix: str) -> dict[int, np.ndarray]:
    sweep_dir = outdir / f"sweep_{SWEEPS[architecture]}{suffix}"
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_dir} is {status['status']}")
    values: dict[int, list[float]] = {}
    for run in status["runs"]:
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        for cell in metrics["cells"]:
            if (cell["reservoir_type"], cell["readout_type"]) == CELL:
                values.setdefault(total(cell["configuration"]), []).append(
                    cell["metrics"]["test"]["mae"]
                )
    return {k: np.array(v) for k, v in values.items()}


def main() -> int:
    outdir = Path(sys.argv[1])
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    paper = {}
    for c in json.loads(
        (PAPER_DIR / "results" / "data" / "paper_tables.json").read_text()
    )["cells"]:
        if (c["reservoir_type"], c["readout_type"]) == CELL:
            paper[(c["architecture"], total(c["configuration"]))] = c["mae"]
    ours = {a: our_values(outdir, a, suffix) for a in SWEEPS}
    result = {}
    print(
        f"{'pair':12s} {'N':>3s} | {'ours: distributed - SRSR (MAE)':>32s} {'/ pooled std':>12s} | {'paper diff':>10s}"
    )
    for architecture in ("mrsr", "srmr", "mrmr"):
        for n in sorted(ours[architecture]):
            a, b = ours[architecture][n], ours["srsr"][n]
            diff = float(a.mean() - b.mean())
            pooled = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
            paper_diff = paper[(architecture, n)] - paper[("srsr", n)]
            result[f"{architecture}-vs-srsr@{n}"] = {
                "ours_diff": diff,
                "ours_diff_in_pooled_std": diff / pooled
                if pooled > 0
                else float("nan"),
                "paper_diff": paper_diff,
                "same_sign": bool(np.sign(diff) == np.sign(paper_diff)),
            }
            print(
                f"{architecture}-vs-srsr {n:3d} | {diff:+32.5f} {diff / pooled:+12.2f} | {paper_diff:+10.4f}"
            )
    out = PAPER_DIR / "results" / "data" / "noise_distribution_comparison.json"
    result = {
        "paper_reference": "Sec. III-B; noisy fully quantum cells of Tables C3, C6, C9, C12; device mapping Table III",
        **result,
    }
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {os.path.relpath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
