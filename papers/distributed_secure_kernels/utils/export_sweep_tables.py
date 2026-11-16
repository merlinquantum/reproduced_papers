"""Write the full candidate table of every included sweep to ``results/sweep_results.{md,json}``.

Each table is recomputed from the runs' ``metrics.json`` with
:func:`utils.summarize_sweep.collect` and must equal the sweep's stored
``summary.json``; a partial sweep or a mismatch stops the export. The sweeps
repeat fixed configurations over seeds 0-4: no candidate is selected across a
sweep (C and kernel hyperparameters of the corrected protocol are chosen by
inner CV inside each run). The JSON keeps the per-seed values, which the
notebook reads.

Usage::

    .venv/bin/python papers/distributed_secure_kernels/utils/export_sweep_tables.py <sweep dir> [<sweep dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))

from distributed_secure_kernels.utils.summarize_sweep import collect  # noqa: E402

ROLES = {
    "table1_original": "Table 1, paper protocol, with the exact-kernel and majority-class references",
    "noise_original": "Fig. 3, the authors' noise model with noiseless state preparation",
    "noise_decomposed_initialize": "Fig. 3 diagnostic (designed after the gap was seen): noisy, decomposed state preparation",
    "shots_original": "Fig. 4, Digits, 128 to 1,024 shots",
    "table1_corrected": "Corrected protocol: paper columns and tuned RBF / polynomial SVMs (Framingham C at most 10)",
    "merlin_original": "MerLin photonic swap test (adaptation), Table 1 and Fig. 4 settings, paper protocol",
    "fair_comparison": "Fair comparison: tuned RBF / polynomial SVMs on the planned grids and MerLin, corrected protocol",
}


def _table(summary: dict) -> list[str]:
    lines = [
        "| Candidate | Method | Model | Runs | Accuracy | Std over seeds | Mean fold std | Balanced accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        acc, bal = row["accuracy"], row["balanced_accuracy"]
        lines.append(
            f"| {row['candidate']} | {row['method']} | {row['model']} | {row['successful_runs']}/{row['expected_runs']} "
            f"| {acc['seed_mean']:.4f} | {acc['seed_std']:.4f} | {acc['mean_fold_std']:.4f} | {bal['seed_mean']:.4f} |"
        )
    return lines


def main(argv: list[str]) -> int:
    lines = [
        "# Sweep results",
        "",
        "Full candidate tables of the included sweeps, recomputed from every run's `metrics.json` "
        "and checked against each sweep's `summary.json`. Metric: stratified 5-fold CV accuracy "
        "(higher is better); aggregate: mean over seeds 0-4 of the 5-fold mean; variability: std over "
        "seeds (ddof = 1); the mean fold std is the paper's ±. Two methods are equivalent when "
        "|difference| ≤ 2 σ_pooled. No candidate is selected across a sweep.",
    ]
    exported = {}
    for arg in argv:
        sweep_dir = Path(arg)
        recomputed = collect(sweep_dir)
        stored = json.loads((sweep_dir / "summary.json").read_text())
        if recomputed["coverage"] != "COMPLETE SWEEP":
            raise ValueError(f"{sweep_dir.name}: {recomputed['coverage']}")
        if json.loads(json.dumps(recomputed, sort_keys=True)) != stored:
            raise ValueError(
                f"{sweep_dir.name}: summary.json differs from metrics.json"
            )
        runs = sum(len(seeds) for seeds in recomputed["planned"].values())
        exported[recomputed["sweep_id"]] = {
            "folder": sweep_dir.name,
            "coverage": recomputed["coverage"],
            "runs": runs,
            "rule": recomputed["rule"],
            "rows": recomputed["rows"],
        }
        lines += [
            "",
            f"## `{recomputed['sweep_id']}`",
            "",
            f"{ROLES[recomputed['sweep_id']]}. Sweep folder `{sweep_dir.name}`; "
            f"{recomputed['coverage']}, {runs}/{runs} runs.",
            "",
            *_table(recomputed),
        ]
    results = PROJECT_DIR / "results"
    (results / "sweep_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (results / "sweep_results.json").write_text(
        json.dumps(exported, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(results / "sweep_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
