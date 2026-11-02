#!/usr/bin/env python3
"""Generate the README results tables from structured artifacts.

1. ``results/paper_items.md``: one row per reproduced paper item (Table I
   metrics for QK-LSTM and LSTM, Table II parameter counts), with paper value,
   reproduced mean ± std, delta and agreement labels. The values come from the
   claim-table JSON written by ``utils/claim_statistics.py``.
2. ``results/sweeps_index.md``: every sweep's protocol, coverage and link to its
   full candidate table, from each ``sweep_status.json`` and ``plan.json``.

Usage::

    .venv/bin/python papers/QK_LSTM/utils/results_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_DIR / "results"
SWEEPS = PROJECT_DIR / "outdir" / "sweeps"
PAPER = {
    "QK-LSTM": {"rmse": 9.20, "mae": 7.15, "mape": 9.14, "r2": 0.84, "params": 209},
    "LSTM": {"rmse": 15.94, "mae": 11.07, "mape": 13.32, "r2": 0.78, "params": 1873},
}
# Sweep protocol descriptions (declared in LOG.md Sweep Records before launch).
SWEEP_PROTOCOL = {
    "s1_paper_hp": (
        "test RMSE (observed AQI), minimize",
        "80/20 test",
        "comparison; no selection",
    ),
    "s2_split2020": (
        "test RMSE (observed AQI), minimize",
        "calendar-2020 test",
        "comparison; no selection",
    ),
    "s3_val_tuning": (
        "validation RMSE (observed AQI), minimize; ties within 1% go to fewer params then smaller lr (ridge: larger alpha)",
        "validation (2018-12-18..2019-05-26)",
        "per-family selection",
    ),
    "s4_tuned_test": (
        "test RMSE (observed AQI), minimize",
        "80/20 test",
        "comparison; no selection",
    ),
    "s5_gap_diagnostics": (
        "test RMSE / R² (diagnostic)",
        "80/20 test",
        "diagnostic; no selection",
    ),
    "s5b_known_answer_attempt2": (
        "test R² > 0.95 (known answer)",
        "80/20 test",
        "diagnostic; no selection",
    ),
    "s5c_known_answer_clean": (
        "test R² > 0.95 (known answer)",
        "80/20 test",
        "diagnostic; no selection",
    ),
    "s7_fresh_seeds": (
        "test RMSE (observed AQI), minimize; paired 95% CI rule",
        "80/20 test",
        "replication on fresh seeds 5-9; no selection",
    ),
    "s8_causal_fill": (
        "test RMSE (observed AQI), minimize; paired 95% CI vs S1",
        "80/20 test",
        "causal (forward-fill) preprocessing check; no selection",
    ),
    "s6_merlin": (
        "test RMSE (observed AQI), minimize; preservation tolerance ±1.0",
        "80/20 test",
        "comparison; no selection",
    ),
}
EXCLUDED_SWEEPS = {"s0_coordinator_selftest"}
SWEEP_PURPOSE = {
    "s1_paper_hp": "S1: all models at paper hyperparameters",
    "s2_split2020": "S2: calendar-2020 split sensitivity",
    "s3_val_tuning": "S3: equal-budget validation tuning + D2/angle-map checks",
    "s4_tuned_test": "S4: tuned LSTMs on test",
    "s5_gap_diagnostics": "S5: gap diagnostics (same-day horizon; known-answer attempt 1)",
    "s5b_known_answer_attempt2": "S5b: known-answer attempt 2 (supersedes the 6 S5 failures)",
    "s5c_known_answer_clean": "S5c: clean known-answer (preprocessing disabled)",
    "s6_merlin": "S6: MerLin photonic adaptation",
    "s7_fresh_seeds": "S7: decisive comparisons on fresh seeds 5-9 (post-handoff check)",
    "s8_causal_fill": "S8: causal preprocessing check (post-handoff)",
}


def _load(name: str) -> dict:
    return json.loads((RESULTS / "claims" / f"{name}.json").read_text())


def _agreement(paper: float, mean: float, std: float) -> str:
    return "within 1 std" if abs(mean - paper) <= std else "not matched"


def paper_items() -> str:
    table_a = _load("table_a_paper_hp")["table"]
    table_b = _load("table_b_tuned")["table"]
    rows = []
    for model, key, claim in (
        ("QK-LSTM", "Q-B quantum", "C1"),
        ("LSTM", "B1 LSTM(16)", "C1 (paper baseline)"),
    ):
        for metric in ("rmse", "mae", "mape", "r2"):
            for label, table in (
                ("paper hp (S1)", table_a),
                ("tuned (S1/S4)", table_b),
            ):
                stats = table[key][metric]
                paper_value = PAPER[model][metric]
                rows.append(
                    f"| Table I, {model} {metric.upper()} | {claim} | {paper_value:.2f} | {stats['mean']:.2f} ± {stats['std']:.2f} ({label}) | "
                    f"{stats['mean'] - paper_value:+.2f} | {_agreement(paper_value, stats['mean'], stats['std'])} | "
                    f"reversed ranking (LSTM better than QK-LSTM) | unsupported | 5 | {key}; observed AQI targets; 80/20 split{'; tuned = S1 runs (S3 selected the paper configuration)' if label.startswith('tuned') and model == 'QK-LSTM' else ''} |"
                )
    for model, key in (("QK-LSTM", "Q-B quantum"), ("LSTM", "B1 LSTM(16)")):
        params = table_a[key]["params"][0]
        rows.append(
            f"| Table II, {model} trainable params | C2 | {PAPER[model]['params']} | {params} | {params - PAPER[model]['params']:+d} | exact | "
            "count ratio reproduced (9.0×) | count reproduced; compactness advantage unsupported (196-param LSTM more accurate) | — | "
            f"{'Linear(27→4)+Linear(4→16)+Linear(16→1)' if model == 'QK-LSTM' else 'nn.LSTM(11,16)+Linear(16,1)'} |"
        )
    rows.append(
        "| Fig. 3(a–d) traces | C3 | qualitative | `results/s1_paper_hp/{test,full_period}_predictions.png` | — | n/a | "
        "not reproduced (our QK-LSTM is biased in 2019-12..2020-02; our LSTM is not) | unsupported | 1 (median seed) | median-RMSE seed |"
    )
    header = [
        "| Paper item | Claim tested | Paper value | Reproduced value (mean ± sample std over seeds) | Delta | Metric agreement | Trend agreement | Claim support | Seeds | Comment |",
        "|---|---|---:|---:|---:|---|---|---|---:|---|",
    ]
    return "\n".join(header + rows) + "\n"


def sweeps_index() -> str:
    lines = [
        "| Sweep | Purpose | Metric / direction / selection | Split | Checkpoint | Aggregation | Successful / expected runs | Coverage | Full candidate table |",
        "|---|---|---|---|---|---|---:|---|---|",
    ]
    for sweep_dir in sorted(SWEEPS.iterdir()):
        status_path = sweep_dir / "sweep_status.json"
        if not status_path.exists():
            continue
        plan = json.loads((sweep_dir / "plan.json").read_text())
        name = plan["sweep_name"]
        if name in EXCLUDED_SWEEPS:
            continue
        status = json.loads(status_path.read_text())
        done = sum(run["status"] == "COMPLETED" for run in status["runs"])
        metric, split, selection = SWEEP_PROTOCOL[name]
        coverage = (
            "COMPLETE SWEEP"
            if status["status"] == "COMPLETED"
            else f"PARTIAL SWEEP ({status['status']})"
        )
        artifact_file = {
            "s3_val_tuning": "selection",
            "s7_fresh_seeds": "comparison",
            "s8_causal_fill": "comparison",
        }.get(name, "report")
        artifact = f"`results/{name}/{artifact_file}.md`"
        lines.append(
            f"| `{status['sweep_id']}` | {SWEEP_PURPOSE[name]} | {metric}; {selection} | {split} | final epoch | mean ± sample std over seeds | "
            f"{done} / {status['expected_runs']} | {coverage} | {artifact} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    (RESULTS / "paper_items.md").write_text(paper_items(), encoding="utf-8")
    (RESULTS / "sweeps_index.md").write_text(sweeps_index(), encoding="utf-8")
    print((RESULTS / "paper_items.md").read_text())
    print((RESULTS / "sweeps_index.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
