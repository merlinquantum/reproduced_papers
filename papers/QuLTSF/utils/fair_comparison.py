"""Phase 5 fair-comparison table: every system against QuLTSF, with provenance.

Applies the protocol frozen in ``LOG.md`` ("Phase 5 Comparison Plan"):
the same better/tie/worse rule as ``utils/compare_results.py`` (mean test
MSE margin > 2x the larger seed std and Diebold-Mariano p < 0.05), run
counts checked against the sweep plans, paper-reported values kept apart
from locally executed ones, and a commensurability status per system.

Usage (from the paper folder)::

    python utils/fair_comparison.py \
        --sweep-dir outdir/sweep_table1_r2 --sweep-dir outdir/sweep_table1_leftovers \
        --sweep-dir outdir/sweep_budget --sweep-dir outdir/sweep_ablations \
        --sweep-dir outdir/sweep_merlin_control \
        --sweep-dir outdir/sweep_merlin_trained_reduced \
        --plan configs/sweeps/table1.json --plan configs/sweeps/budget.json \
        --plan configs/sweeps/ablations.json --plan configs/sweeps/merlin_control.json \
        --plan configs/sweeps/merlin_trained_reduced.json \
        --out-dir outdir/fair_comparison_<stamp>

Writes ``fair_comparison.json`` and ``fair_comparison.md`` to ``--out-dir`` and
to ``results/``; ``--verify results/fair_comparison.json`` instead recomputes
and compares without writing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "utils"))

import compare_results as cr  # noqa: E402
from run_sweep import expand_plan  # noqa: E402

# (name, role, matching basis, commensurability with QuLTSF, scope label)
SYSTEMS = (
    ("qultsf", "paper method (reproduced)", "reference", "REFERENCE", "paper-accurate, 3 seeds"),
    ("linear", "paper baseline, rerun", "same pipeline, budget, seeds", "COMMENSURABLE", "paper-accurate, 3 seeds"),
    ("nlinear", "paper baseline, rerun", "same pipeline, budget, seeds", "COMMENSURABLE", "paper-accurate, 3 seeds"),
    ("dlinear", "paper baseline, rerun", "same pipeline, budget, seeds", "COMMENSURABLE", "paper-accurate, 3 seeds"),
    ("linear_ltsf_budget", "paper baseline, rerun at LTSF-Linear budget", "same pipeline and seeds; 10 epochs, patience 3, type1", "PARTIALLY_COMMENSURABLE", "paper-accurate for Zeng et al., 3 seeds"),
    ("nlinear_ltsf_budget", "paper baseline, rerun at LTSF-Linear budget", "same pipeline and seeds; 10 epochs, patience 3, type1", "PARTIALLY_COMMENSURABLE", "paper-accurate for Zeng et al., 3 seeds"),
    ("dlinear_ltsf_budget", "paper baseline, rerun at LTSF-Linear budget", "same pipeline and seeds; 10 epochs, patience 3, type1", "PARTIALLY_COMMENSURABLE", "paper-accurate for Zeng et al., 3 seeds"),
    ("persistence", "indexing control (no training)", "same pipeline; no parameters", "COMMENSURABLE", "deterministic, 1 run"),
    ("mlp", "capacity-matched classical (Phase 1 plan)", "same pipeline, budget, seeds; parameter count matched to QuLTSF", "COMMENSURABLE", "paper-accurate settings, 3 seeds"),
    ("rank_bottleneck", "QuLTSF skeleton without hidden layer", "same pipeline, budget, seeds; rank 10", "COMMENSURABLE", "paper-accurate settings, 3 seeds"),
    ("identity_circuit", "QuLTSF without circuit (C3 control)", "same pipeline, budget, seeds; QuLTSF minus 90 circuit weights", "COMMENSURABLE", "paper-accurate settings, 3 seeds"),
    ("frozen_circuit", "QuLTSF with random frozen circuit (C3 control)", "same pipeline, budget, seeds", "COMMENSURABLE", "paper-accurate settings, 3 seeds"),
    ("identity_interferometer_merlin", "MerLin adaptation, no interferometer", "same pipeline, budget, seeds; 1001-state photonic embedding", "COMMENSURABLE", "MerLin adaptation (simulation), 3 seeds"),
    ("qultsf_merlin_e20", "MerLin adaptation, trained interferometer", "same pipeline; seed 2021 only; 20-epoch cap", "PARTIALLY_COMMENSURABLE", "MerLin adaptation (simulation), reduced scope, 1 seed"),
)  # fmt: skip
CONCURRENCY = {
    "sweep_table1_r2": 6,
    "sweep_table1_leftovers": 4,
    "sweep_budget": 6,
    "sweep_ablations": 6,
    "sweep_merlin_control": 6,
    "sweep_merlin_trained_reduced": 2,
}
NOT_COMMENSURABLE = {
    "C4 PatchTST / MTST": "INCOMPARABLE: values quoted from Zhang et al. 2024 (different pipeline, RevIN); not rerun",
    "Transformer baselines (FEDformer, Autoformer, Informer)": "INCOMPARABLE: quoted from Zeng et al. 2023, not rerun; far worse than the linear models",
}


def planned_counts(plan_paths: list[Path]) -> dict[tuple[str, int], int]:
    """Planned runs per (candidate name, horizon) over all plans at L = 336."""
    counts: dict[tuple[str, int], int] = {}
    for plan_path in plan_paths:
        for run in expand_plan(json.loads(plan_path.read_text())):
            match = cr.RUN_ID.match(run["run_id"])
            if int(match["L"]) != cr.TABLE1_LOOKBACK:
                continue
            key = (match["name"], int(match["T"]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def timing(candidate_runs: list[dict]) -> dict:
    """Epochs and per-sweep mean wall-clock (with concurrency) of a candidate.

    Runs from different sweeps shared the GPU with different numbers of
    parallel runs, so their wall-clock is averaged per sweep, not pooled.
    """
    epochs = [run["metrics"]["training"]["epochs_run"] for run in candidate_runs]
    per_sweep = {}
    for sweep in sorted({run["sweep"] for run in candidate_runs}):
        seconds = [
            run["metrics"]["wall_clock_seconds"]
            for run in candidate_runs
            if run["sweep"] == sweep
        ]
        per_sweep[sweep] = {
            "runs": len(seconds),
            "parallel_runs": CONCURRENCY[sweep],
            "wall_clock_minutes_mean": statistics.fmean(seconds) / 60.0,
        }
    return {"epochs_mean": statistics.fmean(epochs), "wall_clock_by_sweep": per_sweep}


def build(runs: dict[str, dict], plan_paths: list[Path]) -> dict:
    """Assemble the per-horizon fair-comparison rows."""
    expected = planned_counts(plan_paths)
    horizons = []
    for horizon in cr.HORIZONS:
        quantum = cr.select(runs, "qultsf", cr.TABLE1_LOOKBACK, horizon)
        rows = [
            {
                "system": "qultsf (paper Table 1)",
                "role": "paper method",
                "provenance": "paper-reported",
                "scope": "single seed (2021), test drop_last",
                "mse": cr.PAPER_TABLE1["qultsf"][horizon][0],
                "mae": cr.PAPER_TABLE1["qultsf"][horizon][1],
            }
        ]
        for name in cr.TABLE1_BASELINES:
            rows.append(
                {
                    "system": f"{name} (paper Table 1, quoted from Zeng et al.)",
                    "role": "paper baseline",
                    "provenance": "paper-reported",
                    "scope": "LTSF-Linear budget, single run, other pipeline",
                    "mse": cr.PAPER_TABLE1[name][horizon][0],
                    "mae": cr.PAPER_TABLE1[name][horizon][1],
                }
            )
        for name, role, basis, status, scope in SYSTEMS:
            chosen = cr.select(runs, name, cr.TABLE1_LOOKBACK, horizon)
            planned = expected.get((name, horizon), 0)
            if not chosen and not planned:
                continue
            row = {
                "system": name,
                "role": role,
                "provenance": "local",
                "scope": scope,
                "matching_basis": basis,
                "commensurability": status if chosen else "EVIDENCE_MISSING",
                "successful_runs": len(chosen),
                "expected_runs": planned,
            }
            if chosen:
                row |= cr.aggregate(chosen) | timing(chosen)
                if name != "qultsf":
                    row["qultsf_vs_system"] = cr.compare(quantum, chosen, horizon)
            rows.append(row)
        horizons.append({"T": horizon, "rows": rows})
    return {
        "schema_version": 1,
        "protocol": "LOG.md 'Phase 5 Comparison Plan'",
        "rule": (
            "QuLTSF better than a system if its mean test MSE is lower by > 2x the "
            "larger seed std and DM p < 0.05 (seed-averaged per-window squared "
            "errors, lag window T); worse symmetrically; otherwise tie"
        ),
        "hardware": "NVIDIA L4 (reproduced-papers-1), float32; MerLin exact simulation",
        "not_commensurable": NOT_COMMENSURABLE,
        "horizons": horizons,
    }


def _number(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def render(table: dict) -> str:
    """Render the fair-comparison table as Markdown."""
    lines = [
        "# QuLTSF fair comparison (Phase 5)",
        "",
        f"Protocol: {table['protocol']}. Rule: {table['rule']}. Hardware: "
        f"{table['hardware']}. Wall-clock is per run, with the number of runs "
        "sharing the GPU in brackets; it describes cost, not a speed claim.",
    ]
    for block in table["horizons"]:
        lines += [
            "",
            f"## T = {block['T']}",
            "",
            "| System | Role | Provenance | Scope | Runs (ok / expected) | Test MSE | MSE std | Test MAE | Params | Wall-clock min (parallel) | Epochs | Commensurability | QuLTSF vs system |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---|",
        ]
        for row in block["rows"]:
            if row["provenance"] == "paper-reported":
                lines.append(
                    f"| {row['system']} | {row['role']} | paper-reported | {row['scope']} "
                    f"| - | {row['mse']:.3f} | - | {row['mae']:.3f} | - | - | - | "
                    "PARTIALLY_COMMENSURABLE (reported value) | - |"
                )
                continue
            if row["successful_runs"] == 0:
                lines.append(
                    f"| {row['system']} | {row['role']} | local | {row['scope']} | "
                    f"0 / {row['expected_runs']} | - | - | - | - | - | - | "
                    f"{row['commensurability']} | - |"
                )
                continue
            wall = "; ".join(
                f"{t['wall_clock_minutes_mean']:.1f} ({t['runs']} run(s), "
                f"{t['parallel_runs']} parallel, `{s}`)"
                for s, t in row["wall_clock_by_sweep"].items()
            )
            verdict = row.get("qultsf_vs_system")
            verdict_text = (
                "-"
                if verdict is None
                else f"{verdict['verdict']} (margin {verdict['margin']:+.4f}, p {verdict['dm_p_value']:.2g})"
            )
            lines.append(
                f"| {row['system']} | {row['role']} | local | {row['scope']} | "
                f"{row['successful_runs']} / {row['expected_runs']} | {row['mse_mean']:.4f} | "
                f"{_number(row['mse_std'])} | {row['mae_mean']:.4f} | "
                f"{row['trainable_parameters']:,} | {wall} | {row['epochs_mean']:.0f} | "
                f"{row['commensurability']} | {verdict_text} |"
            )
    lines += ["", "## Not compared", ""]
    for system, reason in table["not_commensurable"].items():
        lines.append(f"- {system}: {reason}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--out-dir", type=Path)
    output.add_argument("--verify", type=Path)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args(argv)

    runs = cr.collect_runs(args.sweep_dir)
    table = build(runs, args.plan)
    coverage = cr.plan_coverage(runs, args.plan)
    table["plan_coverage"] = coverage
    for plan in coverage["plans"]:
        print(
            f"{plan['plan']}: {plan['observed_runs']}/{plan['expected_runs']}; "
            f"missing {plan['missing'] or 'none'}; {plan['sweep_label']}"
        )
    print(f"unplanned: {coverage['unplanned'] or 'none'}")
    status = 0 if coverage["all_planned_runs_completed"] else 1

    if args.verify is not None:
        if json.loads(json.dumps(table)) != json.loads(args.verify.read_text()):
            print(f"VERIFY FAILED: {args.verify} differs from the recomputed table")
            return 1
        print(f"VERIFY OK: {args.verify} matches the recomputed table")
        return status

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fair_comparison.json").write_text(json.dumps(table, indent=2))
    (args.out_dir / "fair_comparison.md").write_text(render(table))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name in ("fair_comparison.json", "fair_comparison.md"):
        shutil.copyfile(args.out_dir / name, args.results_dir / name)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
