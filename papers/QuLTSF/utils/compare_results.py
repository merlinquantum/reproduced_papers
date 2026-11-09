"""Build the Table 1, budget, ablation and lookback comparisons from sweeps.

Reads only structured artifacts: each sweep's ``sweep_status.json`` ledger,
each run's ``run_status.json``, ``metrics.json`` and ``test_errors.npz``.
Applies the better/tie/worse rule predeclared in ``LOG.md`` (Fair Baseline
Plan): QuLTSF is "better" than a reference at a setting only if its mean test
MSE is lower by more than 2x the larger seed std and a Diebold-Mariano test on
the seed-averaged per-window squared errors gives p < 0.05 in its favour.

Included runs: a run is included when its sweep ledger and its own
``run_status.json`` both say COMPLETED. Each run ID may be included once
across all ``--sweep-dir`` folders, so a sweep that timed out contributes only
its finished runs and its retry sweep must hold exactly the rest. ``--plan``
checks that every planned run ID is included and that no included run lies
outside the plans (exit code 1 otherwise). A plan is labelled COMPLETE SWEEP
only if, in addition, every sweep it draws on reached COMPLETED; a timed-out
sweep completed by a retry stays PARTIAL SWEEP, and that label is appended to
the plan's result label.

Usage (from the paper folder)::

    python utils/compare_results.py \
        --sweep-dir outdir/sweep_table1_r2 --sweep-dir outdir/sweep_table1_leftovers \
        --sweep-dir outdir/sweep_budget --sweep-dir outdir/sweep_ablations \
        --sweep-dir outdir/sweep_lookback \
        --plan configs/sweeps/table1.json --plan configs/sweeps/budget.json \
        --plan configs/sweeps/ablations.json --plan configs/sweeps/lookback.json \
        --out-dir outdir/comparison_<stamp>

Writes ``comparison.json`` and ``comparison.md`` to ``--out-dir`` and to
``results/``. With ``--verify results/comparison.json`` in place of
``--out-dir`` it writes nothing: it recomputes the comparison and exits 0
only if it equals the given file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

from papers.QuLTSF.lib.statistics import diebold_mariano  # noqa: E402

sys.path.insert(0, str(PROJECT_DIR / "utils"))
from run_sweep import expand_plan  # noqa: E402

RUN_ID = re.compile(r"^(?P<name>.+)_L(?P<L>\d+)_T(?P<T>\d+)_seed(?P<seed>\d+)$")
HORIZONS = (96, 192, 336, 720)
TABLE1_LOOKBACK = 336
SIGNIFICANCE = 0.05

# Paper Table 1 (arXiv:2412.13769), Weather, L = 336: (MSE, MAE). The linear
# rows are quoted there from Zeng et al. (2023), not rerun by the authors.
PAPER_TABLE1 = {
    "qultsf": {96: (0.156, 0.211), 192: (0.199, 0.253), 336: (0.248, 0.296), 720: (0.315, 0.346)},
    "linear": {96: (0.176, 0.236), 192: (0.218, 0.276), 336: (0.262, 0.312), 720: (0.326, 0.365)},
    "nlinear": {96: (0.182, 0.232), 192: (0.225, 0.269), 336: (0.271, 0.301), 720: (0.338, 0.348)},
    "dlinear": {96: (0.176, 0.237), 192: (0.220, 0.282), 336: (0.265, 0.319), 720: (0.323, 0.362)},
}  # fmt: skip
TABLE1_BASELINES = ("linear", "nlinear", "dlinear")
BUDGET_BASELINES = ("linear_ltsf_budget", "nlinear_ltsf_budget", "dlinear_ltsf_budget")
ABLATIONS = ("identity_circuit", "frozen_circuit", "rank_bottleneck", "mlp")
PHOTONIC_MODEL = "qultsf_merlin_e20"
PHOTONIC_CONTROL = "identity_interferometer_merlin"


def collect_runs(sweep_dirs: list[Path]) -> dict[str, dict]:
    """Return every COMPLETED run of the given sweeps, keyed by run ID.

    Parameters
    ----------
    sweep_dirs : list[pathlib.Path]
        Sweep directories holding ``sweep_status.json``.

    Returns
    -------
    dict[str, dict]
        ``name``, ``L``, ``T``, ``seed``, ``metrics``, ``errors_path`` and
        ``sweep`` per run ID.

    Raises
    ------
    ValueError
        If a run ID appears twice, a ledger entry disagrees with its run
        status, or a test metric is not finite.
    """
    runs: dict[str, dict] = {}
    for sweep_dir in sweep_dirs:
        ledger = json.loads((sweep_dir / "sweep_status.json").read_text())
        for entry in ledger["runs"]:
            if entry["status"] != "COMPLETED":
                continue
            run_dir = sweep_dir / entry["run_dir"]
            status = json.loads((run_dir / "run_status.json").read_text())
            if status["status"] != "COMPLETED" or status["run_id"] != entry["run_id"]:
                raise ValueError(f"{run_dir}: ledger and run status disagree")
            if entry["run_id"] in runs:
                raise ValueError(f"Duplicate run {entry['run_id']} in {sweep_dir}")
            metrics = json.loads((run_dir / "metrics.json").read_text())
            for key in ("mse", "mae"):
                if not math.isfinite(metrics["test"][key]):
                    raise ValueError(f"{run_dir}: non-finite test {key}")
            match = RUN_ID.match(entry["run_id"])
            runs[entry["run_id"]] = {
                "name": match["name"],
                "L": int(match["L"]),
                "T": int(match["T"]),
                "seed": int(match["seed"]),
                "metrics": metrics,
                "errors_path": run_dir / metrics["test_errors_path"],
                "sweep": sweep_dir.name,
                "sweep_status": ledger["status"],
            }
    return runs


def select(runs: dict[str, dict], name: str, lookback: int, horizon: int) -> list[dict]:
    """Return the runs of one candidate, ordered by seed."""
    chosen = [
        run
        for run in runs.values()
        if run["name"] == name and run["L"] == lookback and run["T"] == horizon
    ]
    return sorted(chosen, key=lambda run: run["seed"])


def _std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def aggregate(candidate_runs: list[dict]) -> dict:
    """Mean and sample std (ddof=1) of test MSE/MAE over seeds."""
    mse = [run["metrics"]["test"]["mse"] for run in candidate_runs]
    mae = [run["metrics"]["test"]["mae"] for run in candidate_runs]
    return {
        "n_runs": len(candidate_runs),
        "seeds": [run["seed"] for run in candidate_runs],
        "sweeps": sorted({run["sweep"] for run in candidate_runs}),
        "mse_mean": statistics.fmean(mse),
        "mse_std": _std(mse),
        "mae_mean": statistics.fmean(mae),
        "mae_std": _std(mae),
        "trainable_parameters": candidate_runs[0]["metrics"]["trainable_parameters"],
    }


def seed_mean_losses(candidate_runs: list[dict], kind: str) -> np.ndarray:
    """Per-window loss averaged over seeds (windows are identical across seeds)."""
    series = [np.load(run["errors_path"])[kind] for run in candidate_runs]
    if len({s.shape for s in series}) != 1:
        raise ValueError("Per-window error series differ in length across seeds")
    return np.mean(series, axis=0)


def compare(quantum: list[dict], reference: list[dict], horizon: int) -> dict:
    """Apply the predeclared better/tie/worse rule to QuLTSF vs one reference.

    Parameters
    ----------
    quantum : list[dict]
        QuLTSF runs (one per seed).
    reference : list[dict]
        Reference model runs on the same seeds.
    horizon : int
        Forecast horizon, the DM autocorrelation window.

    Returns
    -------
    dict
        Means, margin, threshold, DM statistic and p-value on the
        seed-averaged losses, per-seed DM p-values, and the verdict.
    """
    q, r = aggregate(quantum), aggregate(reference)
    margin = r["mse_mean"] - q["mse_mean"]
    stds = [s for s in (q["mse_std"], r["mse_std"]) if s is not None]
    threshold = 2.0 * max(stds) if stds else 0.0
    statistic, p_value = diebold_mariano(
        seed_mean_losses(reference, "squared"),
        seed_mean_losses(quantum, "squared"),
        horizon,
    )
    per_seed = {}
    by_seed = {run["seed"]: run for run in reference}
    for run in quantum:
        if run["seed"] in by_seed:
            _, p_seed = diebold_mariano(
                np.load(by_seed[run["seed"]]["errors_path"])["squared"],
                np.load(run["errors_path"])["squared"],
                horizon,
            )
            per_seed[str(run["seed"])] = p_seed
    significant = p_value < SIGNIFICANCE
    if margin > threshold and significant and statistic > 0:
        verdict = "better"
    elif -margin > threshold and significant and statistic < 0:
        verdict = "worse"
    else:
        verdict = "tie"
    return {
        "quantum_mse": q["mse_mean"],
        "reference_mse": r["mse_mean"],
        "margin": margin,
        "threshold": threshold,
        "dm_statistic": statistic,
        "dm_p_value": p_value,
        "dm_p_value_per_seed": per_seed,
        "verdict": verdict,
    }


def build_comparison(runs: dict[str, dict]) -> dict:
    """Assemble every comparison table from the collected runs."""
    table1, budget, ablations, lookback = [], [], [], []
    for horizon in HORIZONS:
        rows = {}
        for name in ("qultsf", *TABLE1_BASELINES, "persistence"):
            candidate = select(runs, name, TABLE1_LOOKBACK, horizon)
            if candidate:
                rows[name] = aggregate(candidate) | {
                    "paper": PAPER_TABLE1.get(name, {}).get(horizon)
                }
        quantum = select(runs, "qultsf", TABLE1_LOOKBACK, horizon)
        best = min(TABLE1_BASELINES, key=lambda name: rows[name]["mse_mean"])
        table1.append(
            {
                "T": horizon,
                "rows": rows,
                "best_baseline": best,
                "vs_best_baseline": compare(
                    quantum, select(runs, best, TABLE1_LOOKBACK, horizon), horizon
                ),
            }
        )

        budget_rows = {
            name: aggregate(select(runs, name, TABLE1_LOOKBACK, horizon))
            | {"paper": PAPER_TABLE1[name.split("_")[0]][horizon]}
            for name in BUDGET_BASELINES
            if select(runs, name, TABLE1_LOOKBACK, horizon)
        }
        if budget_rows:
            best_budget = min(budget_rows, key=lambda n: budget_rows[n]["mse_mean"])
            budget.append(
                {
                    "T": horizon,
                    "rows": budget_rows,
                    "best_baseline": best_budget,
                    "vs_best_baseline": compare(
                        quantum,
                        select(runs, best_budget, TABLE1_LOOKBACK, horizon),
                        horizon,
                    ),
                }
            )

        ablation_rows = {}
        for name in ABLATIONS:
            control = select(runs, name, TABLE1_LOOKBACK, horizon)
            if control:
                ablation_rows[name] = aggregate(control) | {
                    "qultsf_vs_control": compare(quantum, control, horizon)
                }
        if ablation_rows:
            ablations.append({"T": horizon, "rows": ablation_rows})

    for horizon in HORIZONS:
        lookbacks = sorted(
            {run["L"] for run in runs.values() if run["sweep"] == "sweep_lookback"}
        )
        for lookback_length in lookbacks:
            rows = {
                name: aggregate(select(runs, name, lookback_length, horizon))
                for name in ("qultsf", *TABLE1_BASELINES)
                if select(runs, name, lookback_length, horizon)
            }
            if rows:
                lookback.append({"T": horizon, "L": lookback_length, "rows": rows})

    photonic = []
    for horizon in HORIZONS:
        models = {
            name: select(runs, name, TABLE1_LOOKBACK, horizon)
            for name in (PHOTONIC_MODEL, PHOTONIC_CONTROL)
        }
        models = {name: chosen for name, chosen in models.items() if chosen}
        if not models:
            continue
        quantum = select(runs, "qultsf", TABLE1_LOOKBACK, horizon)
        best = min(
            TABLE1_BASELINES,
            key=lambda name: aggregate(select(runs, name, TABLE1_LOOKBACK, horizon))[
                "mse_mean"
            ],
        )
        baseline = select(runs, best, TABLE1_LOOKBACK, horizon)
        block = {"T": horizon, "best_baseline": best, "rows": {}}
        for name, chosen in models.items():
            # A "tie" with QuLTSF means the photonic model preserves its behaviour.
            block["rows"][name] = aggregate(chosen) | {
                "vs_qultsf": compare(chosen, quantum, horizon),
                "vs_best_baseline": compare(chosen, baseline, horizon),
            }
        if len(models) == 2:
            block["trained_vs_no_interferometer"] = compare(
                models[PHOTONIC_MODEL], models[PHOTONIC_CONTROL], horizon
            )
        photonic.append(block)

    return {
        "schema_version": 1,
        "metric_space": "standardised (train-fitted scaler), full test set",
        "rule": (
            "better: mean test MSE lower by > 2x the larger seed std and DM p < "
            f"{SIGNIFICANCE} (seed-averaged per-window squared errors, lag window T)"
        ),
        "labels": {
            "table1": "paper-accurate (3 seeds, full data, paper hyperparameters)",
            "budget": "paper-accurate (3 seeds, LTSF-Linear defaults)",
            "ablations": "paper-accurate settings, 3 seeds, controls not in the paper",
            "lookback": "preliminary (1 seed)",
            "merlin": (
                "MerLin adaptation, not the paper's model, exact simulation; "
                "trained interferometer reduced-scope (seed 2021, 20-epoch cap); "
                "no-interferometer control 3 seeds, Table 1 settings"
            ),
        },
        "n_runs": len(runs),
        "table1": table1,
        "budget": budget,
        "ablations": ablations,
        "lookback": lookback,
        "photonic": photonic,
    }


def _cell(row: dict) -> str:
    def pm(mean, std):
        return f"{mean:.4f}" + (f" ± {std:.4f}" if std is not None else "")

    return (
        f"{pm(row['mse_mean'], row['mse_std'])} / {pm(row['mae_mean'], row['mae_std'])}"
    )


def _paper(row: dict) -> str:
    return f"{row['paper'][0]:.3f} / {row['paper'][1]:.3f}" if row.get("paper") else "-"


def _verdict(result: dict) -> str:
    return (
        f"{result['verdict']} (margin {result['margin']:+.4f}, threshold "
        f"{result['threshold']:.4f}, DM {result['dm_statistic']:+.2f}, "
        f"p {result['dm_p_value']:.2g})"
    )


def render_markdown(comparison: dict) -> str:
    """Render the comparison as Markdown tables."""
    lines = [
        "# QuLTSF comparison tables",
        "",
        f"Generated by `utils/compare_results.py` from {comparison['n_runs']} "
        "completed runs. Test MSE / MAE on standardised Weather data, mean ± "
        "sample std over seeds. Rule: " + comparison["rule"] + ".",
        "",
    ]
    coverage = comparison.get("plan_coverage")
    if coverage:
        lines.append(
            "All planned runs completed: "
            f"{'yes' if coverage['all_planned_runs_completed'] else 'no'}."
        )
        lines.append("")
        for plan in coverage["plans"]:
            sources = ", ".join(
                f"{n} from `{s}` (ledger {plan['sweep_states'][s]})"
                for s, n in plan["sweeps"].items()
            )
            lines.append(
                f"- `{plan['plan']}`: {plan['observed_runs']}/{plan['expected_runs']} "
                f"runs ({sources}): {plan['sweep_label']}"
            )
        lines.append("")
    lines += [
        f"## Table 1 ({comparison['labels']['table1']})",
        "",
        "| T | Model | Ours | Paper | Params |",
        "|---|---|---|---|---|",
    ]
    for block in comparison["table1"]:
        for name, row in block["rows"].items():
            lines.append(
                f"| {block['T']} | {name} | {_cell(row)} | {_paper(row)} | "
                f"{row['trainable_parameters']:,} |"
            )
    lines += ["", "QuLTSF vs the best linear baseline (same budget):", ""]
    for block in comparison["table1"]:
        lines.append(
            f"- T={block['T']}: vs {block['best_baseline']}: "
            f"{_verdict(block['vs_best_baseline'])}"
        )
    if comparison["budget"]:
        lines += [
            "",
            f"## Linear baselines under LTSF-Linear defaults ({comparison['labels']['budget']})",
            "",
            "| T | Model | Ours | Paper (Zeng et al.) |",
            "|---|---|---|---|",
        ]
        for block in comparison["budget"]:
            for name, row in block["rows"].items():
                lines.append(
                    f"| {block['T']} | {name} | {_cell(row)} | {_paper(row)} |"
                )
        lines += ["", "QuLTSF vs the best LTSF-default baseline:", ""]
        for block in comparison["budget"]:
            lines.append(
                f"- T={block['T']}: vs {block['best_baseline']}: "
                f"{_verdict(block['vs_best_baseline'])}"
            )
    if comparison["ablations"]:
        lines += [
            "",
            f"## Ablations ({comparison['labels']['ablations']})",
            "",
            "| T | Control | Ours | Params | QuLTSF vs control |",
            "|---|---|---|---|---|",
        ]
        for block in comparison["ablations"]:
            for name, row in block["rows"].items():
                lines.append(
                    f"| {block['T']} | {name} | {_cell(row)} | "
                    f"{row['trainable_parameters']:,} | {_verdict(row['qultsf_vs_control'])} |"
                )
    if comparison["lookback"]:
        lines += [
            "",
            f"## Lookback ({comparison['labels']['lookback']}), test MSE",
            "",
            "| T | L | qultsf | linear | nlinear | dlinear |",
            "|---|---|---|---|---|---|",
        ]
        for block in comparison["lookback"]:
            cells = [
                f"{block['rows'][name]['mse_mean']:.4f}"
                if name in block["rows"]
                else "-"
                for name in ("qultsf", *TABLE1_BASELINES)
            ]
            lines.append(f"| {block['T']} | {block['L']} | " + " | ".join(cells) + " |")
    if comparison["photonic"]:
        lines += [
            "",
            f"## Photonic adaptation ({comparison['labels']['merlin']})",
            "",
            "| T | Model | Ours | Params |",
            "|---|---|---|---|",
        ]
        for block in comparison["photonic"]:
            for name, row in block["rows"].items():
                lines.append(
                    f"| {block['T']} | {name} | {_cell(row)} | "
                    f"{row['trainable_parameters']:,} |"
                )
        lines += [
            "",
            "Each photonic model against QuLTSF and the best linear model:",
            "",
        ]
        for block in comparison["photonic"]:
            for name, row in block["rows"].items():
                lines.append(
                    f"- T={block['T']}, {name}: vs QuLTSF {_verdict(row['vs_qultsf'])}; "
                    f"vs {block['best_baseline']} {_verdict(row['vs_best_baseline'])}"
                )
            if "trained_vs_no_interferometer" in block:
                lines.append(
                    f"- T={block['T']}, trained vs no interferometer: "
                    f"{_verdict(block['trained_vs_no_interferometer'])}"
                )
    return "\n".join(lines) + "\n"


def plan_coverage(runs: dict[str, dict], plan_paths: list[Path]) -> dict:
    """Compare the included runs with the planned run IDs of each plan.

    Parameters
    ----------
    runs : dict[str, dict]
        Included runs from :func:`collect_runs`.
    plan_paths : list[pathlib.Path]
        Sweep plan files (``configs/sweeps/*.json``).

    Returns
    -------
    dict
        Per plan: expected and observed run counts, missing run IDs, how
        many included runs came from each sweep folder and each folder's
        ledger state, and a ``sweep_label``: ``COMPLETE SWEEP`` only when no
        run is missing and every contributing sweep reached COMPLETED,
        otherwise ``PARTIAL SWEEP``. Also ``unplanned`` run IDs and
        ``all_planned_runs_completed`` over all plans.
    """
    planned_ids: set[str] = set()
    plans = []
    for plan_path in plan_paths:
        plan = json.loads(plan_path.read_text())
        ids = {run["run_id"] for run in expand_plan(plan)}
        planned_ids |= ids
        observed = ids & runs.keys()
        sweeps = sorted({runs[run_id]["sweep"] for run_id in observed})
        plans.append(
            {
                "plan": plan_path.as_posix(),
                "expected_runs": len(ids),
                "observed_runs": len(observed),
                "missing": sorted(ids - runs.keys()),
                "sweeps": {
                    sweep: sum(runs[run_id]["sweep"] == sweep for run_id in observed)
                    for sweep in sweeps
                },
                "sweep_states": {
                    sweep: next(
                        runs[run_id]["sweep_status"]
                        for run_id in sorted(observed)
                        if runs[run_id]["sweep"] == sweep
                    )
                    for sweep in sweeps
                },
            }
        )
        record = plans[-1]
        unfinished = sorted(
            sweep
            for sweep, state in record["sweep_states"].items()
            if state != "COMPLETED"
        )
        record["all_planned_runs_completed"] = not record["missing"]
        # A sweep whose coordinator never reached a terminal state cannot be
        # labelled complete, even when a retry sweep supplies its missing runs.
        record["sweep_label"] = (
            "COMPLETE SWEEP"
            if not record["missing"] and not unfinished
            else "PARTIAL SWEEP"
        )
        record["unfinished_sweeps"] = unfinished
    unplanned = sorted(runs.keys() - planned_ids)
    return {
        "all_planned_runs_completed": not unplanned
        and all(plan["all_planned_runs_completed"] for plan in plans),
        "plans": plans,
        "unplanned": unplanned,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, action="append", default=[])
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--out-dir", type=Path)
    output.add_argument("--verify", type=Path)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args(argv)

    runs = collect_runs(args.sweep_dir)
    comparison = build_comparison(runs)
    comparison["sweep_dirs"] = [path.as_posix() for path in args.sweep_dir]
    comparison["plan_coverage"] = plan_coverage(runs, args.plan)
    coverage = comparison["plan_coverage"]
    for plan in coverage["plans"]:
        print(
            f"{plan['plan']}: {plan['observed_runs']}/{plan['expected_runs']} "
            f"runs from {plan['sweeps']}; missing {plan['missing'] or 'none'}; "
            f"sweep states {plan['sweep_states']}; {plan['sweep_label']}"
        )
        label_key = Path(plan["plan"]).stem
        if plan["sweep_label"] == "PARTIAL SWEEP" and label_key in comparison["labels"]:
            retries = sorted(set(plan["sweeps"]) - set(plan["unfinished_sweeps"]))
            comparison["labels"][label_key] += (
                f"; PARTIAL SWEEP: {', '.join(plan['unfinished_sweeps'])} did not finish, "
                f"remaining runs from {', '.join(retries)}"
            )
    print(
        f"unplanned: {coverage['unplanned'] or 'none'}; "
        f"all planned runs completed: {coverage['all_planned_runs_completed']}"
    )
    status = 0 if coverage["all_planned_runs_completed"] else 1

    if args.verify is not None:
        recomputed = json.loads(json.dumps(comparison))
        stored = json.loads(args.verify.read_text())
        if recomputed != stored:
            changed = sorted(
                key
                for key in recomputed.keys() | stored.keys()
                if recomputed.get(key) != stored.get(key)
            )
            print(f"VERIFY FAILED: {args.verify} differs in {changed}")
            return 1
        print(f"VERIFY OK: {args.verify} matches the recomputed comparison")
        return status

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    (args.out_dir / "comparison.md").write_text(render_markdown(comparison))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name in ("comparison.json", "comparison.md"):
        shutil.copyfile(args.out_dir / name, args.results_dir / name)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
