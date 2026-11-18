"""Summarise a QPE_GNN sweep from its structured run artifacts.

Reads ``sweep_status.json`` and, for every run it lists, the run's
``run_status.json``, ``metrics.json``, ``config_snapshot.json`` and
``test_outputs.npz``. A run is included only when its status is COMPLETED,
its metrics file matches the SHA-256 recorded in its status, and its test
metric is finite. Everything else is listed as excluded with the reason.

For every candidate the summary reports the per-seed test metric at full
precision, the mean, the sample standard deviation and the run counts. For
every pair of candidates it applies the plan's tie rule on the seeds both
have completed:

* regression (``test_mae``): two-sided Wilcoxon signed-rank test on the
  per-graph absolute errors averaged over those seeds;
* classification (``test_acc``): exact two-sided McNemar test on the test
  predictions pooled over those seeds.

A pair has a winner only when the mean margin exceeds twice the larger seed
standard deviation and the test gives p < 0.05.

Usage (from ``papers/QPE_GNN``)::

    python utils/summarise_sweep.py --sweep-dir outdir/sweep_zinc_reduced \
        --plan configs/sweeps/zinc_reduced.json
    python utils/summarise_sweep.py --sweep-dir outdir/sweep_zinc_reduced \
        --plan configs/sweeps/zinc_reduced.json --verify
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from lib.experiment_logging import sha256, write_json  # noqa: E402

SIGNIFICANCE_LEVEL = 0.05
MARGIN_STD_MULTIPLE = 2.0


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run(sweep_dir: Path, entry: dict, metric: str) -> tuple[dict | None, str]:
    """Return the included run record, or ``None`` and the exclusion reason."""
    if entry["status"] != "COMPLETED":
        return None, f"sweep ledger status {entry['status']}"
    run_dir = sweep_dir / entry["run_dir"]
    run_status = _read_json(run_dir / "run_status.json")
    if run_status["status"] != "COMPLETED":
        return None, f"run_status {run_status['status']}"
    metrics_path = run_dir / run_status["metrics_path"]
    if sha256(metrics_path) != run_status["metrics_sha256"]:
        return None, "metrics.json does not match the SHA-256 in run_status.json"
    metrics = _read_json(metrics_path)
    if f"test_{metrics['test_metric']['name']}" != metric:
        return (
            None,
            f"metric {metrics['test_metric']['name']} is not the plan's {metric}",
        )
    value = float(metrics["test_metric"]["value"])
    if not math.isfinite(value):
        return None, f"non-finite test metric {value}"
    config = _read_json(run_dir / "config_snapshot.json")
    outputs = dict(np.load(run_dir / metrics["test_outputs_path"]))
    return {
        "run_id": entry["run_id"],
        "seed": int(entry["seed"]),
        "run_dir": entry["run_dir"],
        "value": value,
        "best_epoch": int(metrics["best_epoch"]),
        "epochs_completed": int(metrics["epochs_completed"]),
        "num_params": int(metrics["num_params"]),
        "wall_clock_s": float(metrics["wall_clock_s"]),
        "resumed_from": metrics.get("resumed_from"),
        "epochs": int(config["epochs"]),
        "outputs": outputs,
    }, ""


def _sample_std(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _paired_test(metric: str, runs_a: dict[int, dict], runs_b: dict[int, dict]) -> dict:
    """Apply the plan's significance test on the seeds both candidates completed."""
    common_seeds = sorted(set(runs_a) & set(runs_b))
    if not common_seeds:
        return {"test": None, "seeds": [], "p_value": None}
    if metric == "test_mae":
        errors_a = np.mean(
            [runs_a[s]["outputs"]["absolute_errors"] for s in common_seeds], axis=0
        )
        errors_b = np.mean(
            [runs_b[s]["outputs"]["absolute_errors"] for s in common_seeds], axis=0
        )
        if np.array_equal(errors_a, errors_b):
            p_value = 1.0
        else:
            p_value = float(stats.wilcoxon(errors_a, errors_b).pvalue)
        return {
            "test": "wilcoxon_signed_rank_seed_averaged_absolute_errors",
            "seeds": common_seeds,
            "num_pairs": int(errors_a.size),
            "p_value": p_value,
        }
    if metric == "test_acc":
        labels_a = np.concatenate(
            [runs_a[s]["outputs"]["labels"] for s in common_seeds]
        )
        labels_b = np.concatenate(
            [runs_b[s]["outputs"]["labels"] for s in common_seeds]
        )
        if not np.array_equal(labels_a, labels_b):
            raise ValueError("candidates were evaluated on different test labels")
        correct_a = np.concatenate(
            [
                runs_a[s]["outputs"]["predictions"] == runs_a[s]["outputs"]["labels"]
                for s in common_seeds
            ]
        )
        correct_b = np.concatenate(
            [
                runs_b[s]["outputs"]["predictions"] == runs_b[s]["outputs"]["labels"]
                for s in common_seeds
            ]
        )
        only_a = int(np.sum(correct_a & ~correct_b))
        only_b = int(np.sum(~correct_a & correct_b))
        discordant = only_a + only_b
        p_value = (
            1.0
            if discordant == 0
            else float(stats.binomtest(min(only_a, only_b), discordant, 0.5).pvalue)
        )
        return {
            "test": "exact_mcnemar_pooled_predictions",
            "seeds": common_seeds,
            "num_pairs": int(correct_a.size),
            "only_a_correct": only_a,
            "only_b_correct": only_b,
            "p_value": p_value,
        }
    raise ValueError(f"no significance test defined for metric {metric}")


def build_summary(sweep_dirs: Path | list[Path], plan: dict) -> dict:
    """Build the sweep summary without writing anything.

    Parameters
    ----------
    sweep_dirs : pathlib.Path | list[pathlib.Path]
        Sweep folder holding ``sweep_status.json``, or several folders when
        one plan ran as several launches (``run_sweep.py --run-seeds``). With
        several folders, run paths are prefixed with their folder name.
    plan : dict
        Parsed sweep plan.

    Returns
    -------
    dict
        Candidates, pairwise comparisons, excluded runs and the sweep verdict.

    Raises
    ------
    ValueError
        If the ledger names an unplanned run, a run appears twice, or runs of
        one candidate differ in epochs or parameter count.
    """
    metric = plan["metric"]
    minimise = plan["direction"] == "minimize"
    single_folder = isinstance(sweep_dirs, Path)
    folders = [sweep_dirs] if single_folder else list(sweep_dirs)
    ledgers = [_read_json(folder / "sweep_status.json") for folder in folders]
    planned = {
        f"{entry['name']}_seed{seed}": entry["name"]
        for entry in plan["entries"]
        for seed in entry["seeds"]
    }
    seen: set[str] = set()
    included: dict[str, dict[int, dict]] = {e["name"]: {} for e in plan["entries"]}
    excluded = []
    for folder, ledger in zip(folders, ledgers, strict=True):
        for entry in ledger["runs"]:
            if entry["run_id"] not in planned:
                raise ValueError(f"{entry['run_id']} is not in the plan")
            if entry["run_id"] in seen:
                raise ValueError(f"{entry['run_id']} appears twice in the ledgers")
            seen.add(entry["run_id"])
            run, reason = _load_run(folder, entry, metric)
            if run is None:
                excluded.append({"run_id": entry["run_id"], "reason": reason})
                continue
            if not single_folder:
                run["run_dir"] = f"{folder.name}/{run['run_dir']}"
            included[entry["candidate"]][run["seed"]] = run

    candidates = []
    for entry in plan["entries"]:
        runs = included[entry["name"]]
        for field in ("epochs", "num_params"):
            if len({run[field] for run in runs.values()}) > 1:
                raise ValueError(f"{entry['name']}: runs differ in {field}")
        values = [runs[seed]["value"] for seed in sorted(runs)]
        complete = sorted(runs) == sorted(entry["seeds"])
        candidates.append(
            {
                "candidate": entry["name"],
                "config": entry["config"],
                "status": "COMPLETE" if complete else "PARTIAL",
                "expected_runs": len(entry["seeds"]),
                "observed_runs": len(runs),
                "missing_seeds": sorted(set(entry["seeds"]) - set(runs)),
                "seeds": sorted(runs),
                "values": values,
                "mean": float(np.mean(values)) if values else None,
                "std": _sample_std(values),
                "num_params": next(iter(runs.values()))["num_params"] if runs else None,
                "epochs": next(iter(runs.values()))["epochs"] if runs else None,
                "best_epochs": [runs[seed]["best_epoch"] for seed in sorted(runs)],
                "mean_wall_clock_s": float(
                    np.mean([run["wall_clock_s"] for run in runs.values()])
                )
                if runs
                else None,
                "resumed_runs": sorted(
                    run["run_id"] for run in runs.values() if run["resumed_from"]
                ),
                "run_dirs": [runs[seed]["run_dir"] for seed in sorted(runs)],
            }
        )

    by_name = {record["candidate"]: record for record in candidates}
    comparisons = []
    for name_a, name_b in itertools.combinations(by_name, 2):
        record_a, record_b = by_name[name_a], by_name[name_b]
        test = _paired_test(metric, included[name_a], included[name_b])
        verdict = "insufficient runs"
        margin = None
        if record_a["std"] is not None and record_b["std"] is not None:
            margin = record_b["mean"] - record_a["mean"]
            threshold = MARGIN_STD_MULTIPLE * max(record_a["std"], record_b["std"])
            b_better = margin < 0 if minimise else margin > 0
            if abs(margin) > threshold and test["p_value"] < SIGNIFICANCE_LEVEL:
                verdict = f"{name_b} better" if b_better else f"{name_a} better"
            else:
                verdict = "not distinguishable"
        comparisons.append(
            {
                "a": name_a,
                "b": name_b,
                "mean_margin_b_minus_a": margin,
                "margin_threshold": None
                if margin is None
                else MARGIN_STD_MULTIPLE * max(record_a["std"], record_b["std"]),
                **test,
                "verdict": verdict,
            }
        )

    complete = not excluded and all(r["status"] == "COMPLETE" for r in candidates)
    expected = sum(r["expected_runs"] for r in candidates)
    observed = sum(r["observed_runs"] for r in candidates)
    summary = {
        "schema_version": 1,
        "sweep_id": plan["sweep_id"],
        "verdict": "COMPLETE SWEEP" if complete else "PARTIAL SWEEP",
        "metric": metric,
        "direction": plan["direction"],
        "tie_rule": plan["tie_rule"],
        "expected_runs": expected,
        "observed_runs": observed,
    }
    if single_folder:
        summary["code_commit"] = ledgers[0]["code"].get("commit")
        summary["code_dirty"] = ledgers[0]["code"].get("dirty")
    else:
        summary["launches"] = [
            {
                "sweep_dir": folder.name,
                "status": ledger["status"],
                "run_seeds": ledger.get("run_seeds"),
                "code_commit": ledger["code"].get("commit"),
                "code_dirty": ledger["code"].get("dirty"),
            }
            for folder, ledger in zip(folders, ledgers, strict=True)
        ]
    return {
        **summary,
        "candidates": candidates,
        "comparisons": comparisons,
        "excluded_runs": excluded,
    }


def _write_csv(path: Path, summary: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "candidate",
                "status",
                "observed_runs",
                "expected_runs",
                f"mean_{summary['metric']}",
                f"std_{summary['metric']}",
                "num_params",
                "epochs",
                "mean_wall_clock_s",
            ]
        )
        for record in summary["candidates"]:
            writer.writerow(
                [
                    record["candidate"],
                    record["status"],
                    record["observed_runs"],
                    record["expected_runs"],
                    record["mean"],
                    record["std"],
                    record["num_params"],
                    record["epochs"],
                    record["mean_wall_clock_s"],
                ]
            )


def summarise_sweep(
    sweep_dirs: Path | list[Path], plan_path: Path, output_dir: Path | None = None
) -> Path:
    """Write ``sweep_summary.json`` and ``sweep_summary.csv``.

    Parameters
    ----------
    sweep_dirs : pathlib.Path | list[pathlib.Path]
        Sweep folder, or the launch folders of one plan.
    plan_path : pathlib.Path
        Sweep plan file.
    output_dir : pathlib.Path | None
        Folder for the two files. Default value is None, which writes them
        into the sweep folder; it is required with several folders.

    Returns
    -------
    pathlib.Path
        Path of ``sweep_summary.json``.

    Raises
    ------
    ValueError
        If several folders are given without ``output_dir``.
    """
    if output_dir is None and not isinstance(sweep_dirs, Path):
        raise ValueError("several sweep folders need an output_dir")
    summary = build_summary(sweep_dirs, _read_json(plan_path))
    target = output_dir or sweep_dirs
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "sweep_summary.json"
    write_json(summary_path, summary)
    _write_csv(target / "sweep_summary.csv", summary)
    return summary_path


def verify_summary(
    sweep_dirs: Path | list[Path], plan_path: Path, summary_path: Path
) -> bool:
    """Recompute the summary and compare it with a written one, read-only."""
    recomputed = json.loads(
        json.dumps(build_summary(sweep_dirs, _read_json(plan_path)))
    )
    return recomputed == _read_json(summary_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        action="append",
        required=True,
        help="Sweep folder; repeat it for the launch folders of one plan",
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write the summary here instead of the sweep folder (e.g. results/)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Recompute and compare with the existing summary; write nothing",
    )
    args = parser.parse_args(argv)

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_DIR / path

    folders = [resolve(path) for path in args.sweep_dir]
    sweep_dirs = folders[0] if len(folders) == 1 else folders
    plan_path = resolve(args.plan)
    output_dir = resolve(args.output_dir) if args.output_dir else None
    if args.verify:
        if output_dir is None and len(folders) > 1:
            parser.error("--verify with several --sweep-dir needs --output-dir")
        summary_path = (output_dir or folders[0]) / "sweep_summary.json"
        if verify_summary(sweep_dirs, plan_path, summary_path):
            print(f"VERIFIED {summary_path}")
            return 0
        print(f"MISMATCH {summary_path}")
        return 1
    summary_path = summarise_sweep(sweep_dirs, plan_path, output_dir)
    summary = _read_json(summary_path)
    print(
        f"{summary['verdict']}: {summary['observed_runs']}/{summary['expected_runs']}"
    )
    for record in summary["candidates"]:
        print(
            f"  {record['candidate']}: {record['mean']} ± {record['std']} "
            f"({record['observed_runs']}/{record['expected_runs']}, {record['status']})"
        )
    for comparison in summary["comparisons"]:
        print(
            f"  {comparison['a']} vs {comparison['b']}: p={comparison['p_value']} "
            f"-> {comparison['verdict']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
