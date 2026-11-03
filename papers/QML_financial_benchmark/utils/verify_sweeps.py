"""Independent Phase 3 verification of every recorded sweep.

For each sweep in ``SWEEP_PLANS`` this script:

1. rebuilds the planned (variant, seed) set and compares it with the observed
   COMPLETED runs in ``sweep_status.json`` (missing, extra, duplicate);
2. checks each run's ``config_snapshot.json`` (seed, config-defined task and
   ticker or regime) against the plan;
3. recomputes the in-run model selection from the raw ``candidates.csv``
   (Task 1: mean validation AUC over the protocol's folds; Task 2: model-select
   AUC and tau-slice Sharpe; ties within 1e-4 go to fewer parameters, then grid
   order) and compares it with ``metrics.json``;
4. runs the experiment-logging validator on the sweep and all its runs.

Writes ``results/sweep_verification.json``; exits non-zero on any mismatch.

Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/verify_sweeps.py --validator <path to validate_logging.py>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR)]
from lib.tasks.directional import matched_ann_width  # noqa: E402

SWEEPS_DIR = PROJECT_DIR / "outdir" / "sweeps"
TIE_TOLERANCE = 1e-4

# sweep name prefix -> (task, {variant: dataset key value}, seeds, dataset key checked)
SWEEP_PLANS = {
    "e1_kchol_2": ("directional", {"main": "KCHOL.IS"}, [0, 1, 2, 3, 4], "ticker"),
    "e1_kchol_sameday": (
        "directional",
        {"main": "KCHOL.IS"},
        [0, 1, 2, 3, 4],
        "ticker",
    ),
    "e3_aapl": ("directional", {"main": "AAPL"}, [0, 1, 2, 3, 4], "ticker"),
    "e2_spx": ("directional", {"main": "^GSPC"}, [0, 1, 2, 3, 4], "ticker"),
    "e1_turkish_rest": (
        "directional",
        {t.split(".")[0]: t for t in ["GARAN.IS", "TCELL.IS", "TUPRS.IS", "ULKER.IS"]},
        [0, 1, 2, 3, 4],
        "ticker",
    ),
    "e3_us_rest_seed0": (
        "directional",
        {t: t for t in ["BA", "GILD", "DVN", "LNC"]},
        [0],
        "ticker",
    ),
    "e3_us_rest_seed12": (
        "directional",
        {t: t for t in ["BA", "GILD", "DVN", "LNC"]},
        [1, 2],
        "ticker",
    ),
    "e3_tablev_aapl": ("directional", {"main": "AAPL"}, [0, 1, 2, 3, 4], "ticker"),
    "e6_e7_seed0": (
        "volatility",
        {
            f"{e}_{t.split('.')[0].lstrip('^')}": t
            for e in ("e6", "e7")
            for t in ["^GSPC", "AAPL", "GARAN.IS", "KCHOL.IS"]
        },
        [0],
        "ticker",
    ),
    "e6_garch_sigma2": (
        "volatility",
        {
            t.split(".")[0].lstrip("^"): t
            for t in ["^GSPC", "AAPL", "GARAN.IS", "KCHOL.IS"]
        },
        [0],
        "ticker",
    ),
    "e6_merlin": (
        "volatility",
        {
            t.split(".")[0].lstrip("^"): t
            for t in ["^GSPC", "AAPL", "GARAN.IS", "KCHOL.IS"]
        },
        [0],
        "ticker",
    ),
    "e2_merlin": ("directional", {"main": "^GSPC"}, [0, 1, 2, 3, 4], "ticker"),
    "e3_tablev_merlin": ("directional", {"main": "AAPL"}, [0, 1, 2, 3, 4], "ticker"),
    "e5_stage1": ("trading", {r: r for r in ["F1", "F2", "F3", "F4"]}, [0], "regime"),
    "e5_stage2_F1F3": (
        "trading",
        {r: r for r in ["F1", "F2", "F3"]},
        [1, 2, 3, 4],
        "regime",
    ),
    "e5_stage2_F4": ("trading", {"F4": "F4"}, [1, 2, 3, 4], "regime"),
}


def find_sweep(prefix: str) -> Path:
    """The single sweep directory whose name is ``<prefix><timestamp>`` or ``<prefix>_<timestamp>``."""
    matches = [
        path
        for path in SWEEPS_DIR.iterdir()
        if path.name.startswith(prefix)
        and path.name[len(prefix) :].lstrip("_")[:1].isdigit()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one sweep for {prefix!r}, found {[m.name for m in matches]}"
        )
    return matches[0]


def variant_of(run: dict) -> str:
    """Variant folder for multi-variant sweeps; ``main`` for single-variant ones (e.g. ``seed0/run_*``)."""
    parts = Path(run["run_dir"]).parts
    return parts[0] if len(parts) == 3 else "main"


def pick(scores: pd.DataFrame, column: str) -> str:
    """Highest ``column``; ties within the tolerance go to fewer parameters, then grid order."""
    tied = scores[scores[column] >= scores[column].max() - TIE_TOLERANCE]
    return str(tied.sort_values(["parameters", "order"]).index[0])


def recompute_directional(run_dir: Path, config: dict) -> dict[str, str]:
    """Selections per protocol and family from ``candidates.csv``."""
    candidates = pd.read_csv(run_dir / "candidates.csv")
    selections = {}
    for protocol, folds in config["directional"]["selection"]["protocols"].items():
        for family in ("qnn", "ann"):
            scores = (
                candidates[
                    (candidates["model"] == family) & candidates["fold"].isin(folds)
                ]
                .groupby("name", sort=False)
                .agg(
                    mean_auc=("validation_auc", "mean"),
                    parameters=("parameters", "first"),
                    order=("order", "first"),
                )
            )
            selections[f"{protocol}:{family}"] = pick(scores, "mean_auc")
    return selections


def recompute_trading(run_dir: Path) -> dict[str, str]:
    """Selections per rule and family, plus the parameter-matched LSTM, from ``candidates.csv``."""
    candidates = pd.read_csv(run_dir / "candidates.csv").set_index("name")
    selections = {
        f"{rule}:{family}": pick(candidates[candidates["model"] == family], column)
        for rule, column in (("auc", "model_select_auc"), ("tau_sharpe", "tau_sharpe"))
        for family in ("qlstm", "lstm")
    }
    lstm = candidates[candidates["model"] == "lstm"]
    for rule in ("auc", "tau_sharpe"):
        budget = candidates.loc[selections[f"{rule}:qlstm"], "parameters"]
        gaps = lstm.assign(gap=(lstm["parameters"] - budget).abs())
        selections[f"{rule}:lstm_parameter_matched"] = str(
            gaps.sort_values(["gap", "order"]).index[0]
        )
    return selections


def recompute_matched_ann(metrics: dict, n_features: int) -> dict[str, str]:
    """Matched-ANN candidate names implied by each protocol's selected QNN parameter count."""
    return {
        f"{protocol}:ann_parameter_matched": f"ann_matched_{matched_ann_width(metrics['selection'][f'{protocol}:qnn']['parameters'], n_features)}"
        for protocol in ("leak_free", "paper")
    }


def solve_input_width(parameters: int, hidden: list[int]) -> int:
    """Input dimension ``d`` of an MLP ``[d - hidden - 1]`` with the given parameter count."""
    after_first = sum(a * b + b for a, b in zip(hidden, hidden[1:] + [1]))
    first_width = hidden[0]
    d = (parameters - after_first - first_width) / first_width
    if d != int(d):
        raise ValueError(
            f"No integer input width for {parameters} parameters and hidden {hidden}"
        )
    return int(d)


def verify_sweep(prefix: str, validator: Path) -> dict:
    task, variants, seeds, key = SWEEP_PLANS[prefix]
    sweep_dir = find_sweep(prefix)
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    planned = {(variant, seed) for variant in variants for seed in seeds}
    observed = [
        (variant_of(run), run["seed"])
        for run in status["runs"]
        if run["status"] == "COMPLETED"
    ]
    problems = []
    if len(observed) != len(set(observed)):
        problems.append("duplicate (variant, seed) runs")
    missing, extra = planned - set(observed), set(observed) - planned
    if missing:
        problems.append(f"missing runs {sorted(missing)}")
    if extra:
        problems.append(f"unplanned runs {sorted(extra)}")
    selection_checks = 0
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        run_dir = sweep_dir / run["run_dir"]
        variant = variant_of(run)
        config = json.loads((run_dir / "config_snapshot.json").read_text())
        if config["seed"] != run["seed"] or config["task"] != task:
            problems.append(f"{run['run_dir']}: config seed/task mismatch")
        if config[task]["dataset"][key] != variants.get(variant):
            problems.append(
                f"{run['run_dir']}: {key}={config[task]['dataset'][key]!r}, planned {variants.get(variant)!r}"
            )
        if task == "volatility":
            continue
        metrics = json.loads((run_dir / "metrics.json").read_text())
        recomputed = (
            recompute_directional(run_dir, config)
            if task == "directional"
            else recompute_trading(run_dir)
        )
        if task == "directional":
            n_features = int(
                pd.read_csv(run_dir / "candidates.csv")
                .query("model == 'ann'")
                .iloc[0]["parameters"]
            )
            hidden = [
                int(w)
                for w in str(
                    pd.read_csv(run_dir / "candidates.csv")
                    .query("model == 'ann'")
                    .iloc[0]["hidden"]
                ).split("-")
            ]
            n_features = solve_input_width(n_features, hidden)
            for selection_key, candidate in recompute_matched_ann(
                metrics, n_features
            ).items():
                selection_checks += 1
                if metrics["test"][selection_key]["candidate"] != candidate:
                    problems.append(
                        f"{run['run_dir']}: {selection_key} recorded {metrics['test'][selection_key]['candidate']}, recomputed {candidate}"
                    )
        for selection_key, candidate in recomputed.items():
            selection_checks += 1
            if metrics["selection"][selection_key]["candidate"] != candidate:
                problems.append(
                    f"{run['run_dir']}: {selection_key} recorded {metrics['selection'][selection_key]['candidate']}, recomputed {candidate}"
                )
    run_dirs = [
        arg
        for run in status["runs"]
        if run["status"] == "COMPLETED"
        for arg in ("--run-dir", str(sweep_dir / run["run_dir"]))
    ]
    validation = subprocess.run(
        [
            sys.executable,
            str(validator),
            *run_dirs,
            "--sweep-dir",
            str(sweep_dir),
            "--require-training",
            "--require-evaluation",
        ],
        capture_output=True,
        text=True,
    )
    if validation.returncode != 0:
        problems.append(
            f"logging validation failed: {validation.stdout.strip()} {validation.stderr.strip()}"
        )
    return {
        "sweep": sweep_dir.name,
        "sweep_status": status["status"],
        "planned_runs": len(planned),
        "completed_runs": len(observed),
        "coverage": "COMPLETE"
        if not missing and not extra and len(observed) == len(set(observed))
        else "PARTIAL",
        "selection_checks": selection_checks,
        "validator": validation.stdout.strip(),
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--validator",
        type=Path,
        required=True,
        help="experiment-logging validate_logging.py",
    )
    args = parser.parse_args(argv)
    report = {prefix: verify_sweep(prefix, args.validator) for prefix in SWEEP_PLANS}
    (PROJECT_DIR / "results").mkdir(exist_ok=True)
    (PROJECT_DIR / "results" / "sweep_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for result in report.values():
        print(
            f"{result['sweep']}: {result['coverage']} {result['completed_runs']}/{result['planned_runs']} | "
            f"selections checked {result['selection_checks']} | {result['validator']} | problems {result['problems'] or 'none'}"
        )
    return 1 if any(result["problems"] for result in report.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
