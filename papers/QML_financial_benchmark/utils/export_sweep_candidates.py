"""Export the complete candidate ranking of every verified sweep into ``results/``.

``outdir/`` is disposable and git-ignored, so the README links the durable
tables written here instead of raw run directories. For each sweep in
``utils/verify_sweeps.py::SWEEP_PLANS`` this script writes:

- ``results/sweep_candidates/<sweep>.csv``: one row per (variant, seed,
  selection rule, model family, candidate) with the declared selection score,
  its within-run rank, whether it ties the best score within 1e-4, and whether
  it was selected (Tasks 1-2). Task 3 sweeps have no candidate grid (each
  model is tuned per refit by Optuna), so their rows are the test QLIKE of
  every model per ticker, ranked ascending.
- ``results/directional_per_seed_test.csv``: fold-5 test metrics of every
  selected Task 1 model (QNN, ANN, parameter-matched ANN, logistic regression,
  majority class) per sweep, variant and seed, so per-seed results stay
  available without ``outdir/``.
- ``results/sweep_index.csv``: per sweep, the declared metric, direction,
  split, checkpoint rule, aggregation, expected and completed run counts,
  completeness, number of selections decided by the tie rule, and the
  candidate file.

Selections are recomputed with the same rule as ``verify_sweeps.py`` and must
equal the recorded ones; the script exits non-zero otherwise.

Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/export_sweep_candidates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_DIR / "utils")]
from verify_sweeps import (  # noqa: E402
    SWEEP_PLANS,
    TIE_TOLERANCE,
    find_sweep,
    pick,
    variant_of,
)

RESULTS = PROJECT_DIR / "results"
OUTPUT_DIR = RESULTS / "sweep_candidates"

PROTOCOL_TEXT = {
    "directional": {
        "metric": "mean validation AUC over folds 1-4 (leak_free) and 1-5 (paper); reported: fold-5 test AUC",
        "direction": "maximize",
        "split": "5-fold walk-forward; fold 5 test 2020-01-02..2021-12-31",
        "checkpoint": "best early-stop-AUC epoch per fold",
        "aggregation": "selection per seed; test metric mean +/- sample std over seeds",
    },
    "trading": {
        "metric": "model-select AUC (auc rule) and tau-slice Sharpe (tau_sharpe rule); reported: whole-regime test Sharpe",
        "direction": "maximize",
        "split": "history_then_regime (Fig. 6)",
        "checkpoint": "best early-stop-AUC epoch",
        "aggregation": "grid on seed 0 (stage 1); selected candidates retrained on seeds 1-4 (stage 2); mean +/- std over 5 seeds",
    },
    "volatility": {
        "metric": "test QLIKE per model (Optuna 50 trials per refit on validation QLIKE)",
        "direction": "minimize",
        "split": "expanding window, 720 initial days, refit every 120 days",
        "checkpoint": "n/a (refit per block)",
        "aggregation": "single Optuna sampler seed; Diebold-Mariano on shared dates",
    },
}


def ranked(scores: pd.DataFrame, column: str, selected: str) -> pd.DataFrame:
    """Rank ``scores`` by ``column`` (descending) and flag ties and the selected candidate."""
    best = scores[column].max()
    return scores.assign(
        rank=scores[column].rank(ascending=False, method="min").astype(int),
        tied_with_best=scores[column] >= best - TIE_TOLERANCE,
        selected=scores.index == selected,
    )


def directional_rows(run_dir: Path, metrics: dict, config: dict) -> list[pd.DataFrame]:
    candidates = pd.read_csv(run_dir / "candidates.csv")
    frames = []
    for protocol, folds in config["directional"]["selection"]["protocols"].items():
        for family in ("qnn", "ann"):
            scores = (
                candidates[
                    (candidates["model"] == family) & candidates["fold"].isin(folds)
                ]
                .groupby("name", sort=False)
                .agg(
                    family_detail=("family", "first"),
                    parameters=("parameters", "first"),
                    order=("order", "first"),
                    score=("validation_auc", "mean"),
                )
            )
            recorded = metrics["selection"][f"{protocol}:{family}"]["candidate"]
            if pick(scores, "score") != recorded:
                raise ValueError(f"{run_dir}: {protocol}:{family} selection mismatch")
            frames.append(
                ranked(scores, "score", recorded).assign(rule=protocol, family=family)
            )
    return frames


def trading_rows(run_dir: Path, metrics: dict) -> list[pd.DataFrame]:
    candidates = pd.read_csv(run_dir / "candidates.csv").set_index("name")
    frames = []
    for rule, column in (("auc", "model_select_auc"), ("tau_sharpe", "tau_sharpe")):
        for family in ("qlstm", "lstm"):
            scores = candidates[candidates["model"] == family][
                ["parameters", "order", column]
            ].rename(columns={column: "score"})
            recorded = metrics["selection"][f"{rule}:{family}"]["candidate"]
            if pick(scores, "score") != recorded:
                raise ValueError(f"{run_dir}: {rule}:{family} selection mismatch")
            frames.append(
                ranked(scores, "score", recorded).assign(rule=rule, family=family)
            )
    return frames


def volatility_rows(metrics: dict) -> list[pd.DataFrame]:
    scores = (
        pd.DataFrame(metrics["test"]).T[["qlike"]].rename(columns={"qlike": "score"})
    )
    best = scores["score"].min()
    return [
        scores.assign(
            rank=scores["score"].rank(method="min").astype(int),
            tied_with_best=scores["score"] <= best + TIE_TOLERANCE,
            selected=False,
            rule="test_qlike",
            family=scores.index,
        )
    ]


def export_sweep(prefix: str) -> dict:
    task, variants, seeds, _ = SWEEP_PLANS[prefix]
    sweep_dir = find_sweep(prefix)
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    completed = [run for run in status["runs"] if run["status"] == "COMPLETED"]
    frames = []
    for run in completed:
        run_dir = sweep_dir / run["run_dir"]
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if task == "directional":
            config = json.loads((run_dir / "config_snapshot.json").read_text())
            run_frames = directional_rows(run_dir, metrics, config)
        elif task == "trading":
            run_frames = trading_rows(run_dir, metrics)
        else:
            run_frames = volatility_rows(metrics)
        for frame in run_frames:
            frames.append(
                frame.rename_axis("candidate")
                .reset_index()
                .assign(
                    variant=variant_of(run), seed=run["seed"], run_dir=run["run_dir"]
                )
            )
    table = pd.concat(frames, ignore_index=True)
    leading = ["variant", "seed", "rule", "family", "candidate", "parameters", "score"]
    leading = [column for column in leading if column in table.columns]
    rest = [column for column in table.columns if column not in leading]
    table = table[leading + rest].sort_values(
        ["variant", "seed", "rule", "family", "rank"]
    )
    output = OUTPUT_DIR / f"{prefix}.csv"
    table.to_csv(output, index=False)
    selected = table[table["selected"]]
    decided_by_tie = (
        int(
            sum(
                group["tied_with_best"].sum() > 1
                for _, group in table.groupby(["variant", "seed", "rule", "family"])
            )
        )
        if task != "volatility"
        else 0
    )
    planned = len(variants) * len(seeds)
    return {
        "sweep": sweep_dir.name,
        "task": task,
        **PROTOCOL_TEXT[task],
        "expected_runs": planned,
        "completed_runs": len(completed),
        "completeness": "COMPLETE" if len(completed) == planned else "PARTIAL",
        "candidates_per_run": int(table.groupby(["variant", "seed"]).size().iloc[0]),
        "selections": len(selected),
        "selections_decided_by_tie_rule": decided_by_tie,
        "candidate_table": str(output.relative_to(PROJECT_DIR)),
    }


def directional_test_rows(prefix: str) -> list[dict]:
    """Per-seed fold-5 test metrics of every selected Task 1 model in one sweep."""
    sweep_dir = find_sweep(prefix)
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    rows = []
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads((sweep_dir / run["run_dir"] / "metrics.json").read_text())
        for model, values in metrics["test"].items():
            rows.append(
                {
                    "sweep_id": prefix,
                    "variant": variant_of(run),
                    "seed": run["seed"],
                    "model": model,
                    "candidate": values.get("candidate", model),
                    "parameters": values.get("parameters"),
                    "auc": values["auc"],
                    "accuracy": values["accuracy"],
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "n": values["n"],
                }
            )
    return rows


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame([export_sweep(prefix) for prefix in SWEEP_PLANS])
    index.insert(0, "sweep_id", list(SWEEP_PLANS))
    index.to_csv(RESULTS / "sweep_index.csv", index=False)
    pd.DataFrame(
        [
            row
            for prefix, plan in SWEEP_PLANS.items()
            if plan[0] == "directional"
            for row in directional_test_rows(prefix)
        ]
    ).to_csv(RESULTS / "directional_per_seed_test.csv", index=False)
    print(
        index[
            [
                "sweep",
                "completed_runs",
                "expected_runs",
                "selections",
                "selections_decided_by_tie_rule",
            ]
        ].to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
