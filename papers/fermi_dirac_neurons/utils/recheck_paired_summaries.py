"""Independent recheck of every stored paired comparison against raw ``metrics.json``.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/recheck_paired_summaries.py \
        papers/fermi_dirac_neurons/outdir/sweeps

Read-only. It does not import ``utils/sweep_statistics.py`` or any
``utils/analyze_*.py``; it re-derives every value from the run artifacts with
its own code and compares with the stored summaries:

* ``results/e8_classical_data_summary.json`` (``comparisons``),
* ``results/e9_robustness_summary.json`` (``grid`` and ``finite_shots``),
* ``results/p4_learning_summary.json`` (``comparisons``),
* ``results/p5_comparison.json`` (``comparisons``, B4 regularization
  selection, claim outcomes),
* ``results/p5_selection.json`` (selected learning rates).

Declared definitions (LOG.md): ``train_loss`` is the loss logged before the
last update; ``validation_*`` the final held-out metric; losses are compared
as log10(other) - log10(better), accuracies as 100 * (better - other);
two-sided Wilcoxon; tolerance 0.1 decade or 1 point. Prints one line per
summary and exits 1 on any difference above 1e-12 (or any changed verdict,
selection or claim outcome).
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_DIR / "results"
TOLERANCE = 1e-12
NUMERIC = (
    "n_pairs",
    "median_advantage",
    "mean_advantage",
    "min_advantage",
    "max_advantage",
    "seeds_favouring",
    "wilcoxon_p",
)


def _runs(sweep_dir: Path) -> list[tuple[dict, dict]]:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if any(run["status"] != "COMPLETED" for run in status["runs"]):
        raise ValueError(f"{sweep_dir}: not every planned run COMPLETED")
    return [
        (run, json.loads((sweep_dir / run["metrics_path"]).read_text()))
        for run in status["runs"]
    ]


def _raw(record: dict, metric: str) -> float:
    if metric == "train_loss":
        return record["last_logged_train_loss"]
    return record["final"]["validation"][metric.removeprefix("validation_")]


def _table(sweep_dir: Path, rename=lambda config: config) -> dict:
    """Return ``values[(config, model, metric)][seed]`` for non-diverged models."""
    values: dict = defaultdict(dict)
    for run, metrics in _runs(sweep_dir):
        for model, record in metrics["models"].items():
            if record.get("diverged_at_epoch") is not None:
                continue
            for metric in (
                "train_loss",
                *(f"validation_{k}" for k in record["final"]["validation"]),
            ):
                values[(rename(run["config"]), model, metric)][run["seed"]] = _raw(
                    record, metric
                )
    return values


def _paired(values: dict, row: dict) -> dict:
    metric = row["metric"]
    better = values[(row["better_config"], row["better_model"], metric)]
    other = values[(row["other_config"], row["other_model"], metric)]
    seeds = sorted(better)
    if sorted(other) != seeds:
        raise ValueError(f"Seed mismatch for {row}")
    if "accuracy" in metric:
        # Accuracy differences sit on an exact grid; round away float noise.
        differences = np.round([100 * (better[s] - other[s]) for s in seeds], 9)
        tolerance = 1.0
    else:
        differences = np.array(
            [math.log10(other[s]) - math.log10(better[s]) for s in seeds]
        )
        tolerance = 0.1
    p_value = 1.0 if np.all(differences == 0) else float(wilcoxon(differences).pvalue)
    median = float(np.median(differences))
    if abs(median) < tolerance:
        verdict = "tie"
    elif median > 0 and p_value < 0.05:
        verdict = "supported"
    else:
        verdict = "not_supported"
    return {
        "n_pairs": len(seeds),
        "median_advantage": median,
        "mean_advantage": float(np.mean(differences)),
        "min_advantage": float(np.min(differences)),
        "max_advantage": float(np.max(differences)),
        "seeds_favouring": int(np.sum(differences > 0)),
        "wilcoxon_p": p_value,
        "verdict": verdict,
    }


def _compare(rows: list[dict], values: dict, label: str) -> int:
    worst, verdicts, checked = 0.0, 0, 0
    for row in rows:
        recomputed = _paired(values, row)
        for key in NUMERIC:
            if key in row:
                worst = max(worst, abs(float(row[key]) - float(recomputed[key])))
        if "verdict" in row and row["verdict"] != recomputed["verdict"]:
            verdicts += 1
        checked += 1
    print(
        f"{label}: {checked} comparisons recomputed; largest absolute difference {worst:.3g}; verdict changes {verdicts}"
    )
    return int(worst > TOLERANCE or verdicts > 0)


def _b4(reference_dir: Path, stored: dict) -> tuple[dict, int]:
    by_task: dict = defaultdict(dict)
    for run, metrics in _runs(reference_dir):
        by_task[run["config"].removeprefix("p5_ref_")][run["seed"]] = metrics
    values: dict = defaultdict(dict)
    mismatches = 0
    stored_selection = {
        (row["task"], row["model"]): row["selected_strength"] for row in stored
    }
    for task, seeds in by_task.items():
        classification = task.startswith("fig8")
        for variant, records in seeds[0]["variants"].items():
            strengths = [record["strength"] for record in records]
            medians = []
            for index in range(len(strengths)):
                metric = [
                    seeds[s]["variants"][variant][index][
                        "selection_logistic_loss" if classification else "selection_mse"
                    ]
                    for s in (0, 1, 2)
                ]
                medians.append(
                    float(np.median(metric if classification else np.log10(metric)))
                )
            best = min(range(len(strengths)), key=lambda i: (medians[i], strengths[i]))
            mismatches += stored_selection[(task, variant)] != strengths[best]
            for seed, metrics in seeds.items():
                record = metrics["variants"][variant][best]
                if classification:
                    values[(task, variant, "validation_accuracy_linear")][seed] = (
                        record["validation_accuracy"]
                    )
                else:
                    values[(task, variant, "validation_mse")][seed] = record[
                        "validation_mse"
                    ]
                    values[(task, variant, "train_loss")][seed] = record["train_mse"]
    return values, mismatches


def _selection(sweep_dir: Path, pattern_prefix: str, stored_rows: list[dict]) -> int:
    per_rate: dict = defaultdict(lambda: defaultdict(list))
    for run, metrics in _runs(sweep_dir):
        task, rate = (
            run["config"]
            .removeprefix(pattern_prefix)
            .rsplit("_eta" if "_eta" in run["config"] else "_lr", 1)
        )
        for model, record in metrics["models"].items():
            selection = record["final"]["selection"]
            if record["diverged_at_epoch"] is not None:
                value = math.nan
            elif metrics["task_kind"] == "regression":
                value = (
                    math.log10(selection["mse"]) if selection["mse"] > 0 else math.nan
                )
            else:
                value = selection["loss"]
            per_rate[(task, model)][float(rate)].append(value)
    mismatches = 0
    for row in stored_rows:
        candidates = per_rate[(row["task"], row["model"])]
        eligible = {
            rate: float(np.median(v))
            for rate, v in candidates.items()
            if np.all(np.isfinite(v))
        }
        chosen = (
            min(eligible, key=lambda rate: (eligible[rate], rate)) if eligible else None
        )
        mismatches += chosen != row["selected_learning_rate"]
    return mismatches


def main() -> int:
    """Run every recheck and return a non-zero exit code on any difference."""
    sweeps = Path(sys.argv[1])
    failures = 0
    e8 = json.loads((RESULTS / "e8_classical_data_summary.json").read_text())
    failures += _compare(
        e8["comparisons"], _table(sweeps / "e8_classical_data_v1"), "e8 comparisons"
    )
    e9 = json.loads((RESULTS / "e9_robustness_summary.json").read_text())
    e9_values = _table(sweeps / "e9_robustness_v1")
    failures += _compare(e9["grid"], e9_values, "e9 grid")
    failures += _compare(
        [row["advantage"] for row in e9["finite_shots"]], e9_values, "e9 finite shots"
    )
    p4 = json.loads((RESULTS / "p4_learning_summary.json").read_text())
    failures += _compare(
        p4["comparisons"], _table(sweeps / "p4_learning_v1"), "p4 learning comparisons"
    )
    p5 = json.loads((RESULTS / "p5_comparison.json").read_text())
    values = {
        **_table(
            sweeps / "p5_final_v1", lambda config: config.removeprefix("p5_final_")
        ),
        **_table(
            sweeps / "p5_photonic_final_v1",
            lambda config: config.removeprefix("p5_phfinal_"),
        ),
    }
    b4_values, b4_mismatches = _b4(sweeps / "p5_reference_v1", p5["b4_selection"])
    values.update(b4_values)
    print(
        f"p5 B4 regularization selections: {len(p5['b4_selection'])} rechecked; mismatches {b4_mismatches}"
    )
    failures += b4_mismatches > 0
    rows = [row for row in p5["comparisons"] if "median_advantage" in row]
    failures += _compare(rows, values, "p5 comparisons")
    outcome_changes = 0
    for entry in p5["claim_outcomes"].values():
        won = 0
        for task, detail in entry["per_task"].items():
            verdicts = []
            for control in detail["verdicts"]:
                (row,) = [
                    r
                    for r in rows
                    if r["task"] == task
                    and r["other_model"] == control
                    and r["metric"] in ("validation_mse", "validation_accuracy_linear")
                ]
                verdicts.append(_paired(values, row)["verdict"])
            won += all(verdict == "supported" for verdict in verdicts)
        outcome = (
            "SUPPORTED"
            if won == entry["tasks"]
            else ("PARTIALLY_SUPPORTED" if won else "UNSUPPORTED")
        )
        outcome_changes += outcome != entry["outcome"]
    print(
        f"p5 claim outcomes: {len(p5['claim_outcomes'])} rechecked; changes {outcome_changes}"
    )
    failures += outcome_changes > 0
    selection = json.loads((RESULTS / "p5_selection.json").read_text())
    qubit = _selection(sweeps / "p5_select_v1", "p5_sel_", selection["qubit"])
    photonic = _selection(
        sweeps / "p5_photonic_select_v1", "p5_phsel_", selection["photonic"]
    )
    print(
        f"p5 learning-rate selections: {len(selection['qubit']) + len(selection['photonic'])} rechecked; mismatches {qubit + photonic}"
    )
    failures += (qubit + photonic) > 0
    print(
        "RECHECK PASSED"
        if failures == 0
        else f"RECHECK FAILED ({failures} summaries differ)"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
