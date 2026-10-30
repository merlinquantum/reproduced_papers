"""Statistics behind the reproduction's claims, recomputed from the sweeps' artifacts.

Usage (from the repository root)::

    python papers/QuaRK/utils/report_statistics.py \\
        --main papers/QuaRK/outdir/sweep_main_comparison \\
        --shot-noise papers/QuaRK/outdir/sweep_shot_noise_ablation \\
        --scaling papers/QuaRK/outdir/sweep_scaling \\
        --noise-floor papers/QuaRK/results/forecast_noise_floor.json

Writes ``results/statistics.json`` (full precision) and ``results/statistics.md``.

Rules
-----
* Metric: test MSE at the ridge selected on validation data (frozen in the fair-baseline
  plan). The normalized test MSE divides it by the variance of the test labels.
* Aggregation: mean and sample standard deviation over the seeds, with a 95 % Student-t
  interval of the mean (descriptive, not part of the frozen rule).
* Paired comparison of A with B, per task, over the shared seeds:
  ``delta = MSE(A) - MSE(B)``, ``SE = std(delta) / sqrt(n)``. The frozen label is
  "A lower" if ``mean < -2 SE``, "B lower" if ``mean > +2 SE``, else "no clear
  separation". Reported next to it: the number of seeds where A has the lower error and
  the 95 % Student-t interval of the mean delta. A row is flagged when that interval
  contains 0 although the 2 SE rule separates the models.
* Comparisons outside the frozen plan (neighbouring shot budgets, n and R) use the same
  rule and are marked "after the sweep".
* QuaRK runs are identified by their recorded settings (estimator, shots, n, R), not by
  candidate names.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

TASKS = ("current_point_readout", "exp_fading", "volterra", "one_step_forecast")
PAPER_TASKS = TASKS[:3]
TASK_LABELS = {
    "current_point_readout": "Current-point readout (F1 as run)",
    "exp_fading": "Exponential fading (F2)",
    "volterra": "Volterra (F3)",
    "one_step_forecast": "One-step forecast (F1 as written)",
}
MAIN_MODELS = {
    "quark": "QuaRK (paper settings)",
    "quark_exact": "QuaRK, exact expectations",
    "quark_shadows": "QuaRK, random-Pauli shadows",
    "raw_window_kernel": "Matérn kernel on the raw window",
    "raw_window_linear": "Linear ridge on the raw window",
    "esn_default": "Echo state network, default",
    "esn_tuned": "Echo state network, tuned",
}
ESTIMATOR_LABELS = {
    "per_observable_shots": "Per-observable noise (authors' code)",
    "random_pauli_shadows": "Random-Pauli shadows",
    "exact": "Exact expectations",
}
PAPER_SEED = 12345
EXACT_SOLUTION_MSE = 1e-20
MAIN_SCOPE = (
    "Paper-accurate settings, 5 seeds (12345 is the authors' data; 1-4 are regenerated "
    "with their procedure). Every model except QuaRK is our addition."
)
SHOT_SCOPE = (
    "Extension: paper settings with the estimator or shot budget changed, 5 seeds."
)
SCALING_SCOPE = "Extension: paper settings with n or R changed, 5 seeds."


# --------------------------------------------------------------------------- loading


def load_sweep(sweep_dir: Path) -> tuple[str, dict]:
    """Return ``(sweep_id, runs)`` with ``runs[(candidate, seed)] = {metrics, reg}``."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise SystemExit(f"{sweep_dir}: sweep status is {status['status']}")
    runs = {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            raise SystemExit(f"{sweep_dir}: run {run['run_id']} is {run['status']}")
        run_dir = (sweep_dir / run["metrics_path"]).parent
        metrics = json.loads((run_dir / "metrics.json").read_text())
        with np.load(run_dir / "regularization_sweep.npz", allow_pickle=False) as data:
            reg = {key: data[key] for key in data.files}
        runs[(run["candidate"]["name"], int(run["seed"]))] = {
            "metrics": metrics,
            "reg": reg,
        }
    return status["sweep_id"], runs


def quark_settings(metrics: dict) -> tuple:
    """``(estimator, shots, n, R)`` of a QuaRK run; shots is None for exact features."""
    quark = metrics["model"]["quark"]
    shots = None if quark["estimator"] == "exact" else int(quark["shots"])
    return (
        quark["estimator"],
        shots,
        int(quark["num_qubits"]),
        int(quark["num_reservoirs"]),
    )


def cell(runs: dict, candidate: str, task: str, key: str, selection="validation"):
    """``{seed: value}`` of one metric for one candidate and task."""
    return {
        seed: run["metrics"]["tasks"][task]["selections"][selection][key]
        for (name, seed), run in runs.items()
        if name == candidate
    }


# ------------------------------------------------------------------------ statistics


def describe(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    n = values.size
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else None
    out = {"n": n, "mean": mean, "std": std}
    if n > 1:
        half = stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)
        out["ci95"] = [mean - half, mean + half]
    return out


def paired(a: dict, b: dict, name_a: str, name_b: str) -> dict:
    seeds = sorted(set(a) & set(b))
    delta = np.array([a[s] - b[s] for s in seeds], dtype=float)
    n = delta.size
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / math.sqrt(n))
    if mean < -2 * se:
        label = f"{name_a} lower"
    elif mean > 2 * se:
        label = f"{name_b} lower"
    else:
        label = "no clear separation"
    half = stats.t.ppf(0.975, n - 1) * se
    ci = [mean - half, mean + half]
    return {
        "seeds": seeds,
        "mean_delta": mean,
        "se_delta": se,
        "label": label,
        "a_lower_seeds": int(sum(a[s] < b[s] for s in seeds)),
        "ci95_delta": ci,
        "flag_interval_contains_zero": label != "no clear separation"
        and ci[0] <= 0.0 <= ci[1],
    }


def rank_of(values: dict, seed: int) -> int:
    """Rank of ``values[seed]`` among all seeds, 1 = lowest."""
    ordered = sorted(values, key=values.get)
    return ordered.index(seed) + 1


# ------------------------------------------------------------------------- sections


def main_table(runs: dict) -> list:
    rows = []
    for candidate in MAIN_MODELS:
        for task in TASKS:
            mse = cell(runs, candidate, task, "mse_test")
            rows.append(
                {
                    "candidate": candidate,
                    "task": task,
                    "mse_test": describe(mse.values()),
                    "normalized_mse_test": describe(
                        cell(runs, candidate, task, "normalized_mse_test").values()
                    ),
                    "mse_test_test_selected": describe(
                        cell(runs, candidate, task, "mse_test", "test").values()
                    ),
                    "paper_seed_mse_test": mse[PAPER_SEED],
                    "paper_seed_rank": rank_of(mse, PAPER_SEED),
                }
            )
    return rows


def main_comparisons(runs: dict) -> list:
    rows = []
    quark = {t: cell(runs, "quark", t, "mse_test") for t in TASKS}
    for candidate in MAIN_MODELS:
        if candidate == "quark":
            continue
        for task in TASKS:
            other = cell(runs, candidate, task, "mse_test")
            rows.append(
                {
                    "a": "quark",
                    "b": candidate,
                    "task": task,
                    "frozen": True,
                    **paired(quark[task], other, "QuaRK", MAIN_MODELS[candidate]),
                }
            )
    return rows


def exact_comparisons(runs: dict) -> list:
    """Noise-free QuaRK against each classical baseline (after the sweep)."""
    rows = []
    exact = {t: cell(runs, "quark_exact", t, "mse_test") for t in TASKS}
    for candidate in MAIN_MODELS:
        if candidate.startswith("quark"):
            continue
        for task in TASKS:
            other = cell(runs, candidate, task, "mse_test")
            rows.append(
                {
                    "a": "quark_exact",
                    "b": candidate,
                    "task": task,
                    "frozen": False,
                    **paired(
                        exact[task],
                        other,
                        MAIN_MODELS["quark_exact"],
                        MAIN_MODELS[candidate],
                    ),
                }
            )
    return rows


def interpolation(runs: dict) -> list:
    rows = []
    for candidate in MAIN_MODELS:
        for task in TASKS:
            floors, transitions = [], []
            for (name, _seed), run in runs.items():
                if name != candidate:
                    continue
                grid = run["reg"]["reg_grid"]
                train = run["reg"][f"{task}__mse_train"]
                floors.append(float(train[0]))
                # Largest ridge at which the training MSE is still below 1 % of its
                # value at the largest ridge (where the readout predicts about 0).
                below = np.flatnonzero(train <= 0.01 * train[-1])
                transitions.append(float(grid[below.max()]) if below.size else None)
            rows.append(
                {
                    "candidate": candidate,
                    "task": task,
                    "max_mse_train_at_smallest_reg": max(floors),
                    "transition_reg": transitions,
                }
            )
    return rows


def decay(runs: dict) -> list:
    rows = []
    for candidate in MAIN_MODELS:
        for task in TASKS:
            slopes = cell(runs, candidate, task, "decay_slope")
            curves = cell(runs, candidate, task, "training_size_mse_test")
            steps = [np.diff(np.asarray(c)) for c in curves.values()]
            decreasing = sum(int((s < 0).sum()) for s in steps)
            total = sum(s.size for s in steps)
            exact = all(
                v < EXACT_SOLUTION_MSE
                for v in cell(runs, candidate, task, "mse_test", "test").values()
            )
            summary = describe(slopes.values())
            rows.append(
                {
                    "candidate": candidate,
                    "task": task,
                    "slope": summary,
                    "ci_contains_minus_half": summary["ci95"][0]
                    <= -0.5
                    <= summary["ci95"][1],
                    "decreasing_steps": decreasing,
                    "total_steps": total,
                    "exact_solution": exact,
                }
            )
    return rows


def hardness(runs: dict) -> list:
    rows = []
    for candidate in MAIN_MODELS:
        raw = {t: cell(runs, candidate, t, "mse_test") for t in TASKS}
        norm = {t: cell(runs, candidate, t, "normalized_mse_test") for t in TASKS}
        seeds = sorted(raw[PAPER_TASKS[0]])
        # The paper's order with F1 as run, and with F1 as written (the forecast).
        as_run = PAPER_TASKS
        as_written = ("one_step_forecast", "exp_fading", "volterra")

        def ordered(values, s, tasks):
            a, b, c = (values[t][s] for t in tasks)
            return a < b < c

        rows.append(
            {
                "candidate": candidate,
                "seeds": len(seeds),
                "paper_order_raw": sum(ordered(raw, s, as_run) for s in seeds),
                "paper_order_normalized": sum(ordered(norm, s, as_run) for s in seeds),
                "paper_seed_raw": ordered(raw, PAPER_SEED, as_run),
                "paper_seed_normalized": ordered(norm, PAPER_SEED, as_run),
                "forecast_order_raw": sum(ordered(raw, s, as_written) for s in seeds),
                "forecast_order_normalized": sum(
                    ordered(norm, s, as_written) for s in seeds
                ),
            }
        )
    return rows


def bound(main_runs: dict, all_quark_runs: list) -> dict:
    paper = []
    for task in TASKS:
        terms = cell(main_runs, "quark", task, "bound")
        nus = {
            seed: run["metrics"]["tasks"][task]["kernel"]["nu"]
            for (name, seed), run in main_runs.items()
            if name == "quark"
        }
        seeds = sorted(terms, key=lambda seed: (seed != PAPER_SEED, seed))
        partial = {
            s: terms[s]["rademacher"] + terms[s]["mixing"] for s in seeds
        }  # truncation term left out: valid lower bound of the total for any nu
        gaps = {s: terms[s]["measured_gap"] for s in seeds}
        variance = {
            seed: run["metrics"]["tasks"][task]["label_variance_test"]
            for (name, seed), run in main_runs.items()
            if name == "quark"
        }
        totals = {s: terms[s]["total"] for s in seeds if terms[s]["total"] is not None}
        paper.append(
            {
                "task": task,
                "nu": [nus[s] for s in seeds],
                "seeds_with_nu_valid": sum(nus[s] > 1.0 for s in seeds),
                "rkhs_norm": describe(terms[s]["rkhs_norm"] for s in seeds),
                "rademacher": describe(terms[s]["rademacher"] for s in seeds),
                "mixing": describe(terms[s]["mixing"] for s in seeds),
                "truncation_max": max(
                    (terms[s]["truncation"] for s in seeds if terms[s]["truncation"]),
                    default=None,
                ),
                "measured_gap": describe(gaps.values()),
                "min_ratio_partial_to_gap": min(partial[s] / gaps[s] for s in seeds),
                "min_ratio_partial_to_label_variance": min(
                    partial[s] / variance[s] for s in seeds
                ),
                "total_defined_seeds": len(totals),
            }
        )
    counts, increases = {}, {}
    for run in all_quark_runs:
        for task in TASKS:
            kernel = run["metrics"]["tasks"][task]["kernel"]
            entry = counts.setdefault(task, {"fits": 0, "nu_at_most_one": 0})
            entry["fits"] += 1
            if kernel["nu"] <= 1.0:
                entry["nu_at_most_one"] += 1
                # Validation MSE of the best candidate that Theorem 2 covers (nu > 1).
                best_valid = min(
                    c["val_mse"] for c in kernel["candidates"] if c["nu"] > 1.0
                )
                increases.setdefault(task, []).append(
                    best_valid / kernel["val_mse"] - 1.0
                )
    for task, values in increases.items():
        counts[task]["val_mse_increase_if_nu_above_one"] = describe(values)
        counts[task]["max_val_mse_increase_if_nu_above_one"] = max(values)
    return {"paper_settings": paper, "nu_counts_all_quark_runs": counts}


def quark_index(*sweeps: dict) -> dict:
    """``index[(estimator, shots, n, R)][seed] = run`` over QuaRK runs of all sweeps."""
    index = {}
    for runs in sweeps:
        for (_name, seed), run in runs.items():
            if run["metrics"]["model"]["kind"] != "quark":
                continue
            key = quark_settings(run["metrics"])
            if seed in index.setdefault(key, {}):
                raise SystemExit(f"duplicate QuaRK run {key} seed {seed}")
            index[key][seed] = run
    return index


def values_of(group: dict, task: str, key="mse_test") -> dict:
    return {
        seed: run["metrics"]["tasks"][task]["selections"]["validation"][key]
        for seed, run in group.items()
    }


def monotone_count(index: dict, keys: list, task: str) -> int:
    """Seeds where the test MSE strictly decreases along ``keys``."""
    groups = [values_of(index[k], task) for k in keys]
    seeds = sorted(set.intersection(*(set(g) for g in groups)))
    return sum(
        all(groups[i][s] > groups[i + 1][s] for i in range(len(groups) - 1))
        for s in seeds
    )


def shot_budget(index: dict) -> dict:
    budgets = (100, 1000, 10000)
    estimators = ("per_observable_shots", "random_pauli_shadows")
    exact_key = ("exact", None, 5, 3)
    table, monotone, comparisons, penalty = [], [], [], []
    for task in TASKS:
        exact_values = values_of(index[exact_key], task)
        for estimator in estimators:
            keys = [(estimator, b, 5, 3) for b in budgets]
            for key in keys:
                values = values_of(index[key], task)
                table.append(
                    {
                        "setting": key,
                        "task": task,
                        "mse_test": describe(values.values()),
                    }
                )
                # Measurement penalty: test MSE over the same seed's exact-feature MSE.
                penalty.append(
                    {
                        "setting": key,
                        "task": task,
                        "ratio_to_exact": describe(
                            values[s] / exact_values[s] for s in sorted(exact_values)
                        ),
                    }
                )
            monotone.append(
                {
                    "estimator": estimator,
                    "task": task,
                    "seeds": monotone_count(index, keys, task),
                }
            )
            for low, high in zip(keys[:-1], keys[1:]):
                comparisons.append(_compare(index, low, high, task))
        table.append(
            {
                "setting": exact_key,
                "task": task,
                "mse_test": describe(values_of(index[exact_key], task).values()),
            }
        )
        for b in budgets:
            comparisons.append(
                _compare(
                    index, (estimators[0], b, 5, 3), (estimators[1], b, 5, 3), task
                )
            )
        for estimator in estimators:
            comparisons.append(
                _compare(index, exact_key, (estimator, 10000, 5, 3), task)
            )
    return {
        "table": table,
        "monotone": monotone,
        "comparisons": comparisons,
        "penalty": penalty,
    }


def scaling(index: dict) -> dict:
    axes = {
        "num_qubits": [("per_observable_shots", 1000, n, 3) for n in (3, 5, 7)],
        "num_reservoirs": [("per_observable_shots", 1000, 5, r) for r in (1, 3, 6)],
    }
    table, monotone, comparisons = [], [], []
    for task in TASKS:
        for axis, keys in axes.items():
            for key in keys:
                table.append(
                    {
                        "axis": axis,
                        "setting": key,
                        "task": task,
                        "mse_test": describe(values_of(index[key], task).values()),
                    }
                )
            monotone.append(
                {"axis": axis, "task": task, "seeds": monotone_count(index, keys, task)}
            )
            for low, high in zip(keys[:-1], keys[1:]):
                comparisons.append(_compare(index, low, high, task))
    return {"table": table, "monotone": monotone, "comparisons": comparisons}


def _compare(index: dict, key_a: tuple, key_b: tuple, task: str) -> dict:
    return {
        "a": key_a,
        "b": key_b,
        "task": task,
        "frozen": False,
        **paired(
            values_of(index[key_a], task),
            values_of(index[key_b], task),
            short_setting(key_a),
            short_setting(key_b),
        ),
    }


def reg_selection(all_runs: list) -> dict:
    """How much choosing the ridge on the test set (authors' rule) changes the test MSE."""
    out = {}
    for group in ("QuaRK", "all models"):
        same, differ, rel = 0, 0, []
        for run in all_runs:
            if group == "QuaRK" and run["metrics"]["model"]["kind"] != "quark":
                continue
            for task in TASKS:
                sel = run["metrics"]["tasks"][task]["selections"]
                if sel["validation"]["reg"] == sel["test"]["reg"]:
                    same += 1
                    continue
                differ += 1
                if sel["test"]["mse_test"] >= EXACT_SOLUTION_MSE:
                    rel.append(
                        sel["validation"]["mse_test"] / sel["test"]["mse_test"] - 1.0
                    )
        out[group] = {
            "fits": same + differ,
            "same_reg": same,
            "relative_increase_when_different": describe(rel) if rel else None,
            "max_relative_increase": max(rel) if rel else None,
        }
    return out


def forecast(runs: dict, floor: dict) -> dict:
    oracle = {r["seed"]: r["oracle_test_mse"] for r in floor["runs"]}
    expected = {r["seed"]: r["expected_floor_mse"] for r in floor["runs"]}
    variance = {r["seed"]: r["label_variance_test"] for r in floor["runs"]}
    rows = []
    for candidate in MAIN_MODELS:
        mse = cell(runs, candidate, "one_step_forecast", "mse_test")
        if set(mse) != set(oracle):
            raise SystemExit(
                f"noise floor seeds {sorted(oracle)} != run seeds {sorted(mse)}"
            )
        rows.append(
            {
                "candidate": candidate,
                "excess_over_oracle": describe(mse[s] - oracle[s] for s in oracle),
                "excess_over_oracle_normalized": describe(
                    (mse[s] - oracle[s]) / variance[s] for s in oracle
                ),
            }
        )
    return {
        "oracle_test_mse": describe(oracle.values()),
        "expected_floor_mse": describe(expected.values()),
        "oracle_test_mse_normalized": describe(oracle[s] / variance[s] for s in oracle),
        "models": rows,
    }


# --------------------------------------------------------------------------- report


def fmt(x, digits=3) -> str:
    if x is None:
        return "—"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.{digits - 1}e}"
    return f"{x:.{digits}g}"


def fmt_ms(d: dict, digits=3) -> str:
    if d["std"] is None:
        return fmt(d["mean"], digits)
    return f"{fmt(d['mean'], digits)} ± {fmt(d['std'], 2)}"


def comparison_line(c: dict, label_a: str, label_b: str) -> str:
    label = c["label"]
    flag = " (interval contains 0)" if c["flag_interval_contains_zero"] else ""
    ci = c["ci95_delta"]
    return (
        f"| {label_a} | {label_b} | {TASK_LABELS[c['task']]} | {fmt(c['mean_delta'])} "
        f"| [{fmt(ci[0])}, {fmt(ci[1])}] | {c['a_lower_seeds']}/{len(c['seeds'])} "
        f"| {label}{flag} |"
    )


def short_setting(key) -> str:
    estimator, shots, n, r = key
    name = {
        "per_observable_shots": "per-observable",
        "random_pauli_shadows": "shadows",
        "exact": "exact",
    }[estimator]
    budget = "" if shots is None else f" {shots}"
    return f"{name}{budget}, n = {n}, R = {r}"


def to_markdown(report: dict) -> str:
    out = [
        "# QuaRK reproduction: statistics",
        "",
        "Generated by `utils/report_statistics.py` from the `metrics.json` and "
        "`regularization_sweep.npz` files of the three sweeps; rules in its docstring. "
        "Test MSE is taken at the ridge selected on validation data; ± is the sample "
        "standard deviation over seeds; intervals are 95 % Student-t intervals.",
        "",
        "## Test MSE by model and task",
        "",
        MAIN_SCOPE,
        "",
        "| Model | Task | Test MSE | Normalized test MSE | Test MSE, ridge chosen on test | Authors' seed: value, rank of 5 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in report["main_table"]:
        out.append(
            f"| {MAIN_MODELS[r['candidate']]} | {TASK_LABELS[r['task']]} | {fmt_ms(r['mse_test'])} "
            f"| {fmt_ms(r['normalized_mse_test'])} | {fmt_ms(r['mse_test_test_selected'])} "
            f"| {fmt(r['paper_seed_mse_test'])}, {r['paper_seed_rank']} |"
        )
    out += [
        "",
        "## Paired comparisons with QuaRK (frozen rule)",
        "",
        MAIN_SCOPE
        + " Difference = test MSE of QuaRK minus test MSE of the other model.",
        "",
        "| Model A | Model B | Task | Mean difference | 95 % interval | Seeds where A is lower | Label (2 SE rule) |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for c in report["main_comparisons"]:
        out.append(comparison_line(c, "QuaRK (paper settings)", MAIN_MODELS[c["b"]]))
    out += [
        "",
        "## Paired comparisons of noise-free QuaRK with the classical baselines (after the sweep)",
        "",
        MAIN_SCOPE
        + " Difference = test MSE of QuaRK with exact expectations minus test MSE of the "
        "other model.",
        "",
        "| Model A | Model B | Task | Mean difference | 95 % interval | Seeds where A is lower | Label (2 SE rule) |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for c in report["exact_comparisons"]:
        out.append(comparison_line(c, MAIN_MODELS[c["a"]], MAIN_MODELS[c["b"]]))
    out += [
        "",
        "## Interpolation: training MSE at the smallest ridge",
        "",
        MAIN_SCOPE
        + " Transition = largest ridge at which the training MSE is below 1 % "
        "of its value at the largest ridge (range over seeds).",
        "",
        "| Model | Task | Largest training MSE at ridge 1e-12 | Transition ridge |",
        "|---|---|---:|---|",
    ]
    for r in report["interpolation"]:
        t = [x for x in r["transition_reg"] if x is not None]
        span = (
            "—"
            if not t
            else (
                fmt(min(t)) if min(t) == max(t) else f"{fmt(min(t))} to {fmt(max(t))}"
            )
        )
        missing = len(r["transition_reg"]) - len(t)
        if missing:
            span += f" ({missing} seeds never below 1 %)"
        out.append(
            f"| {MAIN_MODELS[r['candidate']]} | {TASK_LABELS[r['task']]} "
            f"| {fmt(r['max_mse_train_at_smallest_reg'])} | {span} |"
        )
    out += [
        "",
        "## Test MSE against the number of training windows",
        "",
        MAIN_SCOPE + " Slope of log test MSE against log N over N = 500 to 8000 "
        "(frozen metric); the paper describes a 1/√N shape, i.e. a slope of -0.5. "
        "Decreasing steps: consecutive N values where the test MSE fell, over all seeds.",
        "",
        "| Model | Task | Slope | 95 % interval | Contains -0.5 | Decreasing steps |",
        "|---|---|---:|---:|---|---:|",
    ]
    for r in report["decay"]:
        ci = r["slope"]["ci95"]
        note = (
            " (exact solution: slope not a learning rate)"
            if r["exact_solution"]
            else ""
        )
        out.append(
            f"| {MAIN_MODELS[r['candidate']]} | {TASK_LABELS[r['task']]} | {fmt_ms(r['slope'])}{note} "
            f"| [{fmt(ci[0])}, {fmt(ci[1])}] | {'yes' if r['ci_contains_minus_half'] else 'no'} "
            f"| {r['decreasing_steps']}/{r['total_steps']} |"
        )
    out += [
        "",
        "## Task ranking (paper: current-point readout < exponential fading < Volterra)",
        "",
        MAIN_SCOPE,
        "",
        "| Model | Seeds with the paper's order, raw MSE | Same, normalized MSE | Authors' seed, raw / normalized | With F1 as written (forecast < F2 < F3), raw / normalized |",
        "|---|---:|---:|---|---|",
    ]
    for r in report["hardness"]:
        out.append(
            f"| {MAIN_MODELS[r['candidate']]} | {r['paper_order_raw']}/{r['seeds']} "
            f"| {r['paper_order_normalized']}/{r['seeds']} "
            f"| {'yes' if r['paper_seed_raw'] else 'no'} / {'yes' if r['paper_seed_normalized'] else 'no'} "
            f"| {r['forecast_order_raw']}/{r['seeds']} / {r['forecast_order_normalized']}/{r['seeds']} |"
        )
    b = report["bound"]
    out += [
        "",
        "## Generalization bound (Theorem 2)",
        "",
        MAIN_SCOPE + " QuaRK at the paper settings, ridge chosen on validation data, "
        "N = 8000, δ = 0.05. Λ is the RKHS norm of the fitted readout; the dependence "
        "correction is dropped (δ' = δ), which can only lower the bound. The window term "
        "needs ν > 1. Ratio = (Rademacher + mixing terms) / measured gap, smallest over seeds; "
        "it is a lower bound of the full bound's ratio.",
        "",
        "| Task | Selected ν, seeds 12345, 1, 2, 3, 4 | Λ | Rademacher term | Mixing term | Largest window term | Measured gap (test − train MSE) | Smallest ratio to gap | Smallest ratio to label variance |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in b["paper_settings"]:
        out.append(
            f"| {TASK_LABELS[r['task']]} | {', '.join(fmt(x) for x in r['nu'])} | {fmt_ms(r['rkhs_norm'])} "
            f"| {fmt_ms(r['rademacher'])} | {fmt_ms(r['mixing'])} | {fmt(r['truncation_max'])} "
            f"| {fmt_ms(r['measured_gap'])} | {fmt(r['min_ratio_partial_to_gap'])} "
            f"| {fmt(r['min_ratio_partial_to_label_variance'])} |"
        )
    out += [
        "",
        "Kernel fits with ν ≤ 1 (Theorem 2 does not apply), over every QuaRK run of the three sweeps:",
        "",
    ]
    out += [
        "| Task | Fits with ν ≤ 1 | Validation MSE increase if ν > 1 is imposed: mean, largest |",
        "|---|---:|---|",
    ]
    for task, c in b["nu_counts_all_quark_runs"].items():
        inc = c.get("val_mse_increase_if_nu_above_one")
        text = (
            f"{fmt(inc['mean'])}, {fmt(c['max_val_mse_increase_if_nu_above_one'])}"
            if inc
            else "—"
        )
        out.append(
            f"| {TASK_LABELS[task]} | {c['nu_at_most_one']}/{c['fits']} | {text} |"
        )
    s = report["shot_budget"]
    out += [
        "",
        "## Shot budget and estimator",
        "",
        SHOT_SCOPE
        + " The 1000-shot and exact rows come from the main comparison (same seeds, data and code).",
        "",
        "| Setting | " + " | ".join(TASK_LABELS[t] for t in TASKS) + " |",
        "|---|" + "---:|" * len(TASKS),
    ]
    settings = []
    for row in s["table"]:
        if tuple(row["setting"]) not in settings:
            settings.append(tuple(row["setting"]))
    for key in settings:
        cells = {r["task"]: r for r in s["table"] if tuple(r["setting"]) == key}
        out.append(
            f"| {short_setting(key)} | "
            + " | ".join(fmt_ms(cells[t]["mse_test"]) for t in TASKS)
            + " |"
        )
    out += [
        "",
        "Measurement penalty: test MSE divided by the test MSE of the same seed with "
        "exact expectations (1 = no penalty); mean ± std over seeds:",
        "",
        "| Setting | " + " | ".join(TASK_LABELS[t] for t in TASKS) + " |",
        "|---|" + "---:|" * len(TASKS),
    ]
    penalty_settings = []
    for row in s["penalty"]:
        if tuple(row["setting"]) not in penalty_settings:
            penalty_settings.append(tuple(row["setting"]))
    for key in penalty_settings:
        cells = {r["task"]: r for r in s["penalty"] if tuple(r["setting"]) == key}
        out.append(
            f"| {short_setting(key)} | "
            + " | ".join(fmt_ms(cells[t]["ratio_to_exact"]) for t in TASKS)
            + " |"
        )
    out += [
        "",
        "Seeds where the test MSE falls at each step 100 → 1000 → 10000 shots:",
        "",
    ]
    out += [
        "| Estimator | " + " | ".join(TASK_LABELS[t] for t in TASKS) + " |",
        "|---|" + "---:|" * len(TASKS),
    ]
    for estimator in ("per_observable_shots", "random_pauli_shadows"):
        counts = {
            m["task"]: m["seeds"] for m in s["monotone"] if m["estimator"] == estimator
        }
        out.append(
            f"| {ESTIMATOR_LABELS[estimator]} | "
            + " | ".join(f"{counts[t]}/5" for t in TASKS)
            + " |"
        )
    out += [
        "",
        "Paired comparisons after the sweep (difference = A − B):",
        "",
        "| A | B | Task | Mean difference | 95 % interval | Seeds where A is lower | Label (2 SE rule) |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for c in s["comparisons"]:
        out.append(comparison_line(c, short_setting(c["a"]), short_setting(c["b"])))
    sc = report["scaling"]
    out += [
        "",
        "## Number of qubits n and of sub-reservoirs R",
        "",
        SCALING_SCOPE
        + " Per-observable noise, 1000 shots; the n = 5, R = 3 row comes from the main comparison.",
        "",
        "| Setting | " + " | ".join(TASK_LABELS[t] for t in TASKS) + " |",
        "|---|" + "---:|" * len(TASKS),
    ]
    seen = []
    for row in sc["table"]:
        key = tuple(row["setting"])
        if key in seen:
            continue
        seen.append(key)
        cells = {r["task"]: r for r in sc["table"] if tuple(r["setting"]) == key}
        out.append(
            f"| {short_setting(key)} | "
            + " | ".join(fmt_ms(cells[t]["mse_test"]) for t in TASKS)
            + " |"
        )
    out += [
        "",
        "Seeds where the test MSE falls at each step (n: 3 → 5 → 7; R: 1 → 3 → 6):",
        "",
    ]
    out += [
        "| Axis | " + " | ".join(TASK_LABELS[t] for t in TASKS) + " |",
        "|---|" + "---:|" * len(TASKS),
    ]
    for axis, name in (
        ("num_qubits", "Qubits n"),
        ("num_reservoirs", "Sub-reservoirs R"),
    ):
        counts = {m["task"]: m["seeds"] for m in sc["monotone"] if m["axis"] == axis}
        out.append(f"| {name} | " + " | ".join(f"{counts[t]}/5" for t in TASKS) + " |")
    out += [
        "",
        "Paired comparisons after the sweep (difference = A − B):",
        "",
        "| A | B | Task | Mean difference | 95 % interval | Seeds where A is lower | Label (2 SE rule) |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for c in sc["comparisons"]:
        out.append(comparison_line(c, short_setting(c["a"]), short_setting(c["b"])))
    rs = report["reg_selection"]
    out += [
        "",
        "## Ridge chosen on validation data versus on the test set",
        "",
        "Every run of the three sweeps. Relative increase = test MSE with the validation-selected "
        "ridge / test MSE with the test-selected ridge − 1, for fits where the two ridges differ "
        "(fits whose test-selected MSE is below 1e-20, exact solutions, are left out).",
        "",
        "| Runs | Fits | Same ridge | Mean relative increase when different | Largest |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, r in rs.items():
        inc = r["relative_increase_when_different"]
        out.append(
            f"| {group} | {r['fits']} | {r['same_reg']} | {fmt(inc['mean']) if inc else '—'} "
            f"| {fmt(r['max_relative_increase'])} |"
        )
    f = report["forecast"]
    out += [
        "",
        "## One-step forecast (F1 as written) and its noise floor",
        "",
        MAIN_SCOPE
        + " The oracle predicts the conditional mean of the next label from the "
        "true past of the process (`utils/forecast_noise_floor.py`); no model that sees only "
        f"the past can beat it in expectation. Oracle test MSE {fmt_ms(f['oracle_test_mse'])} "
        f"(normalized {fmt_ms(f['oracle_test_mse_normalized'])}); expected floor "
        f"{fmt_ms(f['expected_floor_mse'])}.",
        "",
        "| Model | Test MSE above the oracle | Same, normalized |",
        "|---|---:|---:|",
    ]
    for r in f["models"]:
        out.append(
            f"| {MAIN_MODELS[r['candidate']]} | {fmt_ms(r['excess_over_oracle'])} "
            f"| {fmt_ms(r['excess_over_oracle_normalized'])} |"
        )
    return "\n".join(out) + "\n"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--shot-noise", type=Path, required=True)
    parser.add_argument("--scaling", type=Path, required=True)
    parser.add_argument("--noise-floor", type=Path, required=True)
    args = parser.parse_args()

    main_id, main_runs = load_sweep(args.main)
    shot_id, shot_runs = load_sweep(args.shot_noise)
    scaling_id, scaling_runs = load_sweep(args.scaling)
    floor = json.loads(args.noise_floor.read_text())
    if floor["sweep_id"] != main_id:
        raise SystemExit(f"noise floor computed on {floor['sweep_id']}, not {main_id}")
    index = quark_index(main_runs, shot_runs, scaling_runs)
    all_runs = [*main_runs.values(), *shot_runs.values(), *scaling_runs.values()]

    report = {
        "sweeps": [main_id, shot_id, scaling_id],
        "runs": len(all_runs),
        "main_table": main_table(main_runs),
        "main_comparisons": main_comparisons(main_runs),
        "exact_comparisons": exact_comparisons(main_runs),
        "interpolation": interpolation(main_runs),
        "decay": decay(main_runs),
        "hardness": hardness(main_runs),
        "bound": bound(
            main_runs, [r for r in all_runs if r["metrics"]["model"]["kind"] == "quark"]
        ),
        "shot_budget": shot_budget(index),
        "scaling": scaling(index),
        "reg_selection": reg_selection(all_runs),
        "forecast": forecast(main_runs, floor),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "statistics.json").write_text(
        json.dumps(_jsonable(report), indent=2)
    )
    (RESULTS_DIR / "statistics.md").write_text(to_markdown(report))
    print(RESULTS_DIR / "statistics.json")
    print(RESULTS_DIR / "statistics.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
