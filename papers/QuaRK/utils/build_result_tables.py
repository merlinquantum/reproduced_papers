"""Result tables of the reproduction: paper items, photonic adaptation and sweeps.

Usage (from the repository root)::

    python papers/QuaRK/utils/build_result_tables.py \\
        --run papers/QuaRK/outdir/run_20261024-100724 \\
        --release-run <authors' release run> \\
        --main papers/QuaRK/outdir/sweep_main_comparison \\
        --shot-noise papers/QuaRK/outdir/sweep_shot_noise_ablation \\
        --scaling papers/QuaRK/outdir/sweep_scaling \\
        --photonic papers/QuaRK/outdir/sweep_photonic_adaptation

``<authors' release run>`` is ``data/QuaRK/authors_release_v1.0.0/storage/results/
reg_sweep/c-shadows/full/N=10000__w=25__d=3__s=100/2026-02-02_14-02-13``. Writes
``results/result_tables.json`` and ``results/result_tables.md``; the JSON also keeps
every seed's values, which the notebook plots. With ``--check`` it writes nothing: it
recomputes both files and compares them byte for byte with those in ``results/``
(exit status 1 if either differs).

Sources
-------
* Paper values: the authors' release, the run behind the paper's figures (seed 12345,
  ridge chosen on the test set); the paper prints no table.
* Reproduced values: our run at the authors' seed (``--run``, Fig. 3 data) and the
  curated ``results/*.json`` files of ``compare_with_release.py``,
  ``report_statistics.py``, ``compare_photonic_with_quark.py`` and
  ``compare_fair_baselines.py``.
* Complete candidate tables: every run of the four sweeps, through
  ``summarize_sweep.summarize``. The script stops when a curated file disagrees with
  these sweeps.

Rules
-----
* Those of ``report_statistics.py``: test MSE at the ridge chosen on validation data,
  mean and sample standard deviation over the seeds, paired 2 SE rule.
* Metric agreement with the paper: relative difference below ``AGREEMENT`` at the
  authors' seed. For the photonic adaptation, the tolerance frozen in its plan
  (within a factor 2 of its QuaRK counterpart).
* No sweep selects a candidate. Per task, the lowest mean normalized test MSE is
  compared with the next one under the paired 2 SE rule; a tie is reported when the
  two are not separated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from compare_photonic_with_quark import CLASSICAL, DETECTION, PHOTONIC, QUARK
from plot_style import load_release, load_run
from report_statistics import (
    MAIN_MODELS,
    PAPER_TASKS,
    PROJECT_DIR,
    RESULTS_DIR,
    TASKS,
    _jsonable,
    cell,
    fmt,
    fmt_ms,
    load_sweep,
    paired,
)
from summarize_sweep import summarize

AGREEMENT = 1e-6
INTERPOLATION_MAX = 1e-10
# Fig. 3 counts as "predictions on the labels" below this fraction of the labels' spread.
FIT_FRACTION = 0.01
SHORT = {
    "current_point_readout": "F1",
    "exp_fading": "F2",
    "volterra": "F3",
    "one_step_forecast": "Forecast",
}
SCALING = {
    "quark_n3": "QuaRK, n = 3 qubits (R = 3)",
    "quark_n7": "QuaRK, n = 7 qubits (R = 3)",
    "quark_r1": "QuaRK, R = 1 sub-reservoir (n = 5)",
    "quark_r6": "QuaRK, R = 6 sub-reservoirs (n = 5)",
}
LABELS = {**QUARK, **CLASSICAL, **PHOTONIC, **SCALING, **MAIN_MODELS}
CLASSICAL_KERNELS = ("raw_window_kernel", "esn_default", "esn_tuned")
PER_SEED = ("mse_test", "normalized_mse_test", "decay_slope", "training_size_mse_test")
SWEEPS = {
    "main": (
        "Main comparison",
        "QuaRK at the paper's settings, its two variants and the four classical "
        "baselines",
    ),
    "shot_noise": (
        "Shot budget",
        "QuaRK at 100 and 10000 shots, with the authors' noise model and with "
        "random-Pauli shadows (1000 shots: main comparison)",
    ),
    "scaling": (
        "Reservoir size",
        "QuaRK with n = 3 or 7 qubits and R = 1 or 6 sub-reservoirs (the paper's "
        "n = 5, R = 3: main comparison)",
    ),
    "photonic": (
        "Photonic adaptation",
        "The photonic reservoir at 100, 1000 and 10000 shots, exact, and exact with "
        "distinguishable photons",
    ),
}
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def power_of_ten(x: float) -> str:
    exponent = round(math.log10(x))
    if not math.isclose(x, 10.0**exponent):
        return fmt(x)
    return "1" if exponent == 0 else "10" + str(exponent).translate(SUPERSCRIPT)


def per_task(values: dict, digits=3) -> str:
    return " / ".join(fmt(values[task], digits) for task in PAPER_TASKS)


def check_close(ours: float, curated: float, what: str) -> None:
    if not math.isclose(ours, curated, rel_tol=1e-12, abs_tol=0.0):
        raise SystemExit(f"stale curated file: {what} ({ours!r} != {curated!r})")


def item(**fields) -> dict:
    keys = (
        "item",
        "claim",
        "paper",
        "ours",
        "delta",
        "metric",
        "trend",
        "support",
        "seeds",
        "scope",
        "comment",
    )
    return {key: fields[key] for key in keys}


# ------------------------------------------------------------------------ sections


def paper_items(release_cmp: dict, release: dict, arrays: dict, stats: dict) -> list:
    tasks = release_cmp["tasks"]
    interp = {(r["candidate"], r["task"]): r for r in stats["interpolation"]}
    decay = {(r["candidate"], r["task"]): r for r in stats["decay"]}
    main = {(r["candidate"], r["task"]): r for r in stats["main_table"]}
    hardness = {r["candidate"]: r for r in stats["hardness"]}
    bound = {r["task"]: r for r in stats["bound"]["paper_settings"]}
    nu_counts = stats["bound"]["nu_counts_all_quark_runs"]
    grid = arrays["regularization_sweep"]["reg_grid"]
    if not np.array_equal(grid, release["reg_grid"]) or grid[0] > grid[-1]:
        raise SystemExit("ridge grids differ from the release or are not ascending")
    single = "paper-accurate settings, single seed"
    five = "paper-accurate settings, 5 seeds"
    both = "paper-accurate settings: single seed, then 5 seeds"

    def agreement(difference: float) -> str:
        return "yes, to numerical precision" if difference < AGREEMENT else "no"

    xi_difference = max(tasks[t]["xi_relative_difference"] for t in PAPER_TASKS)
    same_nu = all(tasks[t]["nu"]["ours"] == tasks[t]["nu"]["release"] for t in tasks)
    rows = [
        item(
            item="Readout tuning (Sec. 4.2): Matérn ν and ξ, F1 / F2 / F3",
            claim="none: the setting behind Figs. 2–4",
            paper=" / ".join(
                f"({fmt(tasks[t]['nu']['release'])}, {tasks[t]['xi']['release']:.3f})"
                for t in PAPER_TASKS
            ),
            ours="same ν" if same_nu else "different ν",
            delta=f"ξ: {fmt(xi_difference, 2)} relative",
            metric=agreement(xi_difference) if same_nu else "no",
            trend="N/A",
            support="N/A",
            seeds="1",
            scope=single,
            comment="ν chosen in {0.5, 1.5, 2.5, 5} on validation data",
        )
    ]
    for i, task in enumerate(PAPER_TASKS):
        t = tasks[task]
        above = t["fig2_training_mse_max_relative_difference_above_rounding_level"]
        quark_max = interp[("quark", task)]["max_mse_train_at_smallest_reg"]
        release_min = float(release["mse_train"][i][0])
        classical_max = max(
            interp[(c, task)]["max_mse_train_at_smallest_reg"]
            for c in CLASSICAL_KERNELS
        )
        transition = interp[("quark", task)]["transition_reg"]
        both_zero = max(quark_max, release_min) <= INTERPOLATION_MAX
        rows.append(
            item(
                item=f"Fig. 2, {SHORT[task]}: training MSE against λ_reg",
                claim="relaxing λ_reg drives the training MSE to numerical zero",
                paper=(
                    f"{fmt(t['fig2_training_mse_at_largest_ridge']['release'], 4)} at "
                    f"λ_reg = {power_of_ten(grid[-1])}; {fmt(release_min, 2)} at "
                    f"{power_of_ten(grid[0])}"
                ),
                ours=(
                    f"same at seed 12345; 5 seeds: at most {fmt(quark_max, 2)} at "
                    f"{power_of_ten(grid[0])}"
                ),
                delta=(
                    f"{fmt(above, 2)} relative where above "
                    f"{power_of_ten(INTERPOLATION_MAX)}; "
                    f"{fmt(t['fig2_training_mse_max_abs_difference_below_rounding_level'], 2)} "
                    "absolute below"
                ),
                metric=agreement(above),
                trend=(
                    f"yes: both fall below {power_of_ten(INTERPOLATION_MAX)}"
                    if both_zero
                    else "no"
                ),
                support=(
                    "supported, not specific to QuaRK: the classical kernel models "
                    f"reach at most {fmt(classical_max, 2)}"
                    if both_zero
                    else "not supported"
                ),
                seeds="1 / 5",
                scope=both,
                comment=(
                    f"QuaRK's transition: λ_reg = {power_of_ten(min(transition))} to "
                    f"{power_of_ten(max(transition))} over the seeds (code's units, "
                    "K + λ_reg·I)"
                ),
            )
        )
    fit = arrays["fit_examples"]
    errors, fractions = {}, {}
    for task in PAPER_TASKS:
        truth, prediction = fit[f"{task}__truth"], fit[f"{task}__prediction"]
        errors[task] = float(np.max(np.abs(prediction - truth)))
        fractions[task] = errors[task] / float(np.std(truth))
    on_labels = max(fractions.values()) < FIT_FRACTION
    rows.append(
        item(
            item="Fig. 3: predicted and true labels, 50 training windows",
            claim="at a small λ_reg the predictions follow the training labels",
            paper="figure only, no number",
            ours=f"largest absolute error, F1 / F2 / F3: {per_task(errors, 2)}",
            delta="N/A: no paper number",
            metric="N/A",
            trend=("yes: below 1 % of the labels' spread" if on_labels else "no"),
            support="supported (as Fig. 2)" if on_labels else "not supported",
            seeds="1",
            scope=single,
            comment="fit at the tuning ridge 10⁻⁶ on the full training set, as in the "
            "authors' figure",
        )
    )
    max_change = stats["reg_selection"]["QuaRK"]["max_relative_increase"]
    for task in PAPER_TASKS:
        t, d, m = tasks[task], decay[("quark", task)], main[("quark", task)]
        first, last = t["fig4_test_mse_first_last"]["release"]
        lo, hi = d["slope"]["ci95"]
        decreasing = d["decreasing_steps"] == d["total_steps"]
        if decreasing and d["ci_contains_minus_half"]:
            support = "supported"
        elif decreasing:
            support = "decrease supported; 1/√N shape not supported"
        else:
            support = "not supported"
        rows.append(
            item(
                item=f"Fig. 4, {SHORT[task]}: test MSE against N",
                claim="the test MSE decreases with N, with the same 1/√N shape for "
                "the three tasks",
                paper=f"{fmt(first, 4)} at N = 100, {fmt(last, 4)} at N = 8000",
                ours=(
                    f"same at seed 12345; 5 seeds at N = 8000: {fmt_ms(m['mse_test'])}; "
                    f"slope {fmt(d['slope']['mean'])} (95 % interval {fmt(lo)} to "
                    f"{fmt(hi)})"
                ),
                delta=f"{fmt(t['fig4_test_mse_max_relative_difference'], 2)} relative",
                metric=(
                    f"{agreement(t['fig4_test_mse_max_relative_difference'])}; the "
                    f"authors' seed ranks {m['paper_seed_rank']} of 5 (1 = lowest)"
                ),
                trend=(
                    f"decrease at {d['decreasing_steps']}/{d['total_steps']} steps; "
                    f"slope −0.5 {'inside' if d['ci_contains_minus_half'] else 'outside'}"
                    " the interval"
                ),
                support=support,
                seeds="1 / 5",
                scope=both,
                comment=(
                    "slope of log test MSE against log N over N = 500–8000; choosing "
                    "λ_reg on the test set (authors) instead of validation data (ours) "
                    f"changes QuaRK's test MSE by at most {max_change:.1%}"
                ),
            )
        )
    order_all = [c for c, h in hardness.items() if h["paper_order_raw"] == h["seeds"]]
    quark = hardness["quark"]
    linear = max(
        main[("raw_window_linear", t)]["normalized_mse_test"]["mean"]
        for t in ("current_point_readout", "exp_fading")
    )
    rows.append(
        item(
            item="Sec. 4.3: ranking of the three tasks",
            claim="test MSE F1 < F2 < F3 mirrors how hard the labels are",
            paper=" < ".join(
                fmt(tasks[t]["fig4_test_mse_first_last"]["release"][-1], 4)
                for t in PAPER_TASKS
            )
            + " at N = 8000",
            ours=(
                f"QuaRK: {quark['paper_order_raw']}/{quark['seeds']} seeds on test MSE, "
                f"{quark['paper_order_normalized']}/{quark['seeds']} normalized; with F1 "
                f"as written (forecast): {quark['forecast_order_raw']}/{quark['seeds']} "
                f"and {quark['forecast_order_normalized']}/{quark['seeds']}"
            ),
            delta="N/A: an order",
            metric="N/A",
            trend="yes" if quark["paper_order_raw"] == quark["seeds"] else "no",
            support=(
                f"partially supported: the order holds in every seed for "
                f"{len(order_all)} of {len(hardness)} models, the linear ridge "
                f"{'included' if 'raw_window_linear' in order_all else 'excluded'}"
            ),
            seeds="5",
            scope=five,
            comment=(
                "F1 and F2 are linear in the window: the linear ridge reaches a "
                f"normalized test MSE of at most {fmt(linear, 2)} on them"
            ),
        )
    )
    ratios = " / ".join(
        f"{bound[t]['min_ratio_partial_to_gap']:,.0f}" for t in PAPER_TASKS
    )
    nu_fits = "; ".join(
        f"{SHORT[t]} {nu_counts[t]['nu_at_most_one']}/{nu_counts[t]['fits']}"
        for t in PAPER_TASKS
    )
    rows.append(
        item(
            item="Theorem 2: generalization bound",
            claim="the bound controls the test error of windows from a β-mixing process",
            paper="stated and proved, not evaluated numerically",
            ours=(
                "at N = 8000, the Rademacher and mixing terms alone are at least "
                f"{ratios} times the measured gap (F1 / F2 / F3)"
            ),
            delta="N/A: no paper number",
            metric="N/A",
            trend="N/A",
            support="far from the measured errors; the proof was not re-checked",
            seeds="5",
            scope=five,
            comment=(
                "ν ≤ 1, outside the theorem's ν > 1, is chosen in these QuaRK fits: "
                f"{nu_fits}"
            ),
        )
    )
    return rows


def photonic_items(photonic: dict, fair: dict) -> list:
    comparisons = {(c["a"], c["b"], c["task"]): c for c in photonic["comparisons"]}
    tolerance = {(t["a"], t["b"], t["task"]): t for t in photonic["tolerance"]}
    evidence = fair["claim_evidence"]
    behaviour = {
        (c, b["task"]): b for c, rows in photonic["behaviour"].items() for b in rows
    }
    scope = "adaptation, simulated, 5 seeds"

    def labels(a: str, b: str, other: str) -> str:
        out = []
        for task in PAPER_TASKS:
            p = comparisons[(a, b, task)]["paired"]
            if p["label"] == "no clear separation":
                text = "not separated"
            elif p["label"] == f"{a} lower":
                text = "photonic lower"
            else:
                text = f"{other} lower"
            if p["flag_interval_contains_zero"]:
                text += ", interval contains 0"
            out.append(f"{SHORT[task]} {text} ({p['a_lower_seeds']}/5 seeds)")
        return "2 SE rule: " + "; ".join(out)

    rows = []
    for a, b, name in (
        ("quark_exact_merlin", "quark_exact", "exact features"),
        ("quark_merlin", "quark_shadows", "1000 shots shared by all features"),
    ):
        pairs = [tolerance[(a, b, task)] for task in PAPER_TASKS]
        within = all(p["factor_2"] for p in pairs)
        rows.append(
            item(
                item=f"Normalized test MSE, {name}, F1 / F2 / F3 "
                f"(counterpart: {LABELS[b]})",
                claim="the adaptation keeps QuaRK's accuracy (frozen tolerance)",
                paper=per_task(
                    {
                        t: comparisons[(a, b, t)]["normalized_b"]["mean"]
                        for t in PAPER_TASKS
                    }
                ),
                ours=per_task(
                    {
                        t: comparisons[(a, b, t)]["normalized_a"]["mean"]
                        for t in PAPER_TASKS
                    }
                ),
                delta="ratio " + " / ".join(f"{p['ratio']:.2f}" for p in pairs),
                metric="yes: within a factor 2" if within else "no",
                trend="N/A",
                support="preserved"
                if all(p["pass"] for p in pairs)
                else "not preserved",
                seeds="5",
                scope=scope,
                comment=labels(a, b, "QuaRK"),
            )
        )
    quark_max = max(
        evidence[c]["max_train_mse_at_smallest_ridge"]
        for c in ("quark_exact", "quark_shadows")
    )
    photonic_rows = [behaviour[(c, t)] for c in PHOTONIC for t in PAPER_TASKS]
    photonic_max = max(b["max_train_mse_at_smallest_reg"] for b in photonic_rows)
    interpolating = min(b["interpolating_seeds"] for b in photonic_rows)
    rows.append(
        item(
            item="Fig. 2: training MSE at λ_reg = 10⁻¹², every photonic configuration "
            "(counterpart: QuaRK, exact and shadows)",
            claim="relaxing λ_reg drives the training MSE to numerical zero",
            paper=f"at most {fmt(quark_max, 2)}",
            ours=f"at most {fmt(photonic_max, 2)}",
            delta="N/A: both at numerical zero",
            metric="N/A",
            trend=(
                f"yes: below {power_of_ten(INTERPOLATION_MAX)} in {interpolating}/5 "
                "seeds, every configuration and task"
            ),
            support="supported" if interpolating == 5 else "not supported",
            seeds="5",
            scope=scope,
            comment="as for QuaRK, every Matérn-kernel model interpolates",
        )
    )
    decaying = min(b["decaying_seeds"] for b in photonic_rows)

    def slopes(candidate: str) -> str:
        return " / ".join(
            fmt(evidence[candidate]["decay_slope"][t]["mean"]) for t in PAPER_TASKS
        )

    half = [
        SHORT[t]
        for t in PAPER_TASKS
        for c in ("quark_exact_merlin", "quark_merlin")
        if evidence[c]["decay_slope"][t]["ci95"][0]
        <= -0.5
        <= evidence[c]["decay_slope"][t]["ci95"][1]
    ]
    rows.append(
        item(
            item="Fig. 4: test MSE against N, every photonic configuration "
            "(counterpart: QuaRK, exact and shadows)",
            claim="the test MSE decreases with N, with a 1/√N shape",
            paper=(
                f"slopes F1 / F2 / F3: exact {slopes('quark_exact')}; shadows "
                f"{slopes('quark_shadows')}"
            ),
            ours=(
                f"slopes: exact {slopes('quark_exact_merlin')}; 1000 shots "
                f"{slopes('quark_merlin')}"
            ),
            delta="N/A: compared through the claim",
            metric="N/A",
            trend=(
                f"decrease: lower at N = 8000 than at N = 100 in {decaying}/5 seeds, "
                "every configuration and task"
            ),
            support=(
                "decrease supported; 1/√N shape "
                + ("on " + ", ".join(sorted(set(half))) if half else "on no task")
            ),
            seeds="5",
            scope=scope,
            comment="slopes over N = 500–8000; −0.5 is 1/√N",
        )
    )
    a, b = "quark_exact_merlin", "quark_distinguishable_merlin"
    largest = max(
        abs(
            comparisons[(a, b, t)]["normalized_a"]["mean"]
            - comparisons[(a, b, t)]["normalized_b"]["mean"]
        )
        for t in PAPER_TASKS
    )
    rows.append(
        item(
            item="Interference control, F1 / F2 / F3 (counterpart: the same circuits "
            "with distinguishable photons)",
            claim="none in the paper: does photon interference matter?",
            paper=per_task(
                {t: comparisons[(a, b, t)]["normalized_b"]["mean"] for t in PAPER_TASKS}
            ),
            ours=per_task(
                {t: comparisons[(a, b, t)]["normalized_a"]["mean"] for t in PAPER_TASKS}
            ),
            delta=f"means differ by at most {fmt(largest, 3)}",
            metric="N/A",
            trend="N/A",
            support="N/A: our control",
            seeds="5",
            scope=scope,
            comment=labels(a, b, "distinguishable"),
        )
    )
    return rows


def hardware(photonic: dict) -> list:
    rows = []
    for s in photonic["setup"]:
        merlin = s["merlin"]
        rows.append(
            {
                "configuration": PHOTONIC[s["candidate"]],
                "photons": f"{merlin['num_photons']}"
                + (", distinguishable" if merlin["distinguishable"] else ""),
                "modes": merlin["num_modes"],
                "sub_reservoirs": merlin["num_reservoirs"],
                "input_state": s["input_state"],
                "click_patterns": s["num_click_patterns"],
                "marginals": s["num_observables"],
                "shots": merlin["shots"]
                if merlin["estimator"] == "click_sampling"
                else "exact",
                "simulator": (
                    f"MerLin simulator ({s['versions']['MerLin']}), Perceval "
                    f"{s['versions']['Perceval']}, {s['device'][0]}, {s['device'][1]}"
                ),
                "seeds": s["seeds"],
                "seconds_mean": s["seconds"]["mean"],
            }
        )
    return rows


def per_seed(runs: dict, candidate: str) -> dict:
    """Every seed's values of one candidate, at the ridge chosen on validation data."""
    out = {}
    for (name, seed), run in sorted(runs.items()):
        if name != candidate:
            continue
        metrics = run["metrics"]
        out[str(seed)] = {
            "seconds": metrics["seconds"],
            "tasks": {
                task: {
                    **{
                        key: result["selections"]["validation"][key] for key in PER_SEED
                    },
                    "mse_test_test_selected": result["selections"]["test"]["mse_test"],
                    "mse_train_at_smallest_reg": result["mse_train_at_smallest_reg"],
                }
                for task, result in metrics["tasks"].items()
            },
        }
    return out


def sweep_section(key: str, sweep_dir: Path) -> dict:
    summary = summarize(sweep_dir)
    _, runs = load_sweep(sweep_dir)
    spec = json.loads((sweep_dir / "sweep_spec.json").read_text())
    table = {}
    for row in summary["table"]:
        table.setdefault(row["candidate"], {})[row["task"]] = row
    versus = {(c["other"], c["task"]): c["label"] for c in summary["comparisons"]}
    plain = {
        "other better": "lower",
        "reference better": "higher",
        "no clear separation": "not separated",
    }
    candidates = []
    for name in [c["name"] for c in spec["candidates"]]:
        rows = table[name]
        candidates.append(
            {
                "candidate": name,
                "label": LABELS[name],
                "runs": min(r["successful_runs"] for r in rows.values()),
                "expected_runs": min(r["expected_runs"] for r in rows.values()),
                "normalized": {
                    t: {
                        "mean": rows[t]["normalized_mse_test_mean"],
                        "std": rows[t]["normalized_mse_test_std"],
                    }
                    for t in TASKS
                },
                "versus_reference": None
                if name == spec["reference"]
                else {t: plain[versus[(name, t)]] for t in TASKS},
                "per_seed": per_seed(runs, name),
            }
        )
    lowest = {}
    for task in TASKS:
        means = {c["candidate"]: c["normalized"][task]["mean"] for c in candidates}
        first, second = sorted(means, key=means.get)[:2]
        test = paired(
            cell(runs, first, task, "mse_test"),
            cell(runs, second, task, "mse_test"),
            first,
            second,
        )
        lowest[task] = {"lowest": first, "next": second, "paired": test}
    title, varies = SWEEPS[key]
    return {
        "sweep_id": summary["sweep_id"],
        "title": title,
        "varies": varies,
        "reference": spec["reference"],
        "coverage": summary["coverage"],
        "completed_runs": summary["completed_runs"],
        "expected_runs": summary["expected_runs"],
        "missing_runs": summary["missing_runs"],
        "invalid_cells": summary["invalid_cells"],
        "candidates": candidates,
        "lowest": lowest,
        "training_sizes": next(iter(runs.values()))["metrics"]["tasks"]["volterra"][
            "selections"
        ]["validation"]["training_sizes"],
    }


def check_curated(sweeps: dict, stats: dict, photonic: dict, fair: dict) -> None:
    """Stop when a curated file was not built from the sweeps read here."""
    normalized = {
        (c["candidate"], t): c["normalized"][t]["mean"]
        for s in sweeps.values()
        for c in s["candidates"]
        for t in TASKS
    }
    for row in stats["main_table"]:
        key = (row["candidate"], row["task"])
        check_close(
            normalized[key], row["normalized_mse_test"]["mean"], f"statistics {key}"
        )
    for c in photonic["comparisons"]:
        if c["a"] in PHOTONIC:
            key = (c["a"], c["task"])
            check_close(normalized[key], c["normalized_a"]["mean"], f"photonic {key}")
    for system in fair["systems"]:
        for task, value in system["normalized"].items():
            key = (system["candidate"], task)
            check_close(normalized[key], value["mean"], f"fair comparison {key}")
    runs = sum(sweeps[k]["completed_runs"] for k in ("main", "shot_noise", "scaling"))
    if runs != stats["runs"]:
        raise SystemExit(f"statistics.json counts {stats['runs']} runs, sweeps {runs}")


def merlin_pin() -> str:
    for line in (PROJECT_DIR / "requirements.txt").read_text().splitlines():
        if line.startswith("merlinquantum=="):
            return line.split("==")[1]
    raise SystemExit("requirements.txt does not pin merlinquantum")


# ------------------------------------------------------------------------- report


def table_lines(rows: list, headers: tuple, keys: tuple) -> list:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
    return lines


def to_markdown(report: dict) -> str:
    keys = (
        "item",
        "claim",
        "paper",
        "ours",
        "delta",
        "metric",
        "trend",
        "support",
        "seeds",
        "scope",
        "comment",
    )
    lines = [
        "# QuaRK reproduction: result tables",
        "",
        "Generated by `utils/build_result_tables.py`; no value is typed by hand. F1: "
        "current-point readout (F1 as run); F2: exponential fading; F3: Volterra; "
        "Forecast: one-step forecast (F1 as the paper's text defines it).",
        "",
        "## Paper items",
        "",
        "Paper values come from the authors' release, the run behind the paper's "
        "figures (seed 12345, λ_reg chosen on the test set); the paper prints no "
        "table. Reproduced values: our run at the same seed and rule, then 5 seeds "
        "(12345 and four draws made with the authors' procedure) with λ_reg chosen on "
        "validation data, mean ± std. Metric agreement means a relative difference "
        f"below {power_of_ten(AGREEMENT)}.",
        "",
        *table_lines(
            report["paper_items"],
            (
                "Paper item",
                "Claim tested",
                "Paper value",
                "Reproduced value",
                "Delta",
                "Metric agreement",
                "Trend agreement",
                "Claim support",
                "Seeds",
                "Scope",
                "Comment",
            ),
            keys,
        ),
        "",
        "## Photonic adaptation (MerLin)",
        "",
        "The paper has no photonic version: each photonic value is set against its "
        "QuaRK counterpart from our runs, on the same windows, split and seeds. "
        "Metric agreement uses the tolerance frozen before the runs: within a factor "
        "2 of the counterpart.",
        "",
        *table_lines(
            report["photonic_items"],
            (
                "Item",
                "Claim tested",
                "Counterpart value",
                "Photonic value",
                "Delta",
                "Metric agreement",
                "Trend agreement",
                "Claim support",
                "Seeds",
                "Scope",
                "Comment",
            ),
            keys,
        ),
        "",
        "### Hardware-aware settings",
        "",
        "Simulated, not hardware-executed; no photonic parameter is trained, so no "
        "gradient is used. Common to every configuration: computation space "
        f"{DETECTION['computation_space']}; detector: {DETECTION['detector']}; "
        f"postselection: {DETECTION['postselection']}; encoding: "
        f"{DETECTION['encoding']}; measurement: {DETECTION['measurement']}. The "
        "simulator version is the one MerLin prints; the installed distribution is "
        f"merlinquantum {report['merlin_pin']}.",
        "",
        "| Configuration | Photons | Modes | Sub-reservoirs | Input state | Click "
        "patterns | Marginals | Shots per window and sub-reservoir | Simulator or "
        "QPU | Seeds | Wall-clock per run (mean) |",
        "|---|---|---:|---:|---|---:|---|---|---|---|---:|",
    ]
    for h in report["hardware"]:
        lines.append(
            f"| {h['configuration']} | {h['photons']} | {h['modes']} "
            f"| {h['sub_reservoirs']} | `{h['input_state']}` | {h['click_patterns']} "
            f"| {h['marginals']} per sub-reservoir | {h['shots']} | {h['simulator']} "
            f"| {', '.join(str(s) for s in h['seeds'])} | {h['seconds_mean']:.0f} s |"
        )
    lines += [
        "",
        "## Sweeps",
        "",
        "No sweep selects a winner: each compares models or settings with its "
        "reference, under the paired 2 SE rule frozen before the runs. Inside every "
        "run, the Matérn ν and ξ, λ_reg and the tuned echo state network's setting "
        "(18 candidates) are chosen on validation data, the inner 20 % of the "
        "training windows; the solves are closed-form, so there are no checkpoints. "
        "Metric: test MSE on 2,000 held-out windows, lower is better; the tables show "
        "the normalized test MSE (test MSE / variance of the test labels), mean ± std "
        "over the seeds. Lowest: the candidate with the lowest mean per task; a tie "
        "means the next candidate is not separated from it.",
        "",
        "| Sweep | What varies | Reference | Runs | Coverage | Lowest mean, F1 / F2 / "
        "F3 / Forecast |",
        "|---|---|---|---:|---|---|",
    ]
    for s in report["sweeps"].values():
        lowest = []
        for task in TASKS:
            low = s["lowest"][task]
            text = LABELS[low["lowest"]]
            if low["paired"]["label"] == "no clear separation":
                text += f" (tie with {LABELS[low['next']]})"
            elif low["paired"]["flag_interval_contains_zero"]:
                text += (
                    f" (ahead of {LABELS[low['next']]} under the 2 SE rule, but the "
                    "95 % interval of the difference contains 0)"
                )
            lowest.append(text)
        lines.append(
            f"| {s['title']} (`{s['sweep_id']}`) | {s['varies']} "
            f"| {LABELS[s['reference']]} | {s['completed_runs']}/{s['expected_runs']} "
            f"| {s['coverage']} | {' / '.join(lowest)} |"
        )
    for s in report["sweeps"].values():
        lines += [
            "",
            f"### {s['title']}",
            "",
            "Versus reference: the 2 SE rule's verdict for F1 / F2 / F3 / Forecast "
            "(lower or higher test MSE than the reference, or not separated).",
            "",
            "| Candidate | Runs | F1 | F2 | F3 | Forecast | Versus reference |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for c in s["candidates"]:
            versus = (
                "reference"
                if c["versus_reference"] is None
                else " / ".join(c["versus_reference"][t] for t in TASKS)
            )
            values = " | ".join(fmt_ms(c["normalized"][t]) for t in TASKS)
            lines.append(
                f"| {c['label']} | {c['runs']}/{c['expected_runs']} | {values} "
                f"| {versus} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--release-run", type=Path, required=True)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--shot-noise", type=Path, required=True)
    parser.add_argument("--scaling", type=Path, required=True)
    parser.add_argument("--photonic", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    metrics, arrays = load_run(args.run)
    if metrics["dataset_meta"]["seed"] != 12345:
        raise SystemExit("--run must be the run at the authors' seed 12345")
    curated = {
        name: json.loads((RESULTS_DIR / f"{name}.json").read_text())
        for name in (
            "release_comparison",
            "statistics",
            "photonic_comparison",
            "fair_comparison",
        )
    }
    if curated["release_comparison"]["run"] != args.run.name:
        raise SystemExit("release_comparison.json was built from another run")
    sweeps = {
        key: sweep_section(key, path)
        for key, path in (
            ("main", args.main),
            ("shot_noise", args.shot_noise),
            ("scaling", args.scaling),
            ("photonic", args.photonic),
        )
    }
    check_curated(
        sweeps,
        curated["statistics"],
        curated["photonic_comparison"],
        curated["fair_comparison"],
    )
    report = {
        "paper_items": paper_items(
            curated["release_comparison"],
            load_release(args.release_run),
            arrays,
            curated["statistics"],
        ),
        "photonic_items": photonic_items(
            curated["photonic_comparison"], curated["fair_comparison"]
        ),
        "hardware": hardware(curated["photonic_comparison"]),
        "merlin_pin": merlin_pin(),
        "sweeps": sweeps,
    }
    outputs = {
        "result_tables.json": json.dumps(_jsonable(report), indent=2),
        "result_tables.md": to_markdown(report),
    }
    for s in sweeps.values():
        print(
            f"{s['sweep_id']}: {s['coverage']} {s['completed_runs']}/{s['expected_runs']}"
        )
    if args.check:
        differing = [
            name
            for name, text in outputs.items()
            if (RESULTS_DIR / name).read_text() != text
        ]
        for name in outputs:
            print(
                f"results/{name}: {'DIFFERENT' if name in differing else 'IDENTICAL'}"
            )
        return 1 if differing else 0
    for name, text in outputs.items():
        (RESULTS_DIR / name).write_text(text)
        print(RESULTS_DIR / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
