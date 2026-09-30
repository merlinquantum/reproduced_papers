#!/usr/bin/env python3
"""Derive the reproduction's result tables and figure from a run's metrics.

Every number is read from the structured ``metrics.json`` of a run; nothing is
parsed out of logs and nothing is retyped by hand.

Usage
-----
``python utils/make_report.py [--run-dir outdir/run_YYYYMMDD-HHMMSS]
[--results-dir results]``

Outputs, under ``results/``:

- ``mutation_scores.csv``: per-operator table for both families, including the
  mean fraction of suite samples whose label flips (a graded alternative to the
  paper's binary killed / survived verdict)
- ``comparison.csv``: the reproduction against the paper's Iris/ZFM/RA row
- ``accuracy.csv``: original model and classical baselines
- ``mutation_scores.png``: per-operator mutation scores

Colours are slots 1 and 2 of the validated default categorical palette of the
``dataviz`` skill (``references/palette.md``), used in fixed order and
unmodified. Identity is never colour-alone: every bar carries its operator
label and a legend is present.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
for _path in (PROJECT_DIR.parents[1], PROJECT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lib.experiment_logging import configure_logging  # noqa: E402

logger = logging.getLogger("report")

# Validated default categorical palette, slots 1 and 2, light surface.
SERIES_NEW = "#2a78d6"
SERIES_PRIOR = "#eb6834"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#e3e3e0"

# Paper values for Iris / ZFeatureMap / RealAmplitudes (Tables I and II).
PAPER_PER_OPERATOR = {
    "APC": (139, 240),
    "DFC": (162, 240),
    "APGC": (729, 960),
    "LS": (24, 36),
    "ILS": (31, 36),
    "ALA": (29, 36),
    "ALD": (20, 36),
    "ADD": (4808, 13500),
    "DELETE": (98, 300),
    "CHANGE": (283, 648),
}
NEW_ORDER = ("APC", "DFC", "APGC", "LS", "ILS", "ALA", "ALD")
PRIOR_ORDER = ("ADD", "DELETE", "CHANGE")


def _latest_run_dir() -> Path:
    """Return the most recent run directory containing a full-pipeline metrics file.

    Returns
    -------
    Path
        Run directory.

    Raises
    ------
    FileNotFoundError
        If no full-pipeline run exists.
    """

    candidates = sorted(
        (PROJECT_DIR / "outdir").glob("run_*"), key=lambda p: p.name, reverse=True
    )
    for candidate in candidates:
        metrics_path = candidate / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "mutation_testing" in metrics:
            return candidate
    raise FileNotFoundError(
        "No run directory with mutation-testing metrics found under outdir/."
    )


def read_kill_strength(run_dir: Path) -> dict[str, dict]:
    """Return the graded kill statistics per operator, from the per-mutant CSVs.

    The paper's mutation score is binary: a mutant is killed as soon as one
    suite sample changes label. On a low-margin photonic readout that
    saturates, so the *fraction* of suite samples a mutant flips is also
    computed, under the analytic protocol. It is derived from the per-mutant
    records written by the run, not recomputed from any model.

    Parameters
    ----------
    run_dir : Path
        Run directory containing ``mutants_<family>_seed<N>.csv``.

    Returns
    -------
    dict[str, dict]
        Operator -> ``{"mean": float, "std": float, "n_seeds": int}`` where the
        value is the mean fraction of suite samples whose label changed.
    """

    per_operator_per_seed: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in sorted(run_dir.glob("mutants_*_seed*.csv")):
        seed = path.stem.split("seed")[-1]
        with path.open(encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                if record["status"] == "incompetent":
                    continue
                n_suite = float(record["n_suite_analytic"])
                if n_suite == 0:
                    continue
                per_operator_per_seed[record["operator"]][seed].append(
                    float(record["n_changed_analytic"]) / n_suite
                )

    summary: dict[str, dict] = {}
    for operator, by_seed in per_operator_per_seed.items():
        seed_means = [statistics.fmean(values) for values in by_seed.values() if values]
        if not seed_means:
            continue
        summary[operator] = {
            "mean": statistics.fmean(seed_means),
            "std": statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0,
            "n_seeds": len(seed_means),
        }
    logger.info(
        "KILL_STRENGTH_READ | operators=%d | seeds=%s",
        len(summary),
        {op: row["n_seeds"] for op, row in summary.items()},
    )
    return summary


def _fmt(value, digits: int = 4) -> str:
    """Format a float or ``None`` for CSV output."""

    return "" if value is None else f"{float(value):.{digits}f}"


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """Write a CSV file with the given header and rows."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info("ARTIFACT_WRITTEN | path=%s | rows=%d", path.name, len(rows))


def write_operator_table(
    metrics: dict, results_dir: Path, kill_strength: dict[str, dict]
) -> list[dict]:
    """Write the per-operator results table.

    Parameters
    ----------
    metrics : dict
        Parsed ``metrics.json``.
    results_dir : Path
        Directory receiving the CSV.
    kill_strength : dict[str, dict]
        Output of :func:`read_kill_strength`.

    Returns
    -------
    list[dict]
        The rows, reused by the figure.
    """

    mutation = metrics["mutation_testing"]
    rows: list[dict] = []
    for family_key, family_label, order in (
        ("new_operators", "new", NEW_ORDER),
        ("prior_operators", "prior", PRIOR_ORDER),
    ):
        family = mutation.get(family_key)
        if family is None:
            continue
        for operator in order:
            row = family["per_operator"].get(operator)
            if row is None:
                continue
            paper_killed, paper_total = PAPER_PER_OPERATOR.get(operator, (None, None))
            rows.append(
                {
                    "family": family_label,
                    "operator": operator,
                    "generated": row["generated"],
                    "distinct_circuits": row["distinct_circuits"],
                    "suppressed_redundant": row["suppressed_redundant"],
                    "killed_mean": row["killed"]["mean"],
                    "killed_std": row["killed"]["std"],
                    "incompetent_mean": row["incompetent"]["mean"],
                    "ms_shots_mean": row["mutation_score"]["mean"],
                    "ms_shots_std": row["mutation_score"]["std"],
                    "ms_analytic_mean": row["mutation_score_analytic"]["mean"],
                    "ms_analytic_std": row["mutation_score_analytic"]["std"],
                    "distinct_behaviours_mean": row["distinct_behaviours"]["mean"],
                    "gen_time_per_mutant_s": row["generation_time_per_mutant_s"][
                        "mean"
                    ],
                    "eval_time_per_mutant_s": row["evaluation_time_per_mutant_s"][
                        "mean"
                    ],
                    "paper_killed": paper_killed,
                    "paper_total": paper_total,
                    "paper_ms": (paper_killed / paper_total if paper_total else None),
                    "flip_fraction_mean": kill_strength.get(operator, {}).get("mean"),
                    "flip_fraction_std": kill_strength.get(operator, {}).get("std"),
                }
            )

    _write_csv(
        results_dir / "mutation_scores.csv",
        [
            "family",
            "operator",
            "generated",
            "distinct_circuits",
            "suppressed_redundant",
            "killed_mean",
            "killed_std",
            "incompetent_mean",
            "ms_shots_mean",
            "ms_shots_std",
            "ms_analytic_mean",
            "ms_analytic_std",
            "flip_fraction_mean",
            "flip_fraction_std",
            "distinct_behaviours_mean",
            "gen_time_per_mutant_s",
            "eval_time_per_mutant_s",
            "paper_killed",
            "paper_total",
            "paper_ms",
        ],
        [
            [
                row["family"],
                row["operator"],
                row["generated"],
                row["distinct_circuits"],
                row["suppressed_redundant"],
                _fmt(row["killed_mean"], 2),
                _fmt(row["killed_std"], 2),
                _fmt(row["incompetent_mean"], 2),
                _fmt(row["ms_shots_mean"]),
                _fmt(row["ms_shots_std"]),
                _fmt(row["ms_analytic_mean"]),
                _fmt(row["ms_analytic_std"]),
                _fmt(row["flip_fraction_mean"]),
                _fmt(row["flip_fraction_std"]),
                _fmt(row["distinct_behaviours_mean"], 2),
                _fmt(row["gen_time_per_mutant_s"], 6),
                _fmt(row["eval_time_per_mutant_s"], 6),
                row["paper_killed"] if row["paper_killed"] is not None else "",
                row["paper_total"] if row["paper_total"] is not None else "",
                _fmt(row["paper_ms"]),
            ]
            for row in rows
        ],
    )
    return rows


def write_comparison_table(metrics: dict, results_dir: Path) -> None:
    """Write the reproduction-versus-paper comparison table."""

    mutation = metrics["mutation_testing"]
    comparison = mutation.get("comparison", {})
    control = (
        mutation.get("shot_noise_control", {})
        .get("per_operator", {})
        .get("CONTROL", {})
    )
    rows = [
        [
            "mutants generated, new operators",
            comparison.get("mutants_new", ""),
            comparison.get("paper_mutants_new", ""),
        ],
        [
            "mutants generated, prior operators",
            comparison.get("mutants_prior", ""),
            comparison.get("paper_mutants_prior", ""),
        ],
        [
            "mutant reduction factor (prior / new)",
            _fmt(comparison.get("mutant_reduction_factor"), 2),
            _fmt(comparison.get("paper_reduction_factor"), 2),
        ],
        [
            "mutation score, new operators (analytic)",
            _fmt(mutation["new_operators"]["mutation_score_analytic"]["mean"]),
            _fmt(comparison.get("paper_mutation_score_new")),
        ],
        [
            "mutation score, prior operators (analytic)",
            _fmt(
                mutation.get("prior_operators", {})
                .get("mutation_score_analytic", {})
                .get("mean")
            ),
            _fmt(comparison.get("paper_mutation_score_prior")),
        ],
        [
            "mutation score, new operators (1024 shots)",
            _fmt(mutation["new_operators"]["mutation_score"]["mean"]),
            "n/a (paper reports one shot-based number)",
        ],
        [
            "mutation score, prior operators (1024 shots)",
            _fmt(
                mutation.get("prior_operators", {})
                .get("mutation_score", {})
                .get("mean")
            ),
            "n/a",
        ],
        [
            "false-kill rate of the null control (1024 shots)",
            _fmt(control.get("mutation_score", {}).get("mean")),
            "not reported",
        ],
        [
            "false-kill rate of the null control (analytic)",
            _fmt(control.get("mutation_score_analytic", {}).get("mean")),
            "not reported",
        ],
        [
            "test-suite size",
            _fmt(mutation.get("suite_size", {}).get("mean"), 2),
            "not reported (20 candidate samples)",
        ],
        [
            "generation time per mutant, new operators (s)",
            _fmt(mutation["new_operators"]["generation_time_per_mutant_s"]["mean"], 6),
            "0.000250",
        ],
        [
            "generation time per mutant, prior operators (s)",
            _fmt(
                mutation.get("prior_operators", {})
                .get("generation_time_per_mutant_s", {})
                .get("mean"),
                6,
            ),
            "0.000320",
        ],
        [
            "evaluation time per mutant, new operators (s)",
            _fmt(mutation["new_operators"]["evaluation_time_per_mutant_s"]["mean"], 6),
            "not directly comparable",
        ],
    ]
    _write_csv(
        results_dir / "comparison.csv",
        ["quantity", "reproduction (photonic MerLin)", "paper (Iris / ZFM / RA)"],
        rows,
    )


def write_accuracy_table(metrics: dict, results_dir: Path) -> None:
    """Write the original-model and classical-baseline accuracy table."""

    aggregate = metrics["aggregate"]
    budget = metrics["parameter_budget"]
    rows = [
        [
            "photonic QNN (this reproduction)",
            budget["photonic_trainable_phases"],
            _fmt(aggregate["test_accuracy"]["mean"]),
            _fmt(aggregate["test_accuracy"]["std"]),
            _fmt(aggregate["test_accuracy"]["min"]),
            _fmt(aggregate["test_accuracy"]["max"]),
        ]
    ]
    for name, row in metrics.get("classical_baselines", {}).items():
        rows.append(
            [
                name,
                row["n_parameters"],
                _fmt(row["test_accuracy_mean"]),
                "",
                _fmt(row["test_accuracy_min"]),
                _fmt(row["test_accuracy_max"]),
            ]
        )
    _write_csv(
        results_dir / "accuracy.csv",
        ["model", "n_parameters", "test_acc_mean", "test_acc_std", "min", "max"],
        rows,
    )


def plot_mutation_scores(rows: list[dict], metrics: dict, results_dir: Path) -> None:
    """Plot per-operator results as two small multiples sharing the operator axis.

    The data's job is comparing a magnitude across ten named categories with
    long labels, so horizontal bars are the form. Two panels rather than two
    x-scales on one panel: the paper's binary mutation score and the graded
    flip fraction are different measures and must never share an axis.
    """

    ordered = [row for row in rows if row["family"] == "new"] + [
        row for row in rows if row["family"] == "prior"
    ]
    labels = [row["operator"] for row in ordered]
    colours = [
        SERIES_NEW if row["family"] == "new" else SERIES_PRIOR for row in ordered
    ]
    positions = list(range(len(ordered)))

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(10.4, 5.2), sharey=True, facecolor=SURFACE
    )

    panels = (
        (
            left,
            [row["ms_analytic_mean"] or 0.0 for row in ordered],
            [row["ms_analytic_std"] or 0.0 for row in ordered],
            [row["paper_ms"] for row in ordered],
            "mutation score (analytic protocol)",
            "a. Mutation score, the paper's binary verdict",
        ),
        (
            right,
            [row["flip_fraction_mean"] or 0.0 for row in ordered],
            [row["flip_fraction_std"] or 0.0 for row in ordered],
            None,
            "mean fraction of suite samples flipped",
            "b. Graded kill strength",
        ),
    )

    for axes, values, errors, reference, xlabel, title in panels:
        axes.set_facecolor(SURFACE)
        axes.barh(
            positions,
            values,
            xerr=errors,
            height=0.62,
            color=colours,
            edgecolor=SURFACE,
            linewidth=1.5,
            error_kw={"ecolor": INK_MUTED, "elinewidth": 1.1, "capsize": 3},
            zorder=3,
        )
        if reference is not None:
            visible = [
                (position, value)
                for position, value in zip(positions, reference, strict=True)
                if value is not None
            ]
            axes.scatter(
                [value for _, value in visible],
                [position for position, _ in visible],
                marker="D",
                s=40,
                facecolor=SURFACE,
                edgecolor=INK_PRIMARY,
                linewidth=1.4,
                zorder=4,
            )
        for position, value in zip(positions, values, strict=True):
            axes.text(
                value + 0.02,
                position,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color=INK_PRIMARY,
            )
        axes.set_xlim(0.0, 1.2)
        axes.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axes.set_xlabel(xlabel, fontsize=9.5, color=INK_PRIMARY)
        axes.set_title(title, fontsize=10, color=INK_PRIMARY, loc="left", pad=8)
        axes.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        axes.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            axes.spines[spine].set_visible(False)
        axes.spines["bottom"].set_color(GRID)
        axes.tick_params(colors=INK_MUTED, length=0)

    left.set_yticks(positions)
    left.set_yticklabels(labels, fontsize=10, color=INK_PRIMARY)
    left.invert_yaxis()

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SERIES_NEW),
        plt.Rectangle((0, 0), 1, 1, color=SERIES_PRIOR),
        plt.Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markerfacecolor=SURFACE,
            markeredgecolor=INK_PRIMARY,
            markersize=7,
        ),
    ]
    left.legend(
        handles,
        [
            "seven new operators (photonic)",
            "prior operators (photonic)",
            "paper value (gate model)",
        ],
        loc="lower right",
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_PRIMARY,
    )

    mutation = metrics["mutation_testing"]
    suite = mutation.get("suite_size", {}).get("mean") or 0.0
    control = (
        mutation.get("shot_noise_control", {})
        .get("per_operator", {})
        .get("CONTROL", {})
    )
    control_rate = control.get("mutation_score", {}).get("mean")
    control_text = f"{control_rate:.0%}" if control_rate is not None else "not measured"
    figure.suptitle(
        "Photonic MerLin adaptation of the QML mutation-testing framework",
        fontsize=12.5,
        color=INK_PRIMARY,
        x=0.006,
        ha="left",
        y=0.985,
    )
    figure.text(
        0.006,
        0.006,
        "Iris, 4 modes / 2 photons / 6 ansatz layers, 24 trainable phases; mean +/- std over 3 seeds; "
        f"test suite {suite:.1f} of 20 candidate samples.\n"
        f"Shot-noise control: an unmutated copy of the circuit is killed {control_text}; "
        "the analytic protocol is therefore the one plotted.\n"
        "Photonic ADAPTATION, not a gate-model reproduction: each operator's mutant inventory depends on the "
        "available component set, so the paper diamonds are context, not a target.",
        fontsize=7.5,
        color=INK_MUTED,
        ha="left",
        va="bottom",
    )
    figure.tight_layout(rect=(0, 0.115, 1, 0.955))
    output = results_dir / "mutation_scores.png"
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    logger.info("ARTIFACT_WRITTEN | path=%s", output.name)


def main() -> int:
    """Build every result artifact for one run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_run_dir()
    if not run_dir.is_absolute():
        run_dir = PROJECT_DIR / run_dir
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(results_dir / "report.log")

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if "mutation_testing" not in metrics:
        raise ValueError(f"{run_dir} has no mutation-testing metrics.")
    logger.info("REPORT_STARTED | run_dir=%s | run_id=%s", run_dir, metrics["run_id"])

    kill_strength = read_kill_strength(run_dir)
    rows = write_operator_table(metrics, results_dir, kill_strength)
    write_comparison_table(metrics, results_dir)
    write_accuracy_table(metrics, results_dir)
    plot_mutation_scores(rows, metrics, results_dir)

    (results_dir / "source_run.txt").write_text(
        f"{run_dir.relative_to(PROJECT_DIR)}\nrun_id={metrics['run_id']}\n",
        encoding="utf-8",
    )
    logger.info("REPORT_COMPLETED | results_dir=%s", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
